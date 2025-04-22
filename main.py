from google import genai
from google.genai import types
from google.genai import errors as genai_errors

from PIL import Image
import os
import json
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryError
from typing import List, Dict, Tuple, Optional, Any
import time
import mimetypes
from ratelimiter import RateLimiter

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("annotation.log")
    ]
)
logger = logging.getLogger(__name__)

# --- Constants ---
SUPPORTED_IMAGE_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
DEFAULT_MAX_WORKERS = 4
RETRY_ATTEMPTS = 3
RETRY_WAIT_MULTIPLIER = 10
RETRY_WAIT_MAX = 40
API_RPM_LIMIT = 15 # <-- Requests per minute limit
API_REQUEST_PERIOD = 60 # seconds (1 minute)

RETRYABLE_EXCEPTIONS = (
    genai_errors.ServerError,
    genai_errors.ClientError,
    TimeoutError,                       # General timeout
    genai_errors.APIError,
)
PYTHON_MIN_VERSION = (3, 9)

# --- Helper Functions ---

def check_python_version():
    """Checks the minimum required Python version."""
    import sys
    if sys.version_info < PYTHON_MIN_VERSION:
        major, minor = PYTHON_MIN_VERSION
        logger.critical(
            f"This script requires Python {major}.{minor} or higher. "
            f"Current version: {sys.version.split()[0]}"
        )
        sys.exit(1)

def get_mime_type(filename: str) -> str:
    """Guesses the MIME type from the filename."""
    mime_type, _ = mimetypes.guess_type(filename)
    if mime_type and mime_type.startswith("image/"):
        return mime_type
    ext = os.path.splitext(filename)[1].lower()
    if ext in ('.jpg', '.jpeg'):
        return 'image/jpeg'
    elif ext == '.png':
        return 'image/png'
    elif ext == '.webp':
        return 'image/webp'
    elif ext == '.bmp':
        return 'image/bmp'
    logger.warning(f"Could not guess MIME type for: {filename}. Using 'application/octet-stream'.")
    return 'application/octet-stream'

def get_image_dimensions(image_path: str) -> Optional[Tuple[int, int]]:
    """Returns the width and height of the given image."""
    try:
        with Image.open(image_path) as img:
            img.verify() # Verify headers without loading full image data
        with Image.open(image_path) as img:
            return img.width, img.height
    except FileNotFoundError:
        logger.error(f"Image not found: {image_path}")
        return None
    except (IOError, SyntaxError, Image.UnidentifiedImageError) as e:
         logger.error(f"Could not read or invalid image file ({image_path}): {e}")
         return None
    except Exception as e:
        logger.error(f"Error getting image dimensions ({image_path}): {e}", exc_info=True)
        return None

def load_class_list(filepath: str) -> List[str]:
    """Reads the class list from the given file (one class per line)."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f if line.strip()]
        if not classes:
            logger.error(f"Class list file ({filepath}) is empty or contains no valid classes.")
            raise ValueError("Class list is empty.")
        logger.info(f"{len(classes)} classes loaded: {', '.join(classes)}")
        return classes
    except FileNotFoundError:
        logger.error(f"Class list file not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error reading class list file ({filepath}): {e}", exc_info=True)
        raise

def create_class_mapping(class_list: List[str]) -> Dict[str, int]:
    """Creates a class name -> class ID mapping from the class list."""
    return {class_name.lower(): i for i, class_name in enumerate(class_list)}

def convert_gemini_to_yolo(
    gemini_annotations: List[Dict[str, Any]],
    class_mapping: Dict[str, int],
    image_width: int,
    image_height: int,
    image_path: str # For logging
) -> List[str]:
    """
    Converts annotations received from the Gemini API to YOLO format.
    Expects Gemini's [ymin, xmin, ymax, xmax] (normalized 0-1000) format.
    """
    yolo_annotations = []
    if not isinstance(gemini_annotations, list):
         logger.warning(f"Unexpected Gemini output format (not a list): {gemini_annotations} - Image: {image_path}")
         return []

    for ann_index, annotation in enumerate(gemini_annotations):
        if not isinstance(annotation, dict):
            logger.warning(f"Unexpected annotation format (not a dictionary): {annotation} at index {ann_index} - Image: {image_path}")
            continue

        try:
            # Case-insensitive key matching
            box_key = next((k for k in annotation if k.lower() == "box_2d"), None)
            label_key = next((k for k in annotation if k.lower() == "label"), None)

            if not box_key or not label_key:
                logger.warning(f"Missing 'box_2d' or 'label' key: {annotation} - Image: {image_path}")
                continue

            box = annotation[box_key]
            label = str(annotation[label_key]).lower() # Convert label to lower case for mapping

            if not isinstance(box, list) or len(box) != 4:
                logger.warning(f"Invalid 'box_2d' format: {box} - Image: {image_path}")
                continue

            if label not in class_mapping:
                logger.debug(f"Class '{label}' not found in class mapping. Skipping. Image: {image_path}")
                continue

            class_id = class_mapping[label]

            try:
                ymin, xmin, ymax, xmax = map(float, box)
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid coordinate value: {box} ({e}) - Image: {image_path}")
                continue

            # Validate coordinates are within 0-1000 and logical (min < max)
            if not (0 <= ymin <= 1000 and 0 <= xmin <= 1000 and 0 <= ymax <= 1000 and 0 <= xmax <= 1000 and xmin < xmax and ymin < ymax):
                 logger.warning(f"Invalid or out-of-range coordinate values (outside 0-1000 or min>=max): {box} - Image: {image_path}")
                 continue

            # Convert to YOLO format (center_x, center_y, width, height) normalized 0-1
            x_center = ((xmin + xmax) / 2) / 1000.0
            y_center = ((ymin + ymax) / 2) / 1000.0
            bbox_width = (xmax - xmin) / 1000.0
            bbox_height = (ymax - ymin) / 1000.0

            # Clamp values to [0.0, 1.0] to handle potential minor floating point inaccuracies
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            bbox_width = max(0.0, min(1.0, bbox_width))
            bbox_height = max(0.0, min(1.0, bbox_height))

            yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}"
            yolo_annotations.append(yolo_line)

        except Exception as e:
            logger.error(f"Annotation conversion error ({annotation}): {e} - Image: {image_path}", exc_info=True)
            continue

    return yolo_annotations

# --- Gemini API Interaction ---

@retry(
    stop=stop_after_attempt(RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=RETRY_WAIT_MULTIPLIER, max=RETRY_WAIT_MAX),
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
    before_sleep=lambda retry_state: logger.warning(
        f"API error/limit ({type(retry_state.outcome.exception()).__name__}), retrying... " # <-- Log message updated
        f"(Attempt {retry_state.attempt_number}/{RETRY_ATTEMPTS})"
    )
)
def call_gemini_api(
    client: genai.Client,
    model_name: str,
    image_path: str,
    class_list: List[str],
    limiter: RateLimiter
) -> Optional[List[Dict[str, Any]]]:
    """Calls the Gemini API using client.generate_content and applies rate limiting."""
    try:
        logger.debug(f"Preparing Gemini API call: {image_path}")
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            if not image_data:
                logger.error(f"Image file is empty: {image_path}")
                return None

        mime_type = get_mime_type(image_path)
        image_part = types.Part.from_bytes(data=image_data, mime_type=mime_type)


        prompt = f"""
        Analyze the provided image and identify all objects belonging to the following classes: {', '.join(class_list)}.
        For each detected object, provide its bounding box and label.
        Return the results as a JSON list. Each item in the list should be a JSON object with two keys:
        1. "label": The name of the detected object class (must be one of the provided classes). Case-insensitive matching is acceptable for the label key itself (e.g., "Label" or "label").
        2. "box_2d": A list of four numbers representing the bounding box coordinates [ymin, xmin, ymax, xmax], normalized to a 0-1000 scale. Case-insensitive matching is acceptable for the box key itself (e.g., "Box_2d" or "box_2d").

        Example for a single object:
        [
          {{"label": "cat", "box_2d": [150, 200, 750, 800]}}
        ]
        If no objects from the list are found, return an empty JSON list: [].
        Provide ONLY the JSON list in your response, without any introductory text, explanations, or markdown formatting like ```json ... ```.
        """
        text_part = types.Part.from_text(text=prompt)
        contents = [image_part, text_part]

        generation_config = types.GenerateContentConfig(
            temperature=0.2, # Lower temperature for more deterministic output
            response_mime_type='application/json', # Request JSON output directly
        )

        # --- Rate Limiting ---
        logger.debug(f"Waiting for rate limiter (limit: {limiter.max_calls}/{limiter.period}s)... {image_path}")
        with limiter: # <-- Rate limiter engages here
            logger.debug(f"Rate limiter passed, calling API... {image_path}")
            # API call
            response = client.models.generate_content( # Use the standard client method
                model=f'models/{model_name}',
                contents=contents,
                config=generation_config # Pass config here
                # request_options={"timeout": 60} # Optional timeout
            )
            logger.debug(f"API call completed. {image_path}")
        # --- /Rate Limiting ---


        if not response.candidates:
             logger.warning(f"No valid candidate response received from Gemini. Image: {image_path}. Reason: {response.prompt_feedback}")
             # Consider specific handling for block reasons if needed
             # if response.prompt_feedback.block_reason == ...
             return None # Treat as failure if no candidates

        candidate = response.candidates[0]

        # Check finish reason for potential issues
        if candidate.finish_reason != types.FinishReason.STOP:
             logger.warning(f"Gemini response finished unexpectedly. Reason: {candidate.finish_reason.name}. Image: {image_path}")
             if candidate.finish_reason == types.FinishReason.SAFETY:
                  logger.error(f"Response blocked due to safety reasons. Image: {image_path}. Safety Ratings: {candidate.safety_ratings}")
             # ResourceExhausted (429) error might also come as FINISH_REASON_OTHER, tenacity will handle it
             elif candidate.finish_reason == types.FinishReason.RECITATION:
                  logger.warning(f"Response blocked due to recitation. Image: {image_path}")
             elif candidate.finish_reason == types.FinishReason.OTHER:
                  logger.warning(f"Response finished due to 'OTHER' (could be API limit or another issue). Image: {image_path}")
             # Any non-STOP finish reason is treated as a failure for this annotation task
             return None # Mark as error

        # Validate content exists
        if not candidate.content or not candidate.content.parts:
             logger.warning(f"Content or parts not found in Gemini response. Image: {image_path}")
             # If the model correctly returns nothing (empty list), it should still have content/parts
             # An empty response here likely indicates an issue.
             return [] # Return empty list, assuming no objects found, but log warning

        # Extract text, assuming the first part contains the JSON
        raw_text = candidate.content.parts[0].text.strip()

        # Clean potential markdown formatting ```json ... ```
        if raw_text.startswith("```json"):
            raw_text = raw_text[len("```json"):].strip()
        elif raw_text.startswith("```"):
             raw_text = raw_text[len("```"):].strip() # Handle ``` without json tag
        if raw_text.endswith("```"):
            raw_text = raw_text[:-len("```")].strip()

        # Handle empty string after cleaning (could mean no objects found)
        if not raw_text:
             logger.info(f"Received empty text response from Gemini (after cleaning). Assuming no objects found. Image: {image_path}")
             return []

        logger.debug(f"Gemini Raw Text Response ({image_path}): {raw_text}")

        # Parse the JSON response
        try:
            annotations = json.loads(raw_text)
            # Ensure the result is a list as requested in the prompt
            if not isinstance(annotations, list):
                logger.warning(f"Expected JSON list from Gemini but received a different structure: {type(annotations)} - Image: {image_path}")
                # Attempt to handle if a single object dict was returned instead of a list
                if isinstance(annotations, dict) and "label" in annotations and "box_2d" in annotations:
                     logger.info(f"Converting single object JSON to list. - Image: {image_path}")
                     return [annotations]
                # If it's neither a list nor a single valid object dict, treat as error
                return None
            return annotations
        except json.JSONDecodeError as e:
            logger.error(f"Could not parse Gemini response as JSON. Response: '{raw_text}'. Error: {e} - Image: {image_path}")
            return None # Treat JSON parsing error as failure

    # Handle specific API errors for retrying
    except genai_errors.APIError as e:
        logger.error(f"Gemini API Error ({type(e).__name__}) ({image_path}): {e}", exc_info=False) # Log less detail for retryable errors
        raise e # Re-raise for tenacity to catch
    # Handle file not found before API call
    except FileNotFoundError:
        logger.error(f"Image not found for API call: {image_path}")
        return None # Non-retryable error for this specific image
    # Catch-all for other unexpected errors during the API call process
    except Exception as e:
        logger.error(f"Unexpected error during Gemini API call ({image_path}): {e}", exc_info=True)
        # Only re-raise if it's a type Tenacity should retry
        if not isinstance(e, RETRYABLE_EXCEPTIONS):
             return None # Treat as non-retryable failure for this image
        raise e # Re-raise for tenacity

# --- Main Processing Logic ---
def process_image(
    image_path: str,
    output_dir: str,
    class_list: List[str],
    class_mapping: Dict[str, int],
    client: genai.Client,
    model_name: str,
    limiter: RateLimiter # <-- RateLimiter object added
) -> Tuple[str, bool, Optional[str]]:
    """Processes a single image: API call (rate-limited), conversion, saving."""
    start_time = time.monotonic()
    base_filename = os.path.basename(image_path)
    logger.info(f"Processing: {base_filename}")

    dimensions = get_image_dimensions(image_path)
    if dimensions is None:
        # Error already logged in get_image_dimensions
        return image_path, False, "Could not get image dimensions or file is corrupt."
    img_width, img_height = dimensions

    gemini_annotations = None # Initialize
    try:
        # Pass limiter to the API call
        gemini_annotations = call_gemini_api(client, model_name, image_path, class_list, limiter)
    except RetryError as e:
        # This catches errors after all retries have failed
        logger.error(f"API call failed after all retries ({base_filename}): {e}")
        # gemini_annotations remains None
    except Exception as e:
         # Catch unexpected errors not handled by retry or specific exceptions in call_gemini_api
         logger.error(f"Unhandled error during API call processing ({base_filename}): {e}", exc_info=True)
         # gemini_annotations remains None

    # If API call failed or returned None (indicating an error or blocking)
    if gemini_annotations is None:
        return image_path, False, "Gemini API call failed, returned invalid response, or was blocked."

    # Prepare output path
    annotation_filename = os.path.splitext(base_filename)[0] + ".txt"
    annotation_path = os.path.join(output_dir, annotation_filename)

    # Handle case where API returned an empty list (no objects found)
    if not gemini_annotations:
        logger.info(f"No objects from the specified classes found in image ({base_filename}) or API returned empty.")
        try:
            os.makedirs(output_dir, exist_ok=True)
            # Create an empty file to indicate processing was successful but no objects were found
            with open(annotation_path, "w") as f:
                pass
            logger.info(f"Empty annotation file created: {annotation_filename}")
            processing_time = time.monotonic() - start_time
            return image_path, True, f"No objects found, empty file created ({processing_time:.2f}s)"
        except IOError as e:
            logger.error(f"Could not write empty annotation file ({annotation_path}): {e}")
            return image_path, False, "Could not write empty annotation file."

    # Convert valid Gemini annotations to YOLO format
    yolo_lines = convert_gemini_to_yolo(gemini_annotations, class_mapping, img_width, img_height, image_path)

    # Handle case where conversion resulted in no valid lines (e.g., all labels unknown)
    if not yolo_lines:
        logger.warning(f"Could not generate valid YOLO annotations (post-conversion). Image: {base_filename}")

        try:
            os.makedirs(output_dir, exist_ok=True)
            with open(annotation_path, "w") as f:
                pass
            logger.info(f"Empty annotation file created (no objects post-conversion): {annotation_filename}")
            processing_time = time.monotonic() - start_time
            return image_path, True, f"No objects post-conversion, empty file ({processing_time:.2f}s)"
        except IOError as e:
            logger.error(f"Could not write empty annotation file ({annotation_path}): {e}")
            return image_path, False, "Could not write empty annotation file (post-conversion)."


    # Save the valid YOLO annotations
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(annotation_path, "w", encoding='utf-8') as f:
            for line in yolo_lines:
                f.write(line + "\n")
        processing_time = time.monotonic() - start_time
        logger.info(f"Annotated ({len(yolo_lines)} objects): {annotation_filename} ({processing_time:.2f}s)")
        return image_path, True, f"{len(yolo_lines)} objects annotated ({processing_time:.2f}s)"

    except IOError as e:
        logger.error(f"Could not write annotation file ({annotation_path}): {e}")
        return image_path, False, "Could not write annotation file."
    except Exception as e:
        # Catch any other unexpected error during file writing
        logger.error(f"Unexpected error while saving annotation file ({annotation_path}): {e}", exc_info=True)
        return image_path, False, "Error occurred while saving annotation file."


def process_dataset(
    image_dir: str,
    output_dir: str,
    class_list_path: str,
    api_key: str,
    model_name: str,
    max_workers: int,
    rpm_limit: int = API_RPM_LIMIT # <-- RPM limit added as parameter
):
    """Processes all images in the dataset in parallel (rate-limited)."""
    logger.info(f"Dataset processing started: {image_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model to use: {model_name}")
    logger.info(f"Maximum workers: {max_workers}")
    logger.info(f"API Request Limit: {rpm_limit} RPM") # <-- Log added

    if not os.path.isdir(image_dir):
        logger.error(f"Image directory not found or is not a directory: {image_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    try:
        class_list = load_class_list(class_list_path)
        class_mapping = create_class_mapping(class_list)
    except Exception:
        # Error logged in load_class_list
        logger.error("Could not load class list, stopping process.")
        return

    try:
        client = genai.Client(api_key=api_key)
        logger.info(f"Gemini Client/Model ({model_name}) configured successfully.")
    except Exception as e:
        logger.error(f"Could not configure Gemini Client/Model: {e}", exc_info=True)
        return

    # --- Rate Limiter Creation ---
    # RateLimiter(max_calls, period) allows max_calls per period seconds.
    # We want rpm_limit per minute (60 seconds).
    limiter = RateLimiter(max_calls=rpm_limit, period=API_REQUEST_PERIOD)
    logger.info(f"Rate Limiter created: {rpm_limit} calls / {API_REQUEST_PERIOD} seconds")
    # --- /Rate Limiter Creation ---

    # Find all supported image files
    image_files = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if os.path.isfile(os.path.join(image_dir, f)) and f.lower().endswith(SUPPORTED_IMAGE_FORMATS)
    ]

    if not image_files:
        logger.warning(f"No supported image formats found in the image directory ({image_dir}).")
        return

    logger.info(f"Total {len(image_files)} images to process.")

    processed_count = 0
    success_count = 0
    failed_count = 0
    start_time_total = time.monotonic()

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='Annotator') as executor:
        # Pass the limiter when submitting futures
        futures = {
            # Pass the already configured client/model object
            executor.submit(process_image, img_path, output_dir, class_list, class_mapping, client, model_name, limiter): img_path
            for img_path in image_files
        }

        # Process results as they complete
        for future in as_completed(futures):
            img_path = futures[future]
            base_filename = os.path.basename(img_path)
            processed_count += 1
            try:
                # Get result from the future
                _, success, message = future.result()
                if success:
                    success_count += 1
                    # Optional: Log success message if needed, but can be verbose
                    # logger.debug(f"Success: {base_filename} - {message}")
                else:
                    failed_count += 1
                    # Log failure reason clearly
                    logger.warning(f"Failed: {base_filename} - Reason: {message}")

                # Log progress periodically
                if processed_count % 20 == 0 or processed_count == len(image_files):
                     elapsed_time = time.monotonic() - start_time_total
                     rate = processed_count / elapsed_time if elapsed_time > 0 else 0
                     logger.info(
                         f"Progress: {processed_count}/{len(image_files)} [{success_count}✓, {failed_count}✗] "
                         f"({elapsed_time:.1f}s, {rate:.2f} img/s)"
                     )

            except Exception as exc:
                # Catch errors that might occur retrieving the result itself
                failed_count += 1
                logger.error(f"Unexpected task error while processing image ({base_filename}): {exc}", exc_info=True)

    # Log summary after all images are processed
    end_time_total = time.monotonic()
    total_duration = end_time_total - start_time_total
    logger.info("-" * 40)
    logger.info("Processing Complete!")
    logger.info(f"Total Images: {len(image_files)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info(f"Total Duration: {total_duration:.2f} seconds")
    if processed_count > 0 : logger.info(f"Average Time/Image: {total_duration / processed_count:.2f} seconds")
    logger.info("-" * 40)

# --- Argument Parser and Main Execution Block ---
def main():
    check_python_version()

    parser = argparse.ArgumentParser(
        description=f"Annotates an image dataset in YOLO format using the Gemini API (google-genai >= 1.11.0, rate-limited). Requires Python >= {PYTHON_MIN_VERSION[0]}.{PYTHON_MIN_VERSION[1]}.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("image_dir", help="Path to the directory containing images to be annotated.")
    parser.add_argument("output_dir", help="Directory where the generated YOLO annotation (.txt) files will be saved.")
    parser.add_argument("class_list_path", help="Path to the .txt file containing one class name per line.")
    parser.add_argument("--api_key", help="Google Gemini API key. If not specified, it reads from the GOOGLE_API_KEY environment variable.", default=None)
    parser.add_argument("--model_name", help="Gemini model to use.", default=DEFAULT_GEMINI_MODEL)
    parser.add_argument("--max_workers", type=int, help="Maximum number of worker threads for parallel processing.", default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--rpm", type=int, help="Maximum API requests per minute limit.", default=API_RPM_LIMIT) # <-- RPM argument
    parser.add_argument("--log_level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO', help="Logging level to set.")

    args = parser.parse_args()

    # Set logging level based on argument
    logger.setLevel(args.log_level.upper())

    # Get API key from argument or environment variable
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.critical("Google Gemini API key was not provided either via --api_key argument or GOOGLE_API_KEY environment variable. Stopping.")
        return 1 # Indicate error

    # Pass RPM limit to process_dataset
    process_dataset(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        class_list_path=args.class_list_path,
        api_key=api_key,
        model_name=args.model_name,
        max_workers=args.max_workers,
        rpm_limit=args.rpm
    )
    return 0 # Indicate success

# --- Script Execution (Example) ---
# The following section is for direct execution, taking arguments from here.
# To run from the command line, remove this section and use `if __name__ == "__main__": main()`

# Example Usage (For direct execution - instead of command line)
API_KEY='YOUR_API_KEY_HERE' # <-- ENTER YOUR OWN API KEY HERE
IMAGE_DIR='images' # <-- Your image directory
OUTPUT_DIR='output' # <-- Your output directory
CLASS_LIST_PATH='class_list.txt' # <-- Your class list file
MODEL_NAME='gemini-2.0-flash' # <-- Model name (e.g., 1.5-flash or higher)
MAX_WORKERS=4 # <-- Number of workers (mind the RPM limit!)
RPM_LIMIT_VALUE = 15 # <-- Requests per minute limit (e.g., 15 for Free Tier)

if __name__ == "__main__":
    # Check API Key
    api_key_to_use = API_KEY
    if not api_key_to_use or api_key_to_use == 'YOUR_API_KEY_HERE':
         env_key = os.environ.get("GOOGLE_API_KEY")
         if env_key:
              api_key_to_use = env_key
              logger.info("Using API key from GOOGLE_API_KEY environment variable.")
         else:
              logger.critical("Please set the API_KEY variable in the script or the GOOGLE_API_KEY environment variable.")
              exit(1) # Exit with error code

    # Set log level to INFO (or DEBUG if desired)
    logger.setLevel(logging.INFO)

    # Check Python version
    check_python_version()

    # Start the main process
    process_dataset(
        image_dir=IMAGE_DIR,
        output_dir=OUTPUT_DIR,
        class_list_path=CLASS_LIST_PATH,
        api_key=api_key_to_use,
        model_name=MODEL_NAME,
        max_workers=MAX_WORKERS,
        rpm_limit=RPM_LIMIT_VALUE
    )