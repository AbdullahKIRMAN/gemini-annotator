# -*- coding: utf-8 -*-
import io

from dotenv import load_dotenv

load_dotenv()
from google import genai
from google.genai import types
from google.genai import errors as genai_errors

from PIL import Image
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
API_RPM_LIMIT = 15  # Requests per minute limit (adjust based on your tier)
API_REQUEST_PERIOD = 60  # seconds (1 minute)
TARGET_SQUARE_SIZE = (1280, 1280)

RETRYABLE_EXCEPTIONS = (
    genai_errors.ServerError,
    genai_errors.ClientError,
    TimeoutError,  # General timeout
    genai_errors.APIError,
)
PYTHON_MIN_VERSION = (3, 9)


# --- Helper Functions ---

def check_python_version():
    """Checks the minimum required Python version."""
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
            img.verify()  # Verify headers without loading full image data
        # Re-open after verify to get dimensions
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


def resize_and_pad_to_square(image: Image.Image, target_size: Tuple[int, int], fill_color: Tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """
    Resmi, en-boy oranını koruyarak hedef boyuta sığdırır ve boşlukları siyahla doldurur.
    """
    original_width, original_height = image.size
    ratio = min(target_size[0] / original_width, target_size[1] / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)

    # Yüksek kaliteli yeniden boyutlandırma için LANCZOS filtresi kullanılır
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Yeni bir kare tuval oluştur ve resmi ortasına yapıştır
    padded_image = Image.new("RGB", target_size, fill_color)
    paste_x = (target_size[0] - new_width) // 2
    paste_y = (target_size[1] - new_height) // 2
    padded_image.paste(resized_image, (paste_x, paste_y))

    return padded_image


def convert_gemini_to_yolo(gemini_annotations: List[Dict[str, Any]], class_mapping: Dict[str, int], image_path: str) -> List[str]:
    """
    Kare bir resimden gelen Gemini etiketlerini doğrudan YOLO formatına dönüştürür.
    Artık karmaşık en-boy oranı matematiğine gerek yok.
    """
    yolo_annotations = []
    if not isinstance(gemini_annotations, list):
        logger.warning(f"Unexpected Gemini output format (not a list): {gemini_annotations} - Image: {image_path}")
        return []

    for annotation in gemini_annotations:
        try:
            box_key = next((k for k in annotation if k.lower() == "box_2d"), None)
            label_key = next((k for k in annotation if k.lower() == "label"), None)
            if not box_key or not label_key: continue

            box = annotation[box_key]
            label = str(annotation[label_key]).lower()
            if label not in class_mapping: continue
            if not isinstance(box, list) or len(box) != 4: continue

            class_id = class_mapping[label]
            ymin, xmin, ymax, xmax = map(float, box)

            # Doğrudan ve basit dönüşüm (artık karmaşık matematik yok)
            x_center = ((xmin + xmax) / 2) / 1000.0
            y_center = ((ymin + ymax) / 2) / 1000.0
            width = (xmax - xmin) / 1000.0
            height = (ymax - ymin) / 1000.0

            # Güvenlik kontrolü
            if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0 and width > 0 and height > 0):
                logger.warning(f"Hesaplama sonrası geçersiz koordinat. Kutu: {box}, Resim: {image_path}. Atlanıyor.")
                continue

            yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            yolo_annotations.append(yolo_line)

        except Exception as e:
            logger.error(f"Anotasyon dönüşüm hatası ({annotation}): {e} - Resim: {image_path}", exc_info=True)
            continue

    return yolo_annotations


# --- Gemini API Interaction ---

# Custom retry logic to log specific API errors
def should_retry_api_call(exception):
    """Determine if we should retry based on the exception type."""
    is_retryable = isinstance(exception, RETRYABLE_EXCEPTIONS)
    if is_retryable:
        logger.warning(f"Detected retryable API error: {type(exception).__name__} - {exception}")
    else:
        logger.error(f"Detected non-retryable error: {type(exception).__name__} - {exception}")
    return is_retryable


@retry(
    stop=stop_after_attempt(RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=RETRY_WAIT_MULTIPLIER, min=4, max=RETRY_WAIT_MAX),  # Added min wait
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),  # Use tuple directly
    # retry=retry_if_exception(should_retry_api_call), # Use custom function if more logic needed
    before_sleep=lambda retry_state: logger.warning(
        f"Retrying API call for {retry_state.args[2] if len(retry_state.args) > 2 else 'unknown image'}... "  # Log image path if possible
        f"(Attempt {retry_state.attempt_number}/{RETRY_ATTEMPTS}, waiting {retry_state.idle_for:.2f}s). "
        f"Reason: {type(retry_state.outcome.exception()).__name__}"
    )
)
def call_gemini_api(
        client: genai.Client,
        model_name: str,
        image_bytes: bytes,
        class_list: List[str],
        limiter: RateLimiter,
        image_path_for_logging: str
) -> Optional[List[Dict[str, Any]]]:
    """Calls the Gemini API using image data in memory (bytes)."""
    try:
        # DEĞİŞİKLİK: Dosya okuma kısmı kaldırıldı, doğrudan byte'lar kullanılıyor
        logger.debug(f"Preparing Gemini API call for: {image_path_for_logging}")
        if not image_bytes:
            logger.error(f"Image data is empty for: {image_path_for_logging}")
            return None

        # MIME type'ı sabit olarak 'image/jpeg' yapabiliriz çünkü hep o formatta göndereceğiz
        image_part = types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg')

        filename_hint = os.path.basename(image_path_for_logging)

        prompt = f"""
        Analyze the provided image and identify all objects belonging ONLY to the following classes: {', '.join(class_list)}.
        For each detected object, provide its bounding box and label.
        Return the results as a valid JSON list. Each item in the list must be a JSON object with exactly two keys:
        1. "label": The name of the detected object class (must be one of the provided classes, case-insensitive matching is okay for the value but the key must be 'label').
        2. "box_2d": A list of four numbers representing the bounding box coordinates [ymin, xmin, ymax, xmax], normalized to a 0-1000 scale (integers or floats are acceptable). The key must be 'box_2d'.

        Example for a single object:
        [
          {{"label": "cat", "box_2d": [150, 200, 750, 800]}}
        ]
        Example for multiple objects:
        [
          {{"label": "cat", "box_2d": [150, 200, 750, 800]}},
          {{"label": "dog", "box_2d": [50, 50, 400, 350]}}
        ]

        If no objects from the specified class list are found in the image, return an empty JSON list: [].
        Provide ONLY the JSON list in your response. Do not include any introductory text, explanations, code block markers (like ```json ... ```), or any other text outside the JSON list itself. The entire response must be parseable as JSON.
        """
        text_part = types.Part.from_text(text=prompt)
        contents = [image_part, text_part]

        generation_config = types.GenerateContentConfig(
            temperature=0.1,  # lower temperature for consistency
            response_mime_type='application/json',  # Request JSON output directly
        )

        # --- Rate Limiting ---
        logger.debug(f"Waiting for rate limiter (limit: {limiter.max_calls}/{limiter.period}s)... {image_path_for_logging}")
        with limiter:  # <-- Rate limiter engages here
            logger.debug(f"Rate limiter passed, calling API... {image_path_for_logging}")
            # API call
            response = client.models.generate_content(  # Use the standard client method
                model=f'models/{model_name}',
                contents=contents,
                config=generation_config  # Pass config here
                # request_options={"timeout": 60} # Optional timeout
            )
            logger.debug(f"API call completed. {image_path_for_logging}")
        # --- /Rate Limiting ---

        if not response.candidates:
            # Check for prompt feedback if available
            block_reason = "Unknown"
            safety_ratings_str = "N/A"
            if response.prompt_feedback:
                block_reason = response.prompt_feedback.block_reason.name if response.prompt_feedback.block_reason else "Not Blocked"
                safety_ratings_str = ", ".join(
                    [f"{sr.category.name}: {sr.probability.name}" for sr in response.prompt_feedback.safety_ratings]) if response.prompt_feedback.safety_ratings else "None"

            logger.warning(
                f"No valid candidate response received from Gemini. Image: {image_path_for_logging}. Block Reason: {block_reason}. Safety Ratings: [{safety_ratings_str}]")
            # If blocked due to safety, treat as failure for this image
            if response.prompt_feedback and response.prompt_feedback.block_reason != types.SafetySetting.HarmBlockThreshold.BLOCK_NONE:
                return None  # Blocked, treat as failure
            # Otherwise, might be an issue, return empty list as a fallback? Or None? Let's return None.
            return None

        candidate = response.candidates[0]

        # Check finish reason for potential issues
        if candidate.finish_reason != types.FinishReason.STOP:
            finish_reason_name = candidate.finish_reason.name
            safety_ratings_str = ", ".join([f"{sr.category.name}: {sr.probability.name}" for sr in candidate.safety_ratings]) if candidate.safety_ratings else "None"
            logger.warning(f"Gemini response finished unexpectedly. Reason: {finish_reason_name}. Image: {image_path_for_logging}. Safety Ratings: [{safety_ratings_str}]")

            # Specific handling based on finish reason
            if finish_reason_name == "MAX_TOKENS":
                logger.error(f"Response truncated due to MAX_TOKENS limit. Image: {image_path_for_logging}. Consider adjusting model or prompt if response is too large.")
            elif finish_reason_name == "SAFETY":
                logger.error(f"Response blocked due to SAFETY reasons during generation. Image: {image_path_for_logging}. Safety Ratings: [{safety_ratings_str}]")
            elif finish_reason_name == "RECITATION":
                logger.warning(f"Response potentially blocked due to RECITATION. Image: {image_path_for_logging}.")
            elif finish_reason_name == "OTHER":
                logger.warning(f"Response finished due to 'OTHER' (could be API limit, internal error). Image: {image_path_for_logging}.")
            # Any non-STOP finish reason is treated as a failure for this annotation task
            return None  # Mark as error

        # Validate content exists and has parts
        if not candidate.content or not candidate.content.parts:
            # This case might happen if the model genuinely returns nothing *and* the API framework represents this as no parts.
            # Let's check safety ratings again here.
            safety_ratings_str = ", ".join([f"{sr.category.name}: {sr.probability.name}" for sr in candidate.safety_ratings]) if candidate.safety_ratings else "None"
            logger.warning(f"Content or parts not found in Gemini response, but finish reason was STOP. Image: {image_path_for_logging}. Safety Ratings: [{safety_ratings_str}]")
            # If safety ratings are clear, assume it meant to return empty.
            if candidate.safety_ratings and all(sr.probability == types.SafetySetting.HarmProbability.NEGLIGIBLE for sr in candidate.safety_ratings):
                logger.info(f"Assuming empty response means no objects found due to clear safety ratings. Image: {image_path_for_logging}")
                return []
            # Otherwise, it's ambiguous, treat as potential error.
            logger.warning(f"Ambiguous empty response. Treating as failure. Image: {image_path_for_logging}")
            return None

        # Extract text, assuming the first part contains the JSON
        # The response_mime_type='application/json' should ensure .text is the parsed JSON string
        raw_text = candidate.content.parts[0].text.strip()

        # Handle empty string (could mean no objects found as requested)
        if not raw_text:
            logger.info(f"Received empty text response from Gemini. Assuming no objects found. Image: {image_path_for_logging}")
            return []

        logger.debug(f"Gemini Raw JSON Text Response ({image_path_for_logging}): {raw_text}")

        # Parse the JSON response
        try:
            # json.loads should work directly because response_mime_type='application/json' was used
            annotations = json.loads(raw_text)

            # Ensure the result is a list as requested in the prompt
            if not isinstance(annotations, list):
                logger.warning(f"Expected JSON list from Gemini but received {type(annotations)}. Response: '{raw_text}'. Image: {image_path_for_logging}")
                # Attempt to handle if a single object dict was returned instead of a list
                if isinstance(annotations, dict):
                    box_key = next((k for k in annotations if k.lower() == "box_2d"), None)
                    label_key = next((k for k in annotations if k.lower() == "label"), None)
                    if box_key and label_key:
                        logger.info(f"Converting single object JSON dictionary to list. Image: {image_path_for_logging}")
                        return [annotations]
                # If it's neither a list nor a single valid object dict, treat as error
                logger.error(f"Response is not a JSON list or a convertible single object. Image: {image_path_for_logging}")
                return None  # Treat as failure
            return annotations
        except json.JSONDecodeError as e:
            logger.error(f"Could not parse Gemini response as JSON. Response: '{raw_text}'. Error: {e}. Image: {image_path_for_logging}")
            return None  # Treat JSON parsing error as failure
        except Exception as e:
            # Catch unexpected errors during JSON processing
            logger.error(f"Unexpected error processing Gemini JSON response ({image_path_for_logging}): {e}. Response: '{raw_text}'", exc_info=True)
            return None

    # Handle specific API errors for retrying
    except genai_errors.APIError as e:
        # Catch other potentially retryable API errors if not covered above
        logger.error(f"Unhandled Gemini API Error ({type(e).__name__}) for {image_path_for_logging}: {e}", exc_info=False)
        # Decide if this specific APIError subclass should be retried
        if isinstance(e, RETRYABLE_EXCEPTIONS):
            raise e  # Re-raise for tenacity
        else:
            logger.error(f"Treating API Error {type(e).__name__} as non-retryable for {image_path_for_logging}.")
            return None  # Non-retryable API error
    # Handle file not found before API call
    except FileNotFoundError:
        logger.error(f"Image not found for API call: {image_path_for_logging}")
        return None  # Non-retryable error for this specific image
    # Catch-all for other unexpected errors during the API call process
    except Exception as e:
        logger.error(f"Unexpected error during Gemini API call preparation or execution ({image_path_for_logging}): {e}", exc_info=True)
        # Only re-raise if it's a type Tenacity should retry
        if isinstance(e, RETRYABLE_EXCEPTIONS):
            raise e  # Re-raise for tenacity
        return None  # Treat as non-retryable failure for this image


# --- Main Processing Logic ---
def process_image(
        image_path: str,
        output_dir: str,
        class_list: List[str],
        class_mapping: Dict[str, int],
        client: genai.Client,
        model_name: str,
        limiter: RateLimiter
) -> Tuple[str, bool, Optional[str]]:
    """Processes a single image: API call (rate-limited), conversion, saving."""
    start_time = time.monotonic()
    base_filename = os.path.basename(image_path)
    logger.info(f"Processing: {base_filename}")

    try:
        original_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        logger.error(f"Orijinal resim açılamadı ({image_path}): {e}")
        return image_path, False, "Could not open original image."

    # Adım 1: Resmi kareye sığdır ve padding ekle
    padded_image = resize_and_pad_to_square(original_image, target_size=TARGET_SQUARE_SIZE)

    # Adım 2: Padded resmi API'ye göndermek için byte'a çevir
    with io.BytesIO() as output_bytes:
        padded_image.save(output_bytes, format="JPEG", quality=95)  # Yüksek kalite gönderelim
        image_bytes_for_api = output_bytes.getvalue()

    gemini_annotations = None
    try:
        # Güncellenmiş API çağrısı
        gemini_annotations = call_gemini_api(client, model_name, image_bytes_for_api, class_list, limiter, image_path)
    except RetryError as e:
        logger.error(f"API call failed after all retries ({base_filename}): {e}")

    if gemini_annotations is None:
        return image_path, False, "Gemini API call failed or returned invalid data."

    # Adım 3: Etiketleri ve YENİ resmi kaydetmek için yolları hazırla
    base_name_no_ext = os.path.splitext(base_filename)[0]
    annotation_filename = f"{base_name_no_ext}.txt"
    output_image_filename = f"{base_name_no_ext}.jpg"  # Çıktıyı standart olarak .jpg yapalım

    annotation_path = os.path.join(output_dir, annotation_filename)
    output_image_path = os.path.join(output_dir, output_image_filename)

    # Adım 4: Basitleştirilmiş fonksiyon ile YOLO'ya dönüştür
    yolo_lines = convert_gemini_to_yolo(gemini_annotations, class_mapping, image_path)

    # Adım 5: Hem etiketi hem de yeni, kare resmi kaydet
    try:
        os.makedirs(output_dir, exist_ok=True)

        # Etiket dosyasını yaz
        with open(annotation_path, "w", encoding='utf-8') as f:
            if yolo_lines:
                f.write("\n".join(yolo_lines) + "\n")

        # Yeni, kare resmi yaz (YOLO eğitimi için)
        padded_image.save(output_image_path, format="JPEG", quality=90)  # Kaydederken kaliteyi biraz düşürebiliriz

        processing_time = time.monotonic() - start_time
        message = f"{len(yolo_lines) if yolo_lines else 'No'} objects annotated ({processing_time:.2f}s)"
        logger.info(f"Success ({base_filename}): {message}")
        return image_path, True, message

    except Exception as e:
        logger.error(f"Could not write output files for '{base_filename}': {e}")
        return image_path, False, "Could not write output files."


def process_dataset(
        image_dir: str,
        output_dir: str,
        class_list_path: str,
        api_key: str,
        model_name: str,
        max_workers: int,
        rpm_limit: int = API_RPM_LIMIT,
        resume: bool = False  # <-- Added resume parameter
):
    """Processes images in the dataset in parallel (rate-limited), optionally resuming."""
    logger.info(f"Dataset processing started: {image_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model to use: {model_name}")
    logger.info(f"Maximum workers: {max_workers}")
    logger.info(f"API Request Limit: {rpm_limit} RPM")
    logger.info(f"Resume mode: {'Enabled' if resume else 'Disabled'}")  # <-- Log resume status

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
        logger.error(f"Could not configure Gemini Model Client: {e}", exc_info=True)
        return

    # --- Rate Limiter Creation ---
    limiter = RateLimiter(max_calls=rpm_limit, period=API_REQUEST_PERIOD)
    logger.info(f"Rate Limiter created: {rpm_limit} calls / {API_REQUEST_PERIOD} seconds")
    # --- /Rate Limiter Creation ---

    # Find all supported image files initially
    all_image_files = sorted([  # Sort for potentially more consistent processing order
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if os.path.isfile(os.path.join(image_dir, f)) and f.lower().endswith(SUPPORTED_IMAGE_FORMATS)
    ])

    if not all_image_files:
        logger.warning(f"No supported image formats found in the image directory ({image_dir}). Nothing to do.")
        return

    # --- Filter images based on resume flag ---
    images_to_process = []
    skipped_count = 0
    if resume:
        logger.info("Resume mode enabled. Checking for existing annotation files...")
        output_path = pathlib.Path(output_dir)
        existing_stems = {p.stem for p in output_path.glob('**/*.txt')}

        for img_path in all_image_files:
            # Orijinal resim dosyasının da uzantısız adını alıp karşılaştırırız.
            image_stem = pathlib.Path(img_path).stem
            if image_stem in existing_stems:
                logger.debug(f"Skipping '{os.path.basename(img_path)}': Annotation for stem '{image_stem}' already exists.")
                skipped_count += 1
            else:
                images_to_process.append(img_path)
        logger.info(f"Found {skipped_count} existing annotation files. Will attempt to process {len(images_to_process)} remaining images.")
    else:
        logger.info("Resume mode disabled. Processing all found images.")
        images_to_process = all_image_files

    # --- End of filtering ---

    if not images_to_process:
        if resume:
            logger.info(f"No images left to process in resume mode (all {len(all_image_files)} images already have annotations or initial list was empty).")
        else:
            # This case should have been caught earlier if all_image_files was empty
            logger.warning(f"No images selected for processing. This might be unexpected if resume mode was off.")
        return  # Exit if nothing to process

    total_to_process = len(images_to_process)
    logger.info(f"Total {total_to_process} images selected for processing.")

    processed_count = 0
    success_count = 0
    failed_count = 0
    start_time_total = time.monotonic()

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='Annotator') as executor:
        # Submit only the filtered list of images
        futures = {
            executor.submit(process_image, img_path, output_dir, class_list, class_mapping, client, model_name, limiter): img_path
            for img_path in images_to_process  # <-- Use the filtered list
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
                if processed_count % 20 == 0 or processed_count == total_to_process:
                    elapsed_time = time.monotonic() - start_time_total
                    rate = processed_count / elapsed_time if elapsed_time > 0 else 0
                    logger.info(
                        f"Progress: {processed_count}/{total_to_process} [{success_count}✓, {failed_count}✗] "
                        f"({elapsed_time:.1f}s, {rate:.2f} img/s)"
                    )

            except Exception as exc:
                # Catch errors that might occur retrieving the result itself (less likely now with better error handling in process_image)
                failed_count += 1
                logger.error(f"Unexpected task error retrieving result for image ({base_filename}): {exc}", exc_info=True)

    # Log summary after all selected images are processed
    end_time_total = time.monotonic()
    total_duration = end_time_total - start_time_total
    logger.info("-" * 40)
    logger.info("Processing Complete!")
    logger.info(f"Initial Images Found: {len(all_image_files)}")
    if resume: logger.info(f"Skipped (already annotated): {skipped_count}")
    logger.info(f"Attempted to Process: {total_to_process}")
    logger.info(f"Successful Annotations: {success_count}")
    logger.info(f"Failed Annotations: {failed_count}")
    logger.info(f"Total Processing Duration: {total_duration:.2f} seconds")
    if processed_count > 0: logger.info(f"Average Time/Processed Image: {total_duration / processed_count:.2f} seconds")
    logger.info("-" * 40)


# --- Argument Parser and Main Execution Block ---
def main():
    check_python_version()

    parser = argparse.ArgumentParser(
        description=f"Annotates an image dataset in YOLO format using the Gemini API (google-genai, rate-limited). Requires Python >= {PYTHON_MIN_VERSION[0]}.{PYTHON_MIN_VERSION[1]}.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("image_dir", help="Path to the directory containing images to be annotated.")
    parser.add_argument("output_dir", help="Directory where the generated YOLO annotation (.txt) files will be saved.")
    parser.add_argument("class_list_path", help="Path to the .txt file containing one class name per line.")
    parser.add_argument("--api_key", help="Google Gemini API key. it reads from the GOOGLE_API_KEY environment variable.", default=None)
    parser.add_argument("--model_name", help="Gemini model to use.", default=DEFAULT_GEMINI_MODEL)
    parser.add_argument("--max_workers", type=int, help="Maximum number of worker threads for parallel processing.", default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--rpm", type=int, help="Maximum API requests per minute limit.", default=API_RPM_LIMIT)
    parser.add_argument("--resume", action="store_true", help="Enable resume mode: skip images with existing annotation files in the output directory.")  # <-- Resume argument
    parser.add_argument("--log_level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO', help="Logging level to set.")

    args = parser.parse_args()

    # Set logging level based on argument
    log_level_int = getattr(logging, args.log_level.upper(), logging.INFO)
    logger.setLevel(log_level_int)
    # Optionally set level for other handlers if needed
    for handler in logger.handlers:
        handler.setLevel(log_level_int)

    # Get API key from argument or environment variable
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.critical("Google Gemini API key was not provided either via --api_key argument or GOOGLE_API_KEY environment variable. Stopping.")
        return 1  # Indicate error

    # Pass RPM limit and resume flag to process_dataset
    process_dataset(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        class_list_path=args.class_list_path,
        api_key=api_key,
        model_name=args.model_name,
        max_workers=args.max_workers,
        rpm_limit=args.rpm,
        resume=args.resume  # <-- Pass resume flag
    )
    return 0  # Indicate success


# --- Script Execution (Example) ---
# The following section is for direct execution, taking arguments from here.
# To run from the command line, remove or comment out this section and use `if __name__ == "__main__": main()`

# Example Usage (For direct execution - instead of command line)
# Set these variables according to your setup
RUN_DIRECTLY = True  # Set to False to use command-line arguments instead
MODEL_NAME = 'gemini-2.0-flash'  # <-- Model name (e.g., 1.5-flash or higher)
MAX_WORKERS = 40  # <-- Number of workers (mind the RPM limit!)
RPM_LIMIT_VALUE = 1500  # <-- Requests per minute limit (e.g., 15 for Free Tier)
RESUME_PROCESSING = True  # <-- Set to True to enable resume mode when running directly
LOG_LEVEL_DIRECT = logging.INFO  # <-- Set desired log level (e.g., logging.DEBUG)

# --------------------------- Direct‑run defaults -----------------------------
IMAGE_DIR = "Havayolu_Veriseti"
OUTPUT_DIR = "Havayolu_Veriseti_Output"
CLASS_LIST_PATH = "class_list.txt"
RESUME = True
# 💡 NEW: explicit three-way split ratios (must sum to 1)
TRAIN_RATIO      = 0.7
VAL_RATIO        = 0.2
TEST_RATIO       = 0.1

# -----------------------------------------------------------------------------
#                           Module entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import random, shutil, pathlib, sys, os, textwrap


    # -------------------- helper: write data.yaml --------------------
    def write_data_yaml(output_dir: str, class_list: list[str]):
        yaml_path = pathlib.Path(output_dir) / "data.yaml"
        nc = len(class_list)
        names_str = ", ".join(f"'{n}'" for n in class_list)
        text = textwrap.dedent(
            f"""
            train: ../train/images
            val: ../val/images
            test: ../test/images

            nc: {nc}
            names: [{names_str}]

            roboflow:
              workspace: general-ecrbw
              project: general_gemini
              version: 1
              license: CC BY 4.0
              url: https://universe.roboflow.com/general-ecrbw/general_gemini/dataset/1
            """
        ).lstrip()
        try:
            yaml_path.write_text(text, encoding="utf-8")
            logger.info("data.yaml written → %s", yaml_path)
        except Exception as e:
            logger.error("Could not write data.yaml: %s", e)


    # ---------------------- helper: split_dataset --------------------
    def split_dataset(
            source_dir: str,  # <-- Artık kaynak olarak output_dir'i alacak
            train_ratio: float = 0.7,
            val_ratio: float = 0.2,
            test_ratio: float = 0.1,
    ):
        """
        Artık hem resimleri hem de etiketleri AYNI kaynak klasörden alır ve böler.
        """
        logger.info(f"Splitting dataset from source: {source_dir}")
        if not abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6:
            raise ValueError("Split ratios must sum to 1.0")

        source_path = pathlib.Path(source_dir)
        # .txt dosyalarını bul, çünkü her resmin bir etiketi olmayabilir ama her etiketin bir resmi olmalı
        lbl_files = list(source_path.glob("*.txt"))
        if not lbl_files:
            logger.warning("No annotation files found in output directory for splitting. Skipping.")
            return

        stems = sorted([p.stem for p in lbl_files])
        random.seed(42)
        random.shuffle(stems)

        n = len(stems)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = {
            'train': stems[:n_train],
            'val': stems[n_train:n_train + n_val],
            'test': stems[n_train + n_val:]
        }

        for split_name, file_stems in splits.items():
            if not file_stems: continue

            # Hedef klasörleri oluştur
            img_dir_dst = source_path / split_name / 'images'
            lbl_dir_dst = source_path / split_name / 'labels'
            img_dir_dst.mkdir(parents=True, exist_ok=True)
            lbl_dir_dst.mkdir(parents=True, exist_ok=True)

            for stem in file_stems:
                # Hem .jpg hem de .txt dosyasını taşı
                shutil.move(str(source_path / f"{stem}.jpg"), str(img_dir_dst / f"{stem}.jpg"))
                shutil.move(str(source_path / f"{stem}.txt"), str(lbl_dir_dst / f"{stem}.txt"))

        logger.info(f"Split complete – train {len(splits['train'])}, val {len(splits['val'])}, test {len(splits['test'])}")

        try:
            # CLASS_LIST_PATH'in global olarak erişilebilir olduğunu varsayıyoruz
            with open(CLASS_LIST_PATH, "r", encoding="utf-8") as fh:
                cls_names = [ln.strip() for ln in fh if ln.strip()]
            # data.yaml dosyasını ana çıktı klasörüne yaz
            write_data_yaml(source_dir, cls_names)
        except Exception as e:
            logger.error(f"Unable to read classes for data.yaml: {e}")


    def post_split(out_dir: str):
        logger.info("🏁 Annotation phase finished – starting dataset split …")
        # Artık hem resimlerin hem de etiketlerin kaynağı OUTPUT_DIR
        split_dataset(out_dir, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
        logger.info("✅ Split completed.")


    # ------------------- Choose CLI vs direct defaults -----------------
    if len(sys.argv) > 1:
        code = main()
        if code == 0:
            img_dir = os.environ.get("IMAGE_DIR", IMAGE_DIR)
            out_dir = os.environ.get("OUTPUT_DIR", OUTPUT_DIR)
            post_split(out_dir)
        sys.exit(code)

    logger.info("No CLI args – using internal defaults.")
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        logger.critical("Google API key missing.")
        sys.exit(1)
    try:
        process_dataset(
            image_dir=IMAGE_DIR,
            output_dir=OUTPUT_DIR,
            class_list_path=CLASS_LIST_PATH,
            api_key=key,
            model_name=MODEL_NAME,
            max_workers=MAX_WORKERS,
            rpm_limit=RPM_LIMIT_VALUE,
            resume=RESUME,
        )
        post_split(OUTPUT_DIR)
        sys.exit(0)
    except Exception as e:
        logger.error("Fatal error: %s", e)
        sys.exit(2)
