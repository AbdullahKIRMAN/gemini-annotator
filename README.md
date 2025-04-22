# Gemini Annotater

Gemini Annotater is a Python project that uses the Gemini API to annotate images in YOLO format. The project utilizes uv for Python package management.

## Features

- Processes images to detect objects and generate YOLO format annotations.
- Utilizes Gemini API with rate limiting.
- Concurrent processing using a thread pool.
- Easy customization via configuration in `pyproject.toml`.

## Requirements

- Python >= 3.10
- google-genai >= 1.11.0
- Pillow >= 11.2.1
- ratelimiter >= 1.2.0.post0
- tenacity >= 9.1.2

## Installation

1. Install uv (if not already installed):
   ```
   pip install uv
   ```
2. Install project dependencies using uv based on the configuration in `pyproject.toml`:
   ```
   uv sync
   ```

## Usage

Run the following command to process a dataset:

```
python main.py <image_dir> <output_dir> <class_list.txt> [--api_key YOUR_API_KEY] [--model_name MODEL_NAME] [--max_workers NUMBER] [--rpm RPM_LIMIT] [--log_level LEVEL]
```

Example:

```
python main.py images output class_list.txt --api_key YOUR_API_KEY --model_name gemini-2.0-flash --max_workers 4 --rpm 15
```

## Project Structure

- `pyproject.toml`: Project configuration and dependencies.
- `main.py`: Main script for image annotation.
- `class_list.txt`: List of object classes (one per line).
- `README.md`: Project documentation.

## License

MIT License.
