# Gemini Annotator

- Gemini Annotator is a Python project that uses the Gemini API to annotate images in YOLO format. The project utilizes uv for Python package management.
- This project structured to use free Gemini API with rate limiting (15 RPM) and concurrent processing using a thread pool.
- If you want to use the paid Gemini API, you can set the RPM limit to a higher value to process more images concurrently.
- To continue previous runs, you can set the `--resume` argument to `True`. This will skip images that have already been processed.

- Try to use understandable names for the class_list.txt file to get better results.

## Features

- Processes images to detect objects and generate YOLO format annotations.
- Utilizes Gemini API with rate limiting.
- Concurrent processing using a thread pool.

## Requirements

- Python >= 3.9
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
Define your desired classes in the `class_list.txt` file. Each class should be on a new line.

Enter your Gemini API key in the `main.py` file. You can also pass it as a command-line argument.
You can set the `--api_key` argument to your Gemini API key. If you don't provide it, the script will use the default key in the code.

You can also set the `--model_name` argument to specify the model you want to use. The default is `gemini-2.0-flash`.
You can set the `--max_workers` argument to specify the number of concurrent threads. The default is 4.
You can set the `--rpm` argument to specify the rate limit for the API. The default is 15 RPM.
You can set the `--log_level` argument to specify the logging level. The default is `INFO`. You can set it to `DEBUG` for more detailed logs.
You can set the `--output_dir` argument to specify the output directory for the annotations. The default is `output`.
You can set the `--image_dir` argument to specify the directory containing the images to be annotated. The default is `images`.
You can set the `--class_list` argument to specify the file containing the list of object classes. The default is `class_list.txt`.
You can set the `--resume` argument to specify whether to resume from a previous run. The default is `False`. If set to `True`, the script will skip images that have already been processed.


Run the following command to process a dataset:

```
python main.py <image_dir> <output_dir> <class_list.txt> [--api_key YOUR_API_KEY] [--model_name MODEL_NAME] [--max_workers NUMBER] [--rpm RPM_LIMIT] [--log_level LEVEL]
```

Example:

```
python main.py images output class_list.txt --api_key YOUR_API_KEY --model_name gemini-2.0-flash --max_workers 4 --rpm 15 --resume True
```

## Project Structure

- `pyproject.toml`: Project configuration and dependencies.
- `main.py`: Main script for image annotation.
- `class_list.txt`: List of object classes (one per line).
- `README.md`: Project documentation.

## License

MIT License.
