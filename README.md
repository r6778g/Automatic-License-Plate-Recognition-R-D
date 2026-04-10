# Automatic License Plate Recognition (ALPR) R&D

This repository contains research and prototype work for automatic license plate recognition using a few different approaches:

- Classic OpenCV + Tesseract
- PaddleOCR for multilingual OCR
- Google Cloud Vision
- Plate Recognizer API
- Jupyter notebooks for experiments and comparisons

## Project Structure

```text
Automatic License Plate Recognition-R&D/
  classic_opencv_tesseract.py
  classic_video_alpr.py
  paddle_ocr_demo.py
  paddle_video_alpr.py
  google_cloud_alpr.py
  plate_recognizer_video.py
  test_google_cloud.py

solutions/
  README.md
  *.ipynb

test/
  sample images
```

## Requirements

- Python 3.10+ recommended
- macOS, Linux, or Windows
- Tesseract OCR installed locally for the classic pipeline

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If you want to run PaddleOCR, install the Paddle runtime too:

```bash
pip install paddlepaddle
```

### 3. Install Tesseract

macOS with Homebrew:

```bash
brew install tesseract
```

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr
```

### 4. Configure environment variables

Create a `.env` file in the repository root:

```env
PLATE_RECOGNIZER_API_KEY=your_api_key_here
GOOGLE_CLOUD_API_KEY=your_google_api_key_here
GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account.json
```

Notes:

- `PLATE_RECOGNIZER_API_KEY` is used by `plate_recognizer_video.py`
- `GOOGLE_CLOUD_API_KEY` or `GOOGLE_APPLICATION_CREDENTIALS` is used by the Google Cloud scripts
- `.env` is ignored by git and should not be committed

## Run the Main Scripts

Run commands from the repository root.

### Classic OpenCV + Tesseract on an image

```bash
python "Automatic License Plate Recognition-R&D/classic_opencv_tesseract.py" test/image-1.png
```

### Classic OpenCV + Tesseract on a video

```bash
python "Automatic License Plate Recognition-R&D/classic_video_alpr.py" \
  "Automatic Number Plate Recognition (ANPR) _ Vehicle Number Plate Recognition (1).mp4"
```

### PaddleOCR image demo

```bash
python "Automatic License Plate Recognition-R&D/paddle_ocr_demo.py"
```

### PaddleOCR video pipeline

```bash
python "Automatic License Plate Recognition-R&D/paddle_video_alpr.py" \
  "Automatic Number Plate Recognition (ANPR) _ Vehicle Number Plate Recognition (1).mp4" \
  --lang ch --skip 10
```

### Google Cloud Vision test

```bash
python "Automatic License Plate Recognition-R&D/test_google_cloud.py"
```

### Plate Recognizer video pipeline

```bash
python "Automatic License Plate Recognition-R&D/plate_recognizer_video.py"
```

## Notebooks

 contains notebook-based experiments for different ALPR strategies. Start with:

- `01_classic_opencv_tesseract.ipynb`
- `03_yolo_paddleocr.ipynb`
- `07_cloud_apis.ipynb`

## Notes

- The repo includes local sample assets and model files used during experimentation
- `venv/`, `venv-paddle/`, caches, and secrets are excluded from git
