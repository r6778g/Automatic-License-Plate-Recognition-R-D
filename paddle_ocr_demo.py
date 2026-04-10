# paddle_ocr_demo.py
# Strong multi-language OCR, exceptionally good for non-Latin plates (Chinese, Korean, Cyrillic, Arabic, etc.)
# Make sure to install: pip install paddlepaddle paddleocr

import os
import sys
from pathlib import Path

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

import matplotlib.pyplot as plt
import cv2
from paddleocr import PaddleOCR


def _recommended_python():
    candidate = Path(__file__).resolve().parent / "venv-paddle" / "bin" / "python"
    return candidate if candidate.exists() else None


def ensure_paddle_runtime():
    py_version = sys.version_info[:2]
    if not ((3, 9) <= py_version <= (3, 13)):
        print(
            f"PaddleOCR needs Python 3.9-3.13, but this script is running on "
            f"{sys.version.split()[0]}."
        )
        recommended = _recommended_python()
        if recommended is not None:
            print(f"Run it with: {recommended} {Path(__file__).name}")
        return False

    try:
        import paddle  # noqa: F401
    except ModuleNotFoundError:
        print("This environment has `paddleocr`, but `paddle` is missing.")
        recommended = _recommended_python()
        if recommended is not None:
            print(f"Use the ready environment instead: {recommended} {Path(__file__).name}")
        else:
            print("Install the runtime first: python -m pip install paddlepaddle")
        return False

    return True


def build_paddleocr(language):
    init_variants = [
        {"use_textline_orientation": True, "lang": language},
        {"use_angle_cls": True, "lang": language},
        {"lang": language},
    ]
    last_error = None
    for kwargs in init_variants:
        try:
            return PaddleOCR(**kwargs)
        except TypeError as exc:
            last_error = exc
        except Exception as exc:
            last_error = exc
            break
    if last_error is not None:
        raise last_error
    raise RuntimeError("Could not initialize PaddleOCR.")


def _page_get(page, key, default=None):
    if hasattr(page, "get"):
        return page.get(key, default)
    return getattr(page, key, default)


def normalize_paddleocr_output(raw_result):
    entries = []
    if not raw_result:
        return entries

    for page in raw_result:
        if page is None:
            continue

        if isinstance(page, list):
            for line in page:
                if not line or len(line) < 2:
                    continue
                bbox = line[0]
                text, confidence = line[1]
                entries.append(
                    {
                        "bbox": bbox,
                        "text": text,
                        "confidence": float(confidence),
                    }
                )
            continue

        texts = _page_get(page, "rec_texts", []) or []
        scores = _page_get(page, "rec_scores", []) or []
        polys = _page_get(page, "rec_polys", []) or _page_get(page, "dt_polys", []) or []

        for idx, text in enumerate(texts):
            entries.append(
                {
                    "bbox": polys[idx] if idx < len(polys) else None,
                    "text": text,
                    "confidence": float(scores[idx]) if idx < len(scores) else 0.0,
                }
            )

    return entries


def run_paddleocr(engine, image):
    try:
        raw_result = engine.predict(image)
    except TypeError:
        try:
            raw_result = engine.ocr(image, cls=True)
        except TypeError:
            raw_result = engine.ocr(image)
    return normalize_paddleocr_output(raw_result)

def detect_multi_language_plate(image_path, language='ch'):
    """
    PaddleOCR supports multiple languages seamlessly.
    Language codes: 
      'en' = English
      'ch' = Chinese & English
      'korean' = Korean
      'japan' = Japanese
      'cyrillic' = Russian/Cyrillic
      'ar' = Arabic
    """
    print(f"Initializing PaddleOCR with language: {language}...")

    if not ensure_paddle_runtime():
        return

    # 1. Initialize PaddleOCR.
    # Newer releases prefer use_textline_orientation/predict(), while older ones still accept use_angle_cls/ocr().
    try:
        ocr = build_paddleocr(language)
    except ModuleNotFoundError:
        print("PaddleOCR could not start because the Paddle runtime is missing.")
        recommended = _recommended_python()
        if recommended is not None:
            print(f"Run it with: {recommended} {Path(__file__).name}")
        return
    except Exception as exc:
        if language != "en":
            print(
                f"Could not initialize language '{language}' offline "
                f"({type(exc).__name__}: {exc})."
            )
            print("Falling back to cached English OCR models.")
            try:
                ocr = build_paddleocr("en")
            except Exception as fallback_exc:
                print(
                    "PaddleOCR initialization failed even after fallback: "
                    f"{type(fallback_exc).__name__}: {fallback_exc}"
                )
                return
        else:
            print(f"PaddleOCR initialization failed: {type(exc).__name__}: {exc}")
            return

    # 2. Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return

    # 3. Run OCR inference
    print(f"Running PaddleOCR on {image_path}...")
    result = run_paddleocr(ocr, image_path)

    # 4. Process and display the results
    plates_detected = []
    
    if result:
        for idx, line in enumerate(result):
            bbox = line["bbox"]
            text = line["text"]
            confidence = line["confidence"]

            # Filter low confidence artifacts
            if confidence > 0.50:
                print(f"Detected Text [{idx}]: {text} | Confidence: {confidence:.2f}")
                plates_detected.append(text)

                # Format bounding box for drawing (Top Left and Bottom Right)
                if bbox is None:
                    continue
                top_left = (int(bbox[0][0]), int(bbox[0][1]))
                bottom_right = (int(bbox[2][0]), int(bbox[2][1]))

                # Draw the bounding box (Green)
                cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)

                # Note: Default OpenCV cannot render non-Latin characters (like Chinese) natively in cv2.putText.
                # So we draw the boxes and show the text in the terminal and matplotlib title instead!

    # 5. Display the final image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb)
    
    if plates_detected:
        plt.title(f"PaddleOCR Results: {' | '.join(plates_detected)}")
    else:
        plt.title("No plates detected.")
        
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    # Test on a local image. 
    # Try providing a Chinese, Russian, or Arabic license plate here!
    test_image = "test/image-9.png" 
    
    # We use 'ch' (Chinese) by default to demonstrate the multi-language capability.
    # It automatically detects English numbers/letters as well!
    detect_multi_language_plate(test_image, language='ch')
