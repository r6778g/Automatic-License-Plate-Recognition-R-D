from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract


ALPHA_FROM_DIGIT = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "4": "A",
    "5": "S",
    "6": "G",
    "7": "T",
    "8": "B",
}

DIGIT_FROM_ALPHA = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
    "G": "6",
    "T": "7",
    "A": "4",
}

TESSERACT_CONFIG = (
    "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)


def resolve_image_path(preferred: str | None = None) -> str:
    candidates = []
    if preferred:
        candidates.append(Path(preferred))

    module_root = Path(__file__).resolve().parent
    cwd = Path.cwd()
    relative_candidates = [
        Path("test/image-1.png"),
        Path("test/image-7.png"),
        Path("../test/image-1.png"),
        Path("../test/image-7.png"),
    ]

    for root in (cwd, module_root, cwd.parent, module_root.parent):
        for rel in relative_candidates:
            candidates.append(root / rel)

    seen = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return str(resolved)

    search_dirs = [
        cwd,
        cwd / "test",
        cwd / "images",
        module_root,
        module_root / "test",
        module_root / "images",
    ]
    suffixes = {".png", ".jpg", ".jpeg", ".bmp"}
    for directory in search_dirs:
        if not directory.exists():
            continue
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() in suffixes:
                return str(path.resolve())

    raise FileNotFoundError("No test image found in test/ or images/")


def visualize_steps(images, titles, cols=3, figsize=(6, 5)):
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(figsize[0] * cols, figsize[1] * rows))
    axes = np.array(axes).reshape(-1)

    for index, (image, title) in enumerate(zip(images, titles)):
        if image.ndim == 2:
            axes[index].imshow(image, cmap="gray")
        else:
            axes[index].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[index].set_title(title, fontweight="bold")
        axes[index].axis("off")

    for index in range(len(images), len(axes)):
        axes[index].axis("off")

    plt.tight_layout()
    plt.show()


def clean_text(text: str) -> str:
    return "".join(char for char in text.upper() if char.isalnum())


def normalize_plate_text(text: str):
    best = None

    for length in (9, 10, 11):
        if len(text) < length:
            continue

        for start in range(0, len(text) - length + 1):
            sample = text[start : start + length]
            middle_letters = length - 8
            groups = [("alpha", 2), ("digit", 2), ("alpha", middle_letters), ("digit", 4)]

            normalized = []
            exact_matches = 0
            penalties = 0
            cursor = 0
            valid = True

            for kind, size in groups:
                chunk = sample[cursor : cursor + size]
                cursor += size

                for char in chunk:
                    if kind == "alpha":
                        if char.isalpha():
                            normalized.append(char)
                            exact_matches += 1
                        elif char in ALPHA_FROM_DIGIT:
                            normalized.append(ALPHA_FROM_DIGIT[char])
                            penalties += 1
                        else:
                            valid = False
                            break
                    else:
                        if char.isdigit():
                            normalized.append(char)
                            exact_matches += 1
                        elif char in DIGIT_FROM_ALPHA:
                            normalized.append(DIGIT_FROM_ALPHA[char])
                            penalties += 1
                        else:
                            valid = False
                            break

                if not valid:
                    break

            if not valid:
                continue

            score = exact_matches * 2 - penalties * 3 - start - abs(length - 10) * 4
            candidate = {
                "text": "".join(normalized),
                "raw_window": sample,
                "score": score,
                "penalties": penalties,
                "length": length,
            }

            if best is None or candidate["score"] > best["score"]:
                best = candidate

    return best


def _ocr_variants(plate_crop):
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 25, 25)
    gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)

    return {
        "Gray + CLAHE": gray,
        "Otsu Threshold": cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1],
        "Adaptive Threshold": cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        ),
    }


def preprocess_for_detection(image):
    height, width = image.shape[:2]
    x0, x1 = int(width * 0.1), int(width * 0.9)
    y0, y1 = int(height * 0.35), int(height * 0.9)
    search_roi = image[y0:y1, x0:x1]

    gray = cv2.cvtColor(search_roi, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 9, 17, 17)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    square_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    blackhat = cv2.morphologyEx(filtered, cv2.MORPH_BLACKHAT, rect_kernel)
    gradient = np.abs(cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=-1))
    gradient = (
        255
        * ((gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-6))
    ).astype("uint8")
    gradient = cv2.GaussianBlur(gradient, (5, 5), 0)
    gradient = cv2.morphologyEx(gradient, cv2.MORPH_CLOSE, rect_kernel)

    binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, square_kernel)
    binary = cv2.erode(binary, None, iterations=2)
    binary = cv2.dilate(binary, None, iterations=2)

    return {
        "roi_bounds": (x0, y0, x1, y1),
        "gray": gray,
        "blackhat": blackhat,
        "gradient": gradient,
        "binary": binary,
    }


def find_plate_candidates(image):
    height, width = image.shape[:2]
    debug = preprocess_for_detection(image)
    x0, y0, _, _ = debug["roi_bounds"]

    contours, _ = cv2.findContours(
        debug["binary"].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
        x, y, w, h = cv2.boundingRect(contour)
        x += x0
        y += y0

        aspect_ratio = w / float(h)
        area = w * h
        if not (2.0 <= aspect_ratio <= 7.0):
            continue
        if area < height * width * 0.004 or area > height * width * 0.18:
            continue
        if not (0.03 * height <= h <= 0.22 * height):
            continue

        center_score = 1.0 - abs((x + w / 2) - width / 2) / (width / 2)
        y_score = (y + h / 2) / height
        aspect_score = 1.0 - min(abs(aspect_ratio - 4.5) / 4.5, 1.0)
        area_score = min(area / (height * width * 0.02), 1.0)
        score = (
            0.35 * center_score
            + 0.25 * y_score
            + 0.2 * aspect_score
            + 0.2 * area_score
        )

        candidates.append(
            {
                "bbox": (x, y, w, h),
                "score": score,
                "area": area,
                "aspect_ratio": aspect_ratio,
            }
        )

    return sorted(candidates, key=lambda item: item["score"], reverse=True), debug


def _extract_plate_crop(image, bbox, pad_x, pad_y):
    x, y, w, h = bbox
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(image.shape[1], x + w + pad_x)
    y2 = min(image.shape[0], y + h + pad_y)
    return image[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)


def ocr_plate_crop(plate_crop):
    attempts = []

    for pad_label, pad_x in (("Tight Crop", 12), ("Wide Crop", 14)):
        variants = _ocr_variants(plate_crop)

        for variant_name, variant in variants.items():
            scale = max(1.0, 140 / variant.shape[0])
            enlarged = cv2.resize(
                variant, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
            padded = cv2.copyMakeBorder(
                enlarged, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255
            )
            raw_text = clean_text(
                pytesseract.image_to_string(padded, config=TESSERACT_CONFIG)
            )
            normalized = normalize_plate_text(raw_text)
            if not normalized:
                continue

            attempts.append(
                {
                    "pad_label": pad_label,
                    "variant_name": variant_name,
                    "raw_text": raw_text,
                    "normalized_text": normalized["text"],
                    "score": normalized["score"],
                    "preview": padded,
                }
            )

    attempts.sort(key=lambda item: item["score"], reverse=True)
    return attempts


def run_classic_pipeline(image_path: str):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    candidates, debug = find_plate_candidates(image)
    annotated = image.copy()
    for index, candidate in enumerate(candidates[:5], start=1):
        x, y, w, h = candidate["bbox"]
        color = (0, 255, 0) if index == 1 else (255, 180, 0)
        thickness = 3 if index == 1 else 2
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(
            annotated,
            f"#{index}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    result = {
        "image": image,
        "annotated_image": annotated,
        "candidates": candidates,
        "debug": debug,
        "plate_crop": None,
        "ocr_attempts": [],
        "best_text": None,
        "best_bbox": None,
    }

    if not candidates:
        return result

    best_candidate = candidates[0]
    plate_crop = None
    ocr_attempts = []
    best_expanded_bbox = None

    for pad_x in (12, 14):
        crop, expanded_bbox = _extract_plate_crop(image, best_candidate["bbox"], pad_x, 8)
        attempts = ocr_plate_crop(crop)
        if attempts:
            best_attempt = attempts[0]
            if not ocr_attempts or best_attempt["score"] > ocr_attempts[0]["score"]:
                plate_crop = crop
                ocr_attempts = attempts
                best_expanded_bbox = expanded_bbox

    result["plate_crop"] = plate_crop
    result["ocr_attempts"] = ocr_attempts
    result["best_bbox"] = best_expanded_bbox or best_candidate["bbox"]
    if ocr_attempts:
        result["best_text"] = ocr_attempts[0]["normalized_text"]

    return result


def classic_alpr(image_path: str, visualize: bool = True):
    result = run_classic_pipeline(image_path)
    best_text = result["best_text"]

    if visualize:
        annotated = result["annotated_image"].copy()
        if best_text and result["best_bbox"]:
            x, y, w, h = result["best_bbox"]
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(
                annotated,
                best_text,
                (x, max(30, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 255, 0),
                2,
            )

        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        plt.title(best_text or "No confident plate text found", fontweight="bold")
        plt.axis("off")
        plt.show()

    return [best_text] if best_text else []


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run the classic OpenCV + Tesseract license plate pipeline."
    )
    parser.add_argument(
        "image",
        nargs="?",
        help="Image path. Defaults to a sample image from test/ or images/ if omitted.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show the annotated detection result with matplotlib.",
    )
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    image_path = resolve_image_path(args.image) if args.image else resolve_image_path()
    results = classic_alpr(image_path, visualize=args.visualize)

    print(f"Image: {image_path}")
    if results:
        print("Detected plates:")
        for plate in results:
            print(f"- {plate}")
    else:
        print("Detected plates: none")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
