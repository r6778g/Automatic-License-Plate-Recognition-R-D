import os
import cv2
import argparse
import numpy as np
from pathlib import Path
from paddleocr import PaddleOCR

# Disable paddle messages
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

def process_paddle_video(video_path, output_path=None, language='ch', skip_frames=10):
    """
    Process video using PaddleOCR.
    """
    print(f"🚀 Initializing PaddleOCR ({language})...")
    # Updated to remove show_log and use use_textline_orientation instead of use_angle_cls
    ocr = PaddleOCR(use_textline_orientation=True, lang=language)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"🎬 Processing: {video_path}")
    
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    active_detections = [] # To store detections for display

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run OCR every N frames
        if frame_count % skip_frames == 0:
            # Using predict() as recommended by the deprecation warning
            # The output format can vary by version, so we parse defensively
            try:
                # PaddleOCR predict() returns a list or a results object
                results = ocr.predict(frame)
                
                # Check if it's the standard list format [[ [bbox, (text, conf)], ... ]]
                # or a newer object-based format
                processed_lines = []
                if isinstance(results, list) and len(results) > 0:
                    # In many versions, it's results[0] for a single image input
                    processed_lines = results[0] if isinstance(results[0], list) else results
                
                active_detections = []
                for line in processed_lines:
                    try:
                        # Standard format check
                        if isinstance(line, (list, tuple)) and len(line) >= 2:
                            bbox = line[0]
                            content = line[1]
                            
                            # Content is usually (text, score)
                            if isinstance(content, (list, tuple)) and len(content) >= 2:
                                text = content[0]
                                conf = content[1]
                            else:
                                # Fallback if content is just a string or different shape
                                text = str(content)
                                conf = 1.0 # Default confidence
                            
                            if isinstance(conf, (int, float)) and conf > 0.5:
                                pts = np.array(bbox, np.int32).reshape((-1, 1, 2))
                                active_detections.append({
                                    "text": text,
                                    "pts": pts,
                                    "ttl": skip_frames + 2
                                })
                    except Exception:
                        continue # Skip malformed lines
            except Exception as e:
                print(f"  ⚠️ OCR Error on frame {frame_count}: {e}")

        # Draw detections
        for det in active_detections:
            if det["ttl"] > 0:
                cv2.polylines(frame, [det["pts"]], True, (0, 255, 0), 2)
                # Bounding rect for text placement
                x, y, w, h = cv2.boundingRect(det["pts"])
                cv2.putText(frame, det["text"], (x, max(30, y - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                det["ttl"] -= 1

        if output_path:
            if writer:
                writer.write(frame)
        else:
            # Only show window if no output path is provided
            cv2.imshow("PaddleOCR Video", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1
        if frame_count % 20 == 0:
            print(f"  Frame {frame_count}/{total_frames}...")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("✅ Finished!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs="?", default="Automatic Number Plate Recognition (ANPR) _ Vehicle Number Plate Recognition (1).mp4", help="Path to video file")
    parser.add_argument("--lang", default="ch", help="Language code (ch, en, etc.)")
    parser.add_argument("--skip", type=int, default=10, help="Frames to skip")
    parser.add_argument("--output", help="Save output video path")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"❌ Video not found: {args.video}")
    else:
        # Check if running in a venv that has paddle
        process_paddle_video(args.video, args.output, args.lang, args.skip)
