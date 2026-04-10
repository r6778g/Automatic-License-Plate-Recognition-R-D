import cv2
import time
import os
import argparse
from classic_opencv_tesseract import run_classic_pipeline

def process_video(video_path, output_path=None, skip_frames=5):
    """
    Process a video file using the classic OpenCV + Tesseract pipeline.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        return

    # Get video properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"🎬 Processing Video: {video_path}")
    print(f"📊 {width}x{height} @ {fps:.2f} FPS | Total Frames: {total_frames}")

    # Setup Video Writer if output path provided
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    results_cache = {} # To keep detected text on screen for a few frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process every Nth frame for performance
        if frame_count % skip_frames == 0:
            # save frame to a temp file because run_classic_pipeline expects a path
            temp_path = "temp_frame.jpg"
            cv2.imwrite(temp_path, frame)
            
            try:
                pipeline_result = run_classic_pipeline(temp_path)
                best_text = pipeline_result.get("best_text")
                best_bbox = pipeline_result.get("best_bbox")
                
                if best_text:
                    results_cache = {
                        "text": best_text,
                        "bbox": best_bbox,
                        "ttl": skip_frames * 2 # frames to hold the text on screen
                    }
            except Exception as e:
                print(f"⚠️ Error processing frame {frame_count}: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        # Draw overlays from cache
        if results_cache and results_cache["ttl"] > 0:
            text = results_cache["text"]
            x, y, w, h = results_cache["bbox"]
            
            # Draw green box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Draw label
            cv2.putText(frame, text, (x, max(30, y - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            results_cache["ttl"] -= 1

        # Show frame
        cv2.imshow("Classic ALPR Video", frame)
        if writer:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1
        if frame_count % 50 == 0:
            print(f"  Processed {frame_count}/{total_frames} frames...")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("✅ Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs="?", default="Automatic Number Plate Recognition (ANPR) _ Vehicle Number Plate Recognition (1).mp4", help="Path to video file")
    parser.add_argument("--output", help="Path to save output video")
    parser.add_argument("--skip", type=int, default=5, help="Number of frames to skip for OCR")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"❌ Video file not found: {args.video}")
    else:
        process_video(args.video, args.output, args.skip)
