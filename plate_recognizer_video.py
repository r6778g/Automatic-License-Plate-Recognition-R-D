import cv2
import os
import requests
import json
import time
from dotenv import load_dotenv

# Load API Key from .env
load_dotenv()

PLATE_RECOGNIZER_API_KEY = os.environ.get('PLATE_RECOGNIZER_API_KEY', '244f60981ec4110f744d32ba1a5c44b3bebbedd8')

def plate_recognizer(image_path, api_key):
    """Call Plate Recognizer Cloud API for a single image."""
    url = 'https://api.platerecognizer.com/v1/plate-reader/'
    headers = {'Authorization': f'Token {api_key}'}
    
    if not os.path.exists(image_path):
        return []
        
    with open(image_path, 'rb') as f:
        # We can also add regions to improve accuracy, e.g., regions=['in']
        response = requests.post(url, headers=headers, files={'upload': f})
    
    if response.status_code in [200, 201]:
        data = response.json()
        results = []
        for r in data.get('results', []):
            result = {
                'plate': r['plate'].upper(),
                'score': r['score'],
                'region': r.get('region', {}).get('code', 'unknown'),
                'vehicle_type': r.get('vehicle', {}).get('type', 'unknown') if r.get('vehicle') else 'unknown',
                'bbox': r['box'],
            }
            results.append(result)
        return results
    else:
        # print(f'  ❌ API Error {response.status_code}: {response.text}')
        return []

def process_video_plate_recognizer(video_path, skip_frames=30):
    """Process video file using Plate Recognizer API."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        return

    print(f"🎬 Starting Plate Recognizer Video Analysis...")
    print(f"🎞️ Video: {video_path}")
    print(f"📡 API Key: {PLATE_RECOGNIZER_API_KEY[:6]}...{PLATE_RECOGNIZER_API_KEY[-4:]}")
    print("-" * 50)

    frame_count = 0
    total_detections = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process every Nth frame to save API quota
        if frame_count % skip_frames == 0:
            temp_path = f"temp_plate_rec_{frame_count}.jpg"
            cv2.imwrite(temp_path, frame)
            
            results = plate_recognizer(temp_path, PLATE_RECOGNIZER_API_KEY)
            
            if results:
                for res in results:
                    print(f"Frame {frame_count:04d}: ✅ Detected {res['plate']} "
                          f"({res['region']}) | Score: {res['score']:.2f} | "
                          f"Vehicle: {res['vehicle_type']}")
                    total_detections += 1
            else:
                print(f"Frame {frame_count:04d}: (no plate found)")

            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

        frame_count += 1

    cap.release()
    print("-" * 50)
    print(f"✅ Finished! Total plates detected: {total_detections}")

if __name__ == "__main__":
    # Default video file from project
    vid = "Automatic Number Plate Recognition (ANPR) _ Vehicle Number Plate Recognition (1).mp4"
    
    if not os.path.exists(vid):
        # Check in test/ folder as well
        vid = os.path.join("test", vid)

    if os.path.exists(vid):
        # skip_frames=60 means approximately every 2 seconds for a 30fps video
        process_video_plate_recognizer(vid, skip_frames=60)
    else:
        print(f"❌ Video not found: {vid}")
