import cv2
import os
import re
import requests
import base64
from google.cloud import vision
from dotenv import load_dotenv

# Load environment variables from .env if it exists
load_dotenv()

class GoogleCloudALPR:
    def __init__(self, api_key=None, credentials_path=None):
        self.api_key = api_key or os.environ.get("GOOGLE_CLOUD_API_KEY")
        cred_path = credentials_path or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        
        if cred_path and os.path.exists(cred_path):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path
            self.client = vision.ImageAnnotatorClient()
        else:
            self.client = None

    def detect_plate_image(self, image_path):
        """Detect plate text using Google Cloud Vision."""
        if self.client:
            # Use official client library (Service Account)
            with open(image_path, 'rb') as f:
                image = vision.Image(content=f.read())
            response = self.client.text_detection(image=image)
            texts = response.text_annotations
            results = []
            for i, text in enumerate(texts):
                if i == 0: continue  # Skip the full-image text
                cleaned = re.sub(r'[^A-Z0-9]', '', text.description.upper())
                if 3 <= len(cleaned) <= 10:
                    results.append({'text': cleaned})
            return results
        
        elif self.api_key:
            # Fallback to REST API (API Key)
            url = f"https://vision.googleapis.com/v1/images:annotate?key={self.api_key}"
            with open(image_path, 'rb') as f:
                img_content = base64.b64encode(f.read()).decode()
            
            payload = {
                "requests": [{
                    "image": {"content": img_content},
                    "features": [{"type": "TEXT_DETECTION"}]
                }]
            }
            try:
                response = requests.post(url, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    res = data['responses'][0].get('textAnnotations', [])
                    results = []
                    for i, text in enumerate(res):
                        if i == 0: continue
                        cleaned = re.sub(r'[^A-Z0-9]', '', text['description'].upper())
                        if 3 <= len(cleaned) <= 10:
                            results.append({'text': cleaned})
                    return results
                else:
                    return []
            except Exception as e:
                print(f"❌ Network Error: {e}")
                return []
        
        return []

    def detect_plate_video(self, video_path, skip_frames=60):
        """Analyze video by sampling frames."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        all_results = []
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if frame_count % skip_frames == 0:
                temp_path = f"temp_frame_{frame_count}.jpg"
                cv2.imwrite(temp_path, frame)
                res = self.detect_plate_image(temp_path)
                if res:
                    for r in res:
                        r['frame'] = frame_count
                        all_results.append(r)
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            frame_count += 1
        cap.release()
        return all_results

def analyze_video_vision(video_path, api_key=None, credentials_path=None, skip_frames=30):
    alpr = GoogleCloudALPR(api_key=api_key, credentials_path=credentials_path)
    
    if not alpr.api_key and not alpr.client:
        print("❌ Error: No Google Cloud API Key or Credentials found.")
        print("Please check your .env file or environment variables.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file {video_path}")
        return

    print(f"🎬 Starting analysis on: {video_path}")
    print(f"📡 Using {'Service Account' if alpr.client else 'API Key'} for detection")

    frame_count = 0
    total_detections = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        if frame_count % skip_frames == 0:
            temp_path = "temp_v_frame.jpg"
            cv2.imwrite(temp_path, frame)
            
            results = alpr.detect_plate_image(temp_path)
            if results:
                print(f"  Frame {frame_count}: ✅ Detected {results[0]['text']}")
                total_detections += 1
            else:
                print(f"  Frame {frame_count}: (no plate detected)")
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
        frame_count += 1
    
    cap.release()
    print(f"✅ Finished! Total plates found: {total_detections}")

if __name__ == "__main__":
    vid = "Automatic Number Plate Recognition (ANPR) _ Vehicle Number Plate Recognition (1).mp4"
    if os.path.exists(vid):
        analyze_video_vision(vid)
    else:
        # Try finding it in test/ folder
        alt_vid = os.path.join("test", vid)
        if os.path.exists(alt_vid):
            analyze_video_vision(alt_vid)
        else:
            print(f"⚠️ Video file not found: {vid}")
