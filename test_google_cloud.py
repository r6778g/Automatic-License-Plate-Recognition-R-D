import os
import sys
from dotenv import load_dotenv
from google_cloud_alpr import GoogleCloudALPR

# Explicitly load .env
load_dotenv()

def main():
    # Credentials will now be automatically picked up by GoogleCloudALPR 
    # from environment variables (via load_dotenv)
    alpr = GoogleCloudALPR()
    
    if not alpr.api_key and not alpr.client:
        print("❌ Error: Please set GOOGLE_CLOUD_API_KEY or GOOGLE_APPLICATION_CREDENTIALS in .env")
        return

    print(f"📡 Using {'Service Account' if alpr.client else 'API Key'} mode\n")

    # 1. Test Image
    test_img = "test/image.png"
    if not os.path.exists(test_img):
        # Fallback to any image in test/
        if os.path.exists("test"):
            for f in os.listdir("test"):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    test_img = os.path.join("test", f)
                    break

    if os.path.exists(test_img):
        print(f"🖼️  Testing image detection on: {test_img}...")
        results = alpr.detect_plate_image(test_img)
        if results:
            print(f"   ✅ Success! Detected: {results}")
        else:
            print("   ℹ️ No license plates detected in the image.")
    else:
        print(f"⚠️  No test image found (tried {test_img})")

    # 2. Test Video
    test_vid = "Automatic Number Plate Recognition (ANPR) _ Vehicle Number Plate Recognition (1).mp4"
    if not os.path.exists(test_vid):
        test_vid = os.path.join("test", test_vid)

    if os.path.exists(test_vid):
        print(f"\n🎥 Testing video detection on: {test_vid}...")
        print("   (Sampling 1 frame every 2 seconds to save on API quota)")
        vid_results = alpr.detect_plate_video(test_vid, skip_frames=100)
        
        if vid_results:
            print(f"   ✅ Success! Found {len(vid_results)} detections.")
            for r in vid_results[:60]: # Show first 5
                print(f"      - Frame {r['frame']}: {r['text']}")
            if len(vid_results) > 60:
                print(f"      - ... and {len(vid_results) - 60} more.")
        else:
            print("   ℹ️ No license plates detected in the video.")
    else:
        print(f"\n⚠️  Test video not found at {test_vid}")

if __name__ == "__main__":
    main()
