

def plate_recognizer(image_path, api_key):
    """Call Plate Recognizer Cloud API."""
    url = 'https://api.platerecognizer.com/v1/plate-reader/'
    headers = {'Authorization': f'Token {api_key}'}
    
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return []
        
    with open(image_path, 'rb') as f:
        # Note: Added 'upload' file parameter as required by Plate Recognizer
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
            print(f" Plate: {result['plate']} | Conf: {result['score']:.2f} | "
                  f"Region: {result['region']} | Vehicle: {result['vehicle_type']}")
        return results
    else:
        print(f' Error {response.status_code}: {response.text}')
        return []

# --- Run on ALL detected images ---
print(f"🚀 Running Plate Recognizer on {len(IMAGE_PATHS)} images...")
all_results = {}

for img in IMAGE_PATHS:
    print(f"\n📄 Processing: {os.path.basename(img)}")
    results = plate_recognizer(img, PLATE_RECOGNIZER_API_KEY)
    all_results[img] = results

print(f"\n✨ Completed! Successfully processed {len(all_results)} images.")
