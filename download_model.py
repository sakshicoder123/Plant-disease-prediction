import urllib.request
import urllib.parse
import re
import os
import sys
from pathlib import Path

def download_model():
    file_id = '1GWJ5HC8LxQsExmHEZWf7q89MRUquqt7U'
    current_dir = Path(__file__).resolve().parent
    
    # Destination in Flask Deployed App
    if (current_dir / 'Flask Deployed App').exists():
        destination = current_dir / 'Flask Deployed App' / 'plant_disease_model_1_latest.pt'
    else:
        destination = current_dir / 'plant_disease_model_1_latest.pt'

    if destination.exists() and destination.stat().st_size > 100 * 1024 * 1024:
        print(f"Model already present at {destination} ({destination.stat().st_size} bytes)")
        return True

    print(f"Downloading pre-trained weights to {destination} (~200MB)...")
    url = f"https://docs.google.com/uc?export=download&id={file_id}"
    cookie_jar = urllib.request.HTTPCookieProcessor()
    opener = urllib.request.build_opener(cookie_jar)

    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    response = opener.open(req)
    html = response.read().decode('utf-8', errors='ignore')

    form_match = re.search(r'<form\s+id="download-form"\s+action="([^"]+)"[^>]*>(.*?)</form>', html, re.DOTALL)
    if form_match:
        action_url = form_match.group(1)
        inputs = re.findall(r'<input\s+type="hidden"\s+name="([^"]+)"\s+value="([^"]*)"', form_match.group(2))
        params = dict(inputs)
        query_str = urllib.parse.urlencode(params)
        final_url = f"{action_url}?{query_str}"
    else:
        final_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"

    req2 = urllib.request.Request(final_url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    with opener.open(req2) as resp, open(destination, 'wb') as out_f:
        total_len = resp.headers.get('content-length')
        total_len = int(total_len) if total_len else None
        downloaded = 0
        chunk_size = 1024 * 1024
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_f.write(chunk)
            downloaded += len(chunk)
            if total_len:
                print(f"Progress: {downloaded / (1024*1024):.1f}MB / {total_len / (1024*1024):.1f}MB ({downloaded*100/total_len:.1f}%)", end='\r')
            else:
                print(f"Progress: {downloaded / (1024*1024):.1f}MB", end='\r')

    print(f"\nDownload completed: {destination} ({destination.stat().st_size} bytes)")
    return True

if __name__ == '__main__':
    download_model()
