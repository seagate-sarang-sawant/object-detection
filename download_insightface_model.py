#!/usr/bin/env python3
"""Helper script to manually download insightface models if automatic download fails."""

import os
import sys
from pathlib import Path
import requests
from tqdm import tqdm

def download_model(model_name: str = "buffalo_l", root: str = "~/.insightface"):
    """Manually download an insightface model."""
    root = Path(root).expanduser()
    model_dir = root / "models" / model_name
    zip_path = root / "models" / f"{model_name}.zip"
    
    # Try multiple possible URLs
    base_urls = [
        f"https://github.com/deepinsight/insightface/releases/download/v0.7/{model_name}.zip",
        f"https://github.com/deepinsight/insightface/releases/download/v0.7.3/{model_name}.zip",
        f"https://github.com/deepinsight/insightface/releases/download/latest/{model_name}.zip",
    ]
    
    print(f"Attempting to download {model_name} model...")
    print(f"Target directory: {model_dir}")
    
    for url in base_urls:
        try:
            print(f"\nTrying URL: {url}")
            response = requests.get(url, stream=True, timeout=30)
            if response.status_code == 200:
                print(f"✓ Found model at: {url}")
                total_size = int(response.headers.get('content-length', 0))
                
                # Create directory if it doesn't exist
                zip_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Download with progress bar
                with open(zip_path, 'wb') as f, tqdm(
                    desc=f"Downloading {model_name}",
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
                
                print(f"\n✓ Downloaded to: {zip_path}")
                print(f"Extracting to: {model_dir}...")
                
                # Extract zip file
                import zipfile
                model_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(model_dir)
                
                print(f"✓ Model extracted successfully!")
                print(f"Model location: {model_dir}")
                return True
            else:
                print(f"✗ Status code {response.status_code}")
        except Exception as e:
            print(f"✗ Error: {e}")
            continue
    
    print(f"\n✗ Failed to download {model_name} from all attempted URLs.")
    print("\nAlternative: You can try:")
    print("1. Check https://github.com/deepinsight/insightface/releases for available models")
    print("2. Manually download and extract to:", model_dir)
    print("3. Use a different model name (buffalo_l, buffalo_s, antelopev2)")
    return False

if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "buffalo_l"
    download_model(model_name)

