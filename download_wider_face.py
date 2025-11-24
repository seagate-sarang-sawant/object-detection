#!/usr/bin/env python3
"""Helper script to download WIDER FACE dataset files from Google Drive using gdown.

This script helps work around Google Drive download issues with TensorFlow Datasets.
Run this script if you encounter "Failed to obtain confirmation link" errors.
"""

import sys
from pathlib import Path

try:
    import gdown
except ImportError:
    print("ERROR: gdown is not installed.")
    print("Install it with: pip install gdown")
    sys.exit(1)


# WIDER FACE Google Drive file IDs (from TFDS)
WIDER_FACE_FILES = {
    "WIDER_train": "15hGDLhsx8bLgLcIRD5DhYt5iSHgxdwGyU5FhB8ulgYEfJFlNDsVPJQQuPH5Upfjm36I",
    "WIDER_val": "1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q",
    "WIDER_test": "1HIfDbVEWKmsYKJZm4lchTBDLfr62-cNGXcnoWarcPOgb67igMZT4ssm73xhXi-__9lo",
    "wider_face_split": "1sAl2oml7hK6aZRdgRjqQJsjVdBnmIqhbodwusV3aejHpctHJSPjDt1acOcAWJ2U7VDY",
}


def download_file(file_id: str, output_path: Path, description: str = "") -> None:
    """Download a file from Google Drive using gdown.
    
    Args:
        file_id: Google Drive file ID
        output_path: Where to save the downloaded file
        description: Description of the file being downloaded
    """
    url = f"https://drive.google.com/uc?id={file_id}"
    
    print(f"\n{'='*80}")
    print(f"Downloading: {description or output_path.name}")
    print(f"URL: {url}")
    print(f"Output: {output_path}")
    print(f"{'='*80}\n")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        gdown.download(url, str(output_path), quiet=False)
        
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"\n✓ Successfully downloaded {output_path.name} ({size_mb:.2f} MB)")
        else:
            print(f"\n✗ Download failed: {output_path} does not exist")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        sys.exit(1)


def main():
    """Main function to download WIDER FACE files."""
    # Determine download directory (same as TFDS default)
    data_dir = Path("data")
    download_dir = data_dir / "wider_face" / "downloads"
    
    print("WIDER FACE Dataset Downloader")
    print("=" * 80)
    print(f"Download directory: {download_dir}")
    print("\nThis script will download the following files:")
    for name, file_id in WIDER_FACE_FILES.items():
        print(f"  - {name} (ID: {file_id})")
    
    response = input("\nProceed with download? (y/n): ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        sys.exit(0)
    
    # Download each file
    for name, file_id in WIDER_FACE_FILES.items():
        output_path = download_dir / f"{name}.zip"
        
        # Skip if file already exists
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"\n⏭ Skipping {name}.zip (already exists, {size_mb:.2f} MB)")
            continue
        
        download_file(file_id, output_path, name)
    
    print("\n" + "=" * 80)
    print("Download complete!")
    print(f"Files saved to: {download_dir}")
    print("\nYou can now retry loading the dataset in your notebook.")
    print("=" * 80)


if __name__ == "__main__":
    main()

