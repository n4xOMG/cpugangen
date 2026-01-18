#!/usr/bin/env python3
"""
Download anime face detector cascade for smart cropping.

Downloads lbpcascade_animeface.xml from nagadomi/lbpcascade_animeface repository.
"""

import os
import urllib.request
from pathlib import Path
import hashlib


CASCADE_URL = "https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml"
EXPECTED_MD5 = "8a6f87c7d9ea6b5f6d1b0a553ef93a18"  # Updated MD5 hash (file was updated on GitHub)


def download_file(url: str, output_path: Path) -> bool:
    """Download file from URL to output path."""
    try:
        print(f"Downloading from {url}...")
        urllib.request.urlretrieve(url, output_path)
        print(f"✅ Downloaded to {output_path}")
        return True
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def verify_md5(file_path: Path, expected_md5: str) -> bool:
    """Verify file MD5 hash."""
    md5_hash = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    
    actual_md5 = md5_hash.hexdigest()
    if actual_md5 == expected_md5:
        print(f"✅ MD5 verification passed: {actual_md5}")
        return True
    else:
        print(f"❌ MD5 mismatch!")
        print(f"   Expected: {expected_md5}")
        print(f"   Got:      {actual_md5}")
        return False


def main():
    # Setup paths
    script_dir = Path(__file__).parent.parent  # cpugangen/
    cascade_dir = script_dir / "hqpd" / "models" / "cascades"
    cascade_file = cascade_dir / "lbpcascade_animeface.xml"
    
    # Create directory
    cascade_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already exists
    if cascade_file.exists():
        print(f"Cascade file already exists: {cascade_file}")
        print("Verifying integrity...")
        if verify_md5(cascade_file, EXPECTED_MD5):
            print("✅ File is valid. No download needed.")
            return
        else:
            print("⚠️  File is corrupted. Re-downloading...")
            cascade_file.unlink()
    
    # Download
    print("\n" + "="*60)
    print("Downloading Anime Face Detector Cascade")
    print("="*60)
    print(f"Source: {CASCADE_URL}")
    print(f"Target: {cascade_file}\n")
    
    if not download_file(CASCADE_URL, cascade_file):
        print("\n❌ Download failed. Please check your internet connection.")
        return
    
    # Verify
    print("\nVerifying file integrity...")
    if not verify_md5(cascade_file, EXPECTED_MD5):
        print("\n❌ Downloaded file is corrupted. Please try again.")
        cascade_file.unlink()
        return
    
    print("\n" + "="*60)
    print("✅ Setup Complete!")
    print("="*60)
    print(f"Anime face detector ready at:")
    print(f"  {cascade_file}")
    print("\nYou can now use smart cropping strategy in preprocessing.")


if __name__ == "__main__":
    main()
