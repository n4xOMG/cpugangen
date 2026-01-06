#!/usr/bin/env python3
"""
Cleanup script to remove duplicate ONNX data and optimize file sizes.
"""

import argparse
import shutil
from pathlib import Path

import onnx


def cleanup_onnx_model(model_dir: Path):
    """Clean up a single ONNX model directory."""
    onnx_file = model_dir / "model.onnx"
    data_file = model_dir / "model.onnx.data"
    
    if not onnx_file.exists():
        return None
    
    original_size = onnx_file.stat().st_size
    if data_file.exists():
        original_size += data_file.stat().st_size
    
    # Load model
    model = onnx.load(str(onnx_file))
    
    # Backup original
    backup_dir = model_dir / "backup"
    backup_dir.mkdir(exist_ok=True)
    shutil.copy(onnx_file, backup_dir / "model.onnx")
    if data_file.exists():
        shutil.copy(data_file, backup_dir / "model.onnx.data")
    
    # Remove old files
    onnx_file.unlink()
    if data_file.exists():
        data_file.unlink()
    
    # Save with external data (properly this time)
    onnx.save_model(
        model,
        str(onnx_file),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.onnx.data",
        size_threshold=1024,
        convert_attribute=False,
    )
    
    new_size = onnx_file.stat().st_size
    if data_file.exists():
        new_size += data_file.stat().st_size
    
    # Remove backup if successful
    shutil.rmtree(backup_dir)
    
    return original_size, new_size


def main():
    parser = argparse.ArgumentParser(description="Cleanup ONNX models")
    parser.add_argument("--dir", default="onnx_models/sdxl_lightning_fixed/onnx",
                        help="ONNX model directory")
    args = parser.parse_args()
    
    onnx_dir = Path(args.dir)
    
    print("\n" + "=" * 50)
    print("🧹 ONNX Cleanup")
    print("=" * 50)
    
    components = ["unet", "vae_decoder", "text_encoder", "text_encoder_2"]
    
    total_saved = 0
    
    for comp in components:
        comp_dir = onnx_dir / comp
        if not comp_dir.exists():
            continue
        
        print(f"\n📦 Processing {comp}...")
        
        result = cleanup_onnx_model(comp_dir)
        if result:
            orig, new = result
            saved = orig - new
            total_saved += saved
            print(f"   Before: {orig/1024/1024:.1f} MB")
            print(f"   After:  {new/1024/1024:.1f} MB")
            print(f"   Saved:  {saved/1024/1024:.1f} MB")
        else:
            print("   ⚠️ No model.onnx found")
    
    print("\n" + "=" * 50)
    print(f"✅ Total saved: {total_saved/1024/1024:.1f} MB")
    print("=" * 50)
    
    # Show final sizes
    print("\nFinal sizes:")
    for comp in components:
        comp_dir = onnx_dir / comp
        if comp_dir.exists():
            total = sum(f.stat().st_size for f in comp_dir.glob("model.onnx*"))
            print(f"  {comp}: {total/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
