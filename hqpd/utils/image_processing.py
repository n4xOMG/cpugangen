#!/usr/bin/env python3
"""
Image processing utilities for intelligent dataset preprocessing.

Provides functions for:
- Aspect ratio analysis and bucketing
- Smart cropping with anime face detection
- Resize & pad with mask generation
- Multi-tile generation
"""

from pathlib import Path
from typing import Tuple, List, Optional, Dict
import numpy as np
from PIL import Image
import torch
import cv2


def calculate_aspect_ratio(image_path: Path) -> float:
    """
    Calculate aspect ratio (width / height) of an image.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Aspect ratio as float
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            return width / height
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return 1.0  # Default to square


def classify_aspect_bucket(ratio: float) -> str:
    """
    Classify aspect ratio into preprocessing buckets.
    
    Buckets:
    - 'square': 0.85 - 1.15 (near-square, safe for center crop)
    - 'moderate': 0.7-0.85 or 1.15-1.5 (needs smart crop or pad)
    - 'extreme': <0.7 or >1.5 (very tall/wide, needs special handling)
    
    Args:
        ratio: Aspect ratio (width / height)
        
    Returns:
        Bucket name as string
    """
    if 0.85 <= ratio <= 1.15:
        return 'square'
    elif (0.7 <= ratio < 0.85) or (1.15 < ratio <= 1.5):
        return 'moderate'
    else:
        return 'extreme'


def smart_crop_with_face(
    image: Image.Image,
    target_size: int,
    cascade_path: Path,
    scale_factor: float = 1.1,
    min_neighbors: int = 5
) -> Tuple[Image.Image, Optional[Dict]]:
    """
    Crop image centered on detected anime face.
    
    Args:
        image: PIL Image
        target_size: Target square size (e.g., 1024)
        cascade_path: Path to lbpcascade_animeface.xml
        scale_factor: Face detection scale factor (lower = more sensitive)
        min_neighbors: Minimum neighbors for detection (higher = stricter)
        
    Returns:
        Tuple of (cropped_image, metadata)
        metadata contains crop coordinates and detection info
    """
    # Convert PIL to OpenCV format
    img_array = np.array(image)
    if len(img_array.shape) == 2:  # Grayscale
        gray = img_array
    else:  # RGB or RGBA
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Load cascade
    if not cascade_path.exists():
        print(f"Warning: Cascade not found at {cascade_path}")
        return center_crop(image, target_size), None
    
    cascade = cv2.CascadeClassifier(str(cascade_path))
    
    # Detect faces
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(20, 20)
    )
    
    width, height = image.size
    
    # If faces detected, crop around largest face
    if len(faces) > 0:
        # Get largest face
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        
        # Calculate center of face
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        
        # Calculate crop box centered on face
        half_size = target_size // 2
        left = max(0, face_center_x - half_size)
        top = max(0, face_center_y - half_size)
        right = min(width, left + target_size)
        bottom = min(height, top + target_size)
        
        # Adjust if crop box is out of bounds
        if right - left < target_size:
            left = max(0, right - target_size)
        if bottom - top < target_size:
            top = max(0, bottom - target_size)
        
        cropped = image.crop((left, top, right, bottom))
        
        # Resize if needed (handles edge cases near image boundaries)
        if cropped.size != (target_size, target_size):
            cropped = cropped.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        metadata = {
            'face_detected': True,
            'face_count': len(faces),
            'crop_box': (left, top, right, bottom),
            'face_center': (face_center_x, face_center_y)
        }
        
        return cropped, metadata
    
    # No face detected, fall back to center crop
    else:
        cropped = center_crop(image, target_size)
        metadata = {
            'face_detected': False,
            'face_count': 0,
            'crop_box': 'center',
            'face_center': None
        }
        return cropped, metadata


def center_crop(image: Image.Image, target_size: int) -> Image.Image:
    """
    Simple center crop to target size.
    
    Args:
        image: PIL Image
        target_size: Target square size
        
    Returns:
        Cropped PIL Image
    """
    width, height = image.size
    
    # Resize so smaller dimension equals target_size
    if width < height:
        new_width = target_size
        new_height = int(height * target_size / width)
    else:
        new_height = target_size
        new_width = int(width * target_size / height)
    
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Center crop
    left = (new_width - target_size) // 2
    top = (new_height - target_size) // 2
    right = left + target_size
    bottom = top + target_size
    
    return image.crop((left, top, right, bottom))


def resize_and_pad(
    image: Image.Image,
    target_size: int,
    pad_color: int = 0
) -> Tuple[Image.Image, torch.Tensor]:
    """
    Resize image preserving aspect ratio and pad to square.
    
    Args:
        image: PIL Image
        target_size: Target square size (e.g., 1024)
        pad_color: Padding color (0-255 for grayscale, or RGB tuple)
        
    Returns:
        Tuple of (padded_image, mask)
        mask is 1 for image area, 0 for padding
    """
    width, height = image.size
    
    # Resize so longer side equals target_size
    if width > height:
        new_width = target_size
        new_height = int(height * target_size / width)
    else:
        new_height = target_size
        new_width = int(width * target_size / height)
    
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create padded image
    if isinstance(pad_color, int):
        # Grayscale or single color
        mode = image.mode
        if mode == 'RGB' or mode == 'RGBA':
            pad_color = (pad_color, pad_color, pad_color)
    
    padded = Image.new(image.mode, (target_size, target_size), pad_color)
    
    # Paste resized image in center
    paste_x = (target_size - new_width) // 2
    paste_y = (target_size - new_height) // 2
    padded.paste(resized, (paste_x, paste_y))
    
    # Create mask (1 for image, 0 for padding)
    mask = torch.zeros((target_size, target_size), dtype=torch.float32)
    mask[paste_y:paste_y + new_height, paste_x:paste_x + new_width] = 1.0
    
    return padded, mask


def create_tiles(
    image: Image.Image,
    tile_size: int,
    overlap: int = 128
) -> List[Dict]:
    """
    Split large image into overlapping tiles for multi-tile training.
    
    Args:
        image: PIL Image
        tile_size: Size of each tile (e.g., 1024)
        overlap: Overlap between tiles in pixels
        
    Returns:
        List of dicts with 'tile' (PIL Image) and 'position' (x, y)
    """
    width, height = image.size
    tiles = []
    
    stride = tile_size - overlap
    
    for y in range(0, height - tile_size + 1, stride):
        for x in range(0, width - tile_size + 1, stride):
            tile = image.crop((x, y, x + tile_size, y + tile_size))
            tiles.append({
                'tile': tile,
                'position': (x, y),
                'size': (tile_size, tile_size)
            })
    
    # Handle remaining edges if image doesn't divide evenly
    # Right edge
    if width % stride != 0:
        x = width - tile_size
        for y in range(0, height - tile_size + 1, stride):
            tile = image.crop((x, y, x + tile_size, y + tile_size))
            tiles.append({
                'tile': tile,
                'position': (x, y),
                'size': (tile_size, tile_size)
            })
    
    # Bottom edge
    if height % stride != 0:
        y = height - tile_size
        for x in range(0, width - tile_size + 1, stride):
            tile = image.crop((x, y, x + tile_size, y + tile_size))
            tiles.append({
                'tile': tile,
                'position': (x, y),
                'size': (tile_size, tile_size)
            })
    
    # Bottom-right corner (if both edges incomplete)
    if width % stride != 0 and height % stride != 0:
        x = width - tile_size
        y = height - tile_size
        tile = image.crop((x, y, x + tile_size, y + tile_size))
        tiles.append({
            'tile': tile,
            'position': (x, y),
            'size': (tile_size, tile_size)
        })
    
    return tiles


def create_padding_mask(
    original_size: Tuple[int, int],
    target_size: int
) -> torch.Tensor:
    """
    Create a binary mask for padded images.
    
    Args:
        original_size: (width, height) of original image
        target_size: Target square size
        
    Returns:
        Binary mask tensor (target_size, target_size)
        1 for image area, 0 for padding
    """
    width, height = original_size
    mask = torch.zeros((target_size, target_size), dtype=torch.float32)
    
    # Calculate resize dimensions
    if width > height:
        new_width = target_size
        new_height = int(height * target_size / width)
    else:
        new_height = target_size
        new_width = int(width * target_size / height)
    
    # Calculate paste position
    paste_x = (target_size - new_width) // 2
    paste_y = (target_size - new_height) // 2
    
    # Set image area to 1
    mask[paste_y:paste_y + new_height, paste_x:paste_x + new_width] = 1.0
    
    return mask


if __name__ == "__main__":
    """Quick test of utilities"""
    print("Image Processing Utilities")
    print("="*60)
    
    # Test aspect ratio classification
    test_ratios = [0.5, 0.8, 1.0, 1.3, 2.0]
    print("\nAspect Ratio Bucketing:")
    for ratio in test_ratios:
        bucket = classify_aspect_bucket(ratio)
        print(f"  Ratio {ratio:.2f} → {bucket}")
    
    # Test padding mask generation
    print("\nPadding Mask Generation:")
    mask = create_padding_mask((800, 1200), 1024)
    print(f"  Mask shape: {mask.shape}")
    print(f"  Image area: {mask.sum().item()} pixels")
    print(f"  Padding area: {(mask == 0).sum().item()} pixels")
    
    print("\n✅ Utilities loaded successfully!")
