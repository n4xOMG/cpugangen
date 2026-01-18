#!/usr/bin/env python3
"""
Prune Teacher UNet to create Student UNet initialization.

This script initializes a Student UNet with the architecture of SSD-1B
and loads weights from a Teacher UNet (e.g., Illustrious) where possible.
This acts as a "smart pruning" or "weight transfer" initialization,
giving the student better initial domain knowledge than random or generic weights.
"""

import argparse
import os
import torch
from diffusers import UNet2DConditionModel, StableDiffusionXLPipeline
from safetensors.torch import save_file

def prune_teacher(teacher_id: str, student_id: str, output_path: str, device: str = "cpu"):
    print(f"\nInitializing Student from Teacher...")
    print(f"  Teacher: {teacher_id}")
    print(f"  Student Arch: {student_id}")
    print(f"  Output: {output_path}")

    # 1. Load Teacher UNet
    print("\nLoading Teacher UNet...")
    teacher_pipe = StableDiffusionXLPipeline.from_pretrained(teacher_id, torch_dtype=torch.float32)
    teacher_unet = teacher_pipe.unet
    teacher_state_dict = teacher_unet.state_dict()
    print(f"  Teacher params: {sum(p.numel() for p in teacher_unet.parameters()):,}")

    # 2. Load Student UNet (Architecture only, weights don't matter yet)
    print("\nLoading Student UNet Architecture...")
    # We load the config/model from the student ID (e.g. segmind/SSD-1B)
    student_unet = UNet2DConditionModel.from_pretrained(student_id, subfolder="unet", torch_dtype=torch.float32)
    student_state_dict = student_unet.state_dict()
    print(f"  Student params: {sum(p.numel() for p in student_unet.parameters()):,}")

    # 3. Transfer Weights
    print("\nTransferring weights...")
    transferred_count = 0
    skipped_count = 0
    
    new_state_dict = {}
    
    for key in student_state_dict.keys():
        if key in teacher_state_dict:
            # Check shape
            student_shape = student_state_dict[key].shape
            teacher_shape = teacher_state_dict[key].shape
            
            if student_shape == teacher_shape:
                new_state_dict[key] = teacher_state_dict[key]
                transferred_count += 1
            else:
                print(f"  Shape mismatch for {key}: Student {student_shape} vs Teacher {teacher_shape} (Skipping)")
                # Keep original student weight (likely random or from SSD-1B if loaded pretrained)
                new_state_dict[key] = student_state_dict[key]
                skipped_count += 1
        else:
            print(f"  Key missing in teacher: {key} (Keeping original)")
            new_state_dict[key] = student_state_dict[key]
            skipped_count += 1
            
    # Load new weights into student
    student_unet.load_state_dict(new_state_dict)
    
    print(f"\nWeight Transfer Complete:")
    print(f"  Transferred: {transferred_count} layers")
    print(f"  Skipped/Kept: {skipped_count} layers")
    
    # 4. Save
    print(f"\nSaving Pruned Student to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    student_unet.save_pretrained(output_path)
    print("✅ Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prune Teacher UNet to Student Architecture")
    parser.add_argument("--teacher", type=str, default="martineux/janku6", help="Teacher model ID")
    parser.add_argument("--student", type=str, default="segmind/SSD-1B", help="Student model ID (for architecture)")
    parser.add_argument("--output", type=str, default="checkpoints/pruned_student_init", help="Output directory")
    args = parser.parse_args()

    prune_teacher(args.teacher, args.student, args.output)
