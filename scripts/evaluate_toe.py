"""
Evaluate and benchmark trained Tag-Optimized Encoder (TOE) against SDXL CLIP.

This script:
1. Loads trained TOE model
2. Loads SDXL CLIP encoders (teacher)
3. Tests on various tag combinations
4. Computes quality metrics (cosine similarity, MSE)
5. Benchmarks inference speed
6. Generates evaluation report
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from diffusers import StableDiffusionXLPipeline

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hqpd.models.toe import TagOptimizedEncoder
from hqpd.utils.danbooru import DanbooruTagProcessor


def load_toe_model(checkpoint_path: str, config: dict, device: str = "cuda") -> TagOptimizedEncoder:
    """Load trained TOE model from checkpoint."""
    print(f"Loading TOE model from {checkpoint_path}")
    
    # Initialize model
    model = TagOptimizedEncoder(
        vocab_size=config['vocab_size'],
        embed_dim=config['embed_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        mlp_ratio=config['mlp_ratio'],
        max_length=config['max_length'],
        quantize_embeddings=config.get('quantize_embeddings', False)
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ TOE loaded: {model.get_num_params():,} params, {model.get_model_size_mb():.2f} MB")
    
    return model


def load_sdxl_clip(model_name: str = "stabilityai/stable-diffusion-xl-base-1.0", device: str = "cuda"):
    """Load SDXL CLIP text encoders (teacher model)."""
    print(f"Loading SDXL CLIP encoders...")
    
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=True,
        variant="fp16" if device == "cuda" else None
    )
    
    tokenizer_1 = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2
    text_encoder_1 = pipe.text_encoder.to(device)
    text_encoder_2 = pipe.text_encoder_2.to(device)
    
    text_encoder_1.eval()
    text_encoder_2.eval()
    
    print(f"✓ SDXL CLIP loaded")
    
    # Cleanup
    del pipe.unet, pipe.vae, pipe.scheduler
    import gc
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    
    return tokenizer_1, tokenizer_2, text_encoder_1, text_encoder_2


def encode_with_clip(prompt: str, tokenizers: tuple, text_encoders: tuple, device: str = "cuda") -> torch.Tensor:
    """Encode prompt with SDXL CLIP."""
    tokenizer_1, tokenizer_2 = tokenizers
    text_encoder_1, text_encoder_2 = text_encoders
    
    with torch.no_grad():
        # Tokenize
        text_inputs_1 = tokenizer_1(
            prompt, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt"
        )
        text_inputs_2 = tokenizer_2(
            prompt, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt"
        )
        
        # Encode
        emb_1 = text_encoder_1(text_inputs_1.input_ids.to(device))
        emb_2 = text_encoder_2(text_inputs_2.input_ids.to(device))
        
        # Concatenate (matching training)
        hidden_1 = emb_1.last_hidden_state
        hidden_2 = emb_2.last_hidden_state
        combined = torch.cat([hidden_1, hidden_2], dim=-1)
        
        # NOTE: No normalization here - training embeddings have norm ~8.77
        # TOE learned this scale, so we keep it for fair comparison
        
        return combined  # (1, 77, 2048)


def encode_with_toe(tags: List[str], toe_model: TagOptimizedEncoder, 
                    tag_processor: DanbooruTagProcessor, device: str = "cuda") -> torch.Tensor:
    """Encode tags with TOE."""
    with torch.no_grad():
        # Encode tags
        tag_ids, tag_weights = tag_processor.encode(tags, max_length=77)
        tag_ids = torch.tensor([tag_ids], dtype=torch.long).to(device)
        tag_weights = torch.tensor([tag_weights], dtype=torch.float32).to(device)
        
        # Forward pass
        embeddings = toe_model(tag_ids, tag_weights)
        
        # NOTE: No normalization - TOE outputs match training scale (~8.77 norm)
        # This is the scale the model learned during training
        
        return embeddings  # (1, 77, 2048)


def compute_metrics(toe_emb: torch.Tensor, clip_emb: torch.Tensor) -> Dict[str, float]:
    """Compute quality metrics between TOE and CLIP embeddings."""
    # Flatten for per-element metrics
    toe_flat = toe_emb.reshape(-1)
    clip_flat = clip_emb.reshape(-1)
    
    # MSE
    mse = F.mse_loss(toe_flat, clip_flat).item()
    
    # Cosine similarity (higher is better)
    cos_sim = F.cosine_similarity(toe_flat.unsqueeze(0), clip_flat.unsqueeze(0)).item()
    
    # L2 distance
    l2_dist = torch.norm(toe_flat - clip_flat, p=2).item()
    
    return {
        'mse': mse,
        'cosine_similarity': cos_sim,
        'l2_distance': l2_dist
    }


def benchmark_speed(
    test_cases: List[List[str]], 
    toe_model: TagOptimizedEncoder,
    tag_processor: DanbooruTagProcessor,
    clip_encoders: tuple,
    device: str = "cuda",
    num_runs: int = 100
) -> Dict[str, float]:
    """Benchmark inference speed."""
    print(f"\nBenchmarking inference speed ({num_runs} runs)...")
    
    tokenizers, text_encoders = clip_encoders
    
    # Warmup
    for tags in test_cases[:5]:
        _ = encode_with_toe(tags, toe_model, tag_processor, device)
        prompt = ", ".join(tags)
        _ = encode_with_clip(prompt, tokenizers, text_encoders, device)
    
    # Benchmark TOE
    torch.cuda.synchronize() if device == "cuda" else None
    toe_times = []
    for _ in range(num_runs):
        tags = test_cases[_ % len(test_cases)]
        start = time.perf_counter()
        _ = encode_with_toe(tags, toe_model, tag_processor, device)
        torch.cuda.synchronize() if device == "cuda" else None
        toe_times.append(time.perf_counter() - start)
    
    # Benchmark CLIP
    torch.cuda.synchronize() if device == "cuda" else None
    clip_times = []
    for _ in range(num_runs):
        tags = test_cases[_ % len(test_cases)]
        prompt = ", ".join(tags)
        start = time.perf_counter()
        _ = encode_with_clip(prompt, tokenizers, text_encoders, device)
        torch.cuda.synchronize() if device == "cuda" else None
        clip_times.append(time.perf_counter() - start)
    
    return {
        'toe_mean_ms': np.mean(toe_times) * 1000,
        'toe_std_ms': np.std(toe_times) * 1000,
        'clip_mean_ms': np.mean(clip_times) * 1000,
        'clip_std_ms': np.std(clip_times) * 1000,
        'speedup': np.mean(clip_times) / np.mean(toe_times)
    }


def generate_test_cases(vocabulary_path: str, num_cases: int = 50) -> List[List[str]]:
    """Generate diverse test cases."""
    with open(vocabulary_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    if 'top_tags' in vocab_data:
        available_tags = [tag['name'] for tag in vocab_data['top_tags'][:500]]
    else:
        available_tags = [
            tag for tag in list(vocab_data['tag_to_id'].keys())[:500]
            if tag not in ['<pad>', '<unk>', '<eos>']
        ]
    
    import random
    test_cases = []
    
    # Varied lengths
    for _ in range(num_cases):
        num_tags = random.randint(3, 15)
        tags = random.sample(available_tags, min(num_tags, len(available_tags)))
        test_cases.append(tags)
    
    return test_cases


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained TOE model")
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/toe/toe_best.pt',
        help='Path to TOE checkpoint'
    )
    parser.add_argument(
        '--vocab-path',
        type=str,
        default='data/vocabulary.json',
        help='Path to vocabulary.json'
    )
    parser.add_argument(
        '--sdxl-model',
        type=str,
        default='stabilityai/stable-diffusion-xl-base-1.0',
        help='SDXL model name'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device (cuda or cpu)'
    )
    parser.add_argument(
        '--num-test-cases',
        type=int,
        default=50,
        help='Number of test cases'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='evaluation_report.json',
        help='Output report path'
    )
    
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU")
        args.device = 'cpu'
    
    print("=" * 70)
    print("TOE Model Evaluation")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Vocabulary: {args.vocab_path}")
    print(f"Device: {args.device}")
    print("=" * 70)
    
    # Load tag processor
    tag_processor = DanbooruTagProcessor(vocab_size=15000)
    tag_processor.load_vocabulary(args.vocab_path)
    
    # Load TOE model
    toe_config = {
        'vocab_size': 15000,
        'embed_dim': 2048,
        'num_layers': 4,
        'num_heads': 8,
        'mlp_ratio': 2,
        'max_length': 77,
        'quantize_embeddings': False
    }
    toe_model = load_toe_model(args.checkpoint, toe_config, args.device)
    
    # Load SDXL CLIP
    tokenizer_1, tokenizer_2, text_encoder_1, text_encoder_2 = load_sdxl_clip(
        args.sdxl_model, args.device
    )
    
    # Generate test cases
    print(f"\nGenerating {args.num_test_cases} test cases...")
    test_cases = generate_test_cases(args.vocab_path, args.num_test_cases)
    
    # Evaluate quality
    print(f"\nEvaluating quality on {len(test_cases)} test cases...")
    all_metrics = []
    
    for tags in tqdm(test_cases, desc="Quality evaluation"):
        prompt = ", ".join(tags)
        
        # Encode with both models
        toe_emb = encode_with_toe(tags, toe_model, tag_processor, args.device)
        clip_emb = encode_with_clip(
            prompt, 
            (tokenizer_1, tokenizer_2),
            (text_encoder_1, text_encoder_2),
            args.device
        )
        
        # Compute metrics
        metrics = compute_metrics(toe_emb, clip_emb)
        all_metrics.append(metrics)
    
    # Aggregate quality metrics
    avg_metrics = {
        'mse': np.mean([m['mse'] for m in all_metrics]),
        'cosine_similarity': np.mean([m['cosine_similarity'] for m in all_metrics]),
        'l2_distance': np.mean([m['l2_distance'] for m in all_metrics])
    }
    
    # Benchmark speed
    speed_metrics = benchmark_speed(
        test_cases,
        toe_model,
        tag_processor,
        ((tokenizer_1, tokenizer_2), (text_encoder_1, text_encoder_2)),
        args.device
    )
    
    # Generate report
    report = {
        'checkpoint': args.checkpoint,
        'num_test_cases': len(test_cases),
        'device': args.device,
        'quality_metrics': avg_metrics,
        'speed_metrics': speed_metrics,
        'model_stats': {
            'toe_params': toe_model.get_num_params(),
            'toe_size_mb': toe_model.get_model_size_mb()
        }
    }
    
    # Print report
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"\n📊 Quality Metrics (averaged over {len(test_cases)} test cases):")
    print(f"  MSE:                {avg_metrics['mse']:.6f}")
    print(f"  Cosine Similarity:  {avg_metrics['cosine_similarity']:.4f} (higher is better)")
    print(f"  L2 Distance:        {avg_metrics['l2_distance']:.4f}")
    
    print(f"\n⚡ Speed Metrics:")
    print(f"  TOE:   {speed_metrics['toe_mean_ms']:.2f} ± {speed_metrics['toe_std_ms']:.2f} ms")
    print(f"  CLIP:  {speed_metrics['clip_mean_ms']:.2f} ± {speed_metrics['clip_std_ms']:.2f} ms")
    print(f"  Speedup: {speed_metrics['speedup']:.2f}x faster")
    
    print(f"\n💾 Model Size:")
    print(f"  Parameters: {report['model_stats']['toe_params']:,}")
    print(f"  Size: {report['model_stats']['toe_size_mb']:.2f} MB")
    
    print("\n" + "=" * 70)
    
    # Save report
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Report saved to: {output_path}")
    
    # Interpretation
    print("\n📈 Interpretation:")
    if avg_metrics['cosine_similarity'] > 0.9:
        print("  ✓ Excellent! TOE embeddings are very similar to CLIP")
    elif avg_metrics['cosine_similarity'] > 0.8:
        print("  ✓ Good! TOE learned CLIP knowledge reasonably well")
    elif avg_metrics['cosine_similarity'] > 0.7:
        print("  ⚠️  Fair. Consider training longer or with more data")
    else:
        print("  ❌ Poor similarity. Model may need retraining")
    
    if speed_metrics['speedup'] > 2:
        print(f"  ✓ Great speedup! TOE is {speed_metrics['speedup']:.1f}x faster than CLIP")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
