"""
Build vocabulary from Danbooru tags.json file using streaming JSON parsing.
This script processes the large tags.json file incrementally to avoid memory issues.
"""

import json
import argparse
from pathlib import Path
from collections import Counter
from tqdm import tqdm

try:
    import ijson
    HAS_IJSON = True
except ImportError:
    HAS_IJSON = False
    print("WARNING: ijson not found. Will use fallback method (may use more memory)")


def build_vocabulary_streaming(json_path, vocab_size=15000, output_path=None):
    """
    Build vocabulary from tags.json using streaming parser.
    
    Args:
        json_path: Path to tags.json file
        vocab_size: Number of most common tags to keep
        output_path: Path to save vocabulary (default: data/vocabulary.json)
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    if output_path is None:
        output_path = json_path.parent / "vocabulary.json"
    else:
        output_path = Path(output_path)
    
    print(f"Processing {json_path.name} ({json_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"Building vocabulary with top {vocab_size} tags...")
    
    # Collect tag data
    tags_data = []
    
    if HAS_IJSON:
        # Use streaming parser for memory efficiency
        with open(json_path, 'rb') as f:
            # Parse array items one by one
            parser = ijson.items(f, 'item')
            for tag_obj in tqdm(parser, desc="Reading tags"):
                if 'name' in tag_obj and 'post_count' in tag_obj:
                    tags_data.append({
                        'name': tag_obj['name'],
                        'post_count': tag_obj['post_count'],
                        'category': tag_obj.get('category', 0)
                    })
    else:
        # Fallback: load entire JSON (may cause memory issues with very large files)
        print("Using fallback JSON parser...")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for tag_obj in tqdm(data, desc="Reading tags"):
                if 'name' in tag_obj and 'post_count' in tag_obj:
                    tags_data.append({
                        'name': tag_obj['name'],
                        'post_count': tag_obj['post_count'],
                        'category': tag_obj.get('category', 0)
                    })
    
    print(f"\nFound {len(tags_data)} tags in total")
    
    # Sort by post_count (descending) and take top vocab_size
    tags_data.sort(key=lambda x: x['post_count'], reverse=True)
    
    # Show statistics
    print(f"\nTop 10 most common tags:")
    for i, tag in enumerate(tags_data[:10], 1):
        category_name = {0: 'general', 1: 'artist', 3: 'copyright', 4: 'character', 5: 'meta'}.get(tag['category'], 'unknown')
        print(f"  {i:2d}. {tag['name']:30s} ({tag['post_count']:>8,} posts, {category_name})")
    
    # Build vocabulary mapping
    # Reserve IDs 0-2 for special tokens
    tag_to_id = {
        '<pad>': 0,
        '<unk>': 1,
        '<eos>': 2
    }
    
    # Add top tags
    num_tags_to_add = min(vocab_size - 3, len(tags_data))
    for idx, tag_data in enumerate(tags_data[:num_tags_to_add], start=3):
        tag_to_id[tag_data['name']] = idx
    
    # Create vocabulary data structure
    vocabulary = {
        'tag_to_id': tag_to_id,
        'vocab_size': len(tag_to_id),
        'special_tokens': {
            'pad': '<pad>',
            'unk': '<unk>',
            'eos': '<eos>'
        },
        'top_tags': [
            {
                'name': tag['name'],
                'post_count': tag['post_count'],
                'category': tag['category']
            }
            for tag in tags_data[:num_tags_to_add]
        ]
    }
    
    # Save vocabulary
    print(f"\nSaving vocabulary to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocabulary, f, indent=2, ensure_ascii=False)
    
    print(f"\nVocabulary built successfully!")
    print(f"  Total unique tags: {len(tag_to_id):,}")
    print(f"  Special tokens: 3")
    print(f"  Regular tags: {len(tag_to_id) - 3:,}")
    print(f"  Output file: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")
    
    return vocabulary


def main():
    parser = argparse.ArgumentParser(description="Build vocabulary from Danbooru tags.json")
    parser.add_argument(
        '--input',
        type=str,
        default='data/tags.json',
        help='Path to input tags.json file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to output vocabulary.json file (default: data/vocabulary.json)'
    )
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=15000,
        help='Vocabulary size (including special tokens)'
    )
    
    args = parser.parse_args()
    
    # Build vocabulary
    vocabulary = build_vocabulary_streaming(
        json_path=args.input,
        vocab_size=args.vocab_size,
        output_path=args.output
    )
    
    return vocabulary


if __name__ == "__main__":
    main()
