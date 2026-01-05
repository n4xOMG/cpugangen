"""Utilities for processing Danbooru tags"""

import re
from typing import List, Dict, Tuple
from collections import Counter


class DanbooruTagProcessor:
    """
    Process Danbooru tags for TOE training.
    
    Tag categories:
    - General: Visual attributes (1girl, blue_eyes, etc.)
    - Character: Character names
    - Copyright: Series/franchise
    - Artist: Creator
    - Meta: Image metadata (highres, etc.)
    """
    
    # Tag category prefixes used in some Danbooru datasets
    CATEGORY_PREFIXES = {
        0: "general",
        1: "artist",
        2: "unknown",
        3: "copyright",
        4: "character",
        5: "meta"
    }
    
    def __init__(self, vocab_size=15000):
        self.vocab_size = vocab_size
        self.tag_to_id = {}
        self.id_to_tag = {}
        self.tag_counts = Counter()
        
        # Special tokens
        self.PAD_TOKEN = "<pad>"
        self.UNK_TOKEN = "<unk>"
        self.EOS_TOKEN = "<eos>"
        
    def build_vocabulary(self, tag_lists: List[List[str]]):
        """
        Build vocabulary from tag lists.
        
        Args:
            tag_lists: List of tag lists from dataset
        """
        # Count all tags
        for tags in tag_lists:
            self.tag_counts.update(self._normalize_tags(tags))
            
        # Add special tokens
        self.tag_to_id[self.PAD_TOKEN] = 0
        self.tag_to_id[self.UNK_TOKEN] = 1
        self.tag_to_id[self.EOS_TOKEN] = 2
        
        # Add most common tags
        most_common = self.tag_counts.most_common(self.vocab_size - 3)
        for idx, (tag, count) in enumerate(most_common, start=3):
            self.tag_to_id[tag] = idx
            
        # Reverse mapping
        self.id_to_tag = {v: k for k, v in self.tag_to_id.items()}
        
        print(f"Built vocabulary: {len(self.tag_to_id)} tags")
        print(f"Most common tags: {most_common[:10]}")
        
    def _normalize_tags(self, tags: List[str]) -> List[str]:
        """Normalize tag formatting."""
        normalized = []
        for tag in tags:
            # Convert to lowercase
            tag = tag.lower().strip()
            # Replace spaces with underscores
            tag = tag.replace(" ", "_")
            # Remove special characters
            tag = re.sub(r'[^\w_]', '', tag)
            if tag:
                normalized.append(tag)
        return normalized
        
    def encode(self, tags: List[str], max_length=77) -> Tuple[List[int], List[float]]:
        """
        Encode tags to IDs and weights.
        
        Args:
            tags: List of tag strings
            max_length: Maximum sequence length
            
        Returns:
            tag_ids: List of vocabulary indices
            tag_weights: List of tag importance weights (1.0 for all by default)
        """
        tags = self._normalize_tags(tags)
        
        # Convert to IDs
        tag_ids = []
        for tag in tags[:max_length]:
            tag_id = self.tag_to_id.get(tag, self.tag_to_id[self.UNK_TOKEN])
            tag_ids.append(tag_id)
            
        # Uniform weights for now (can be customized later)
        tag_weights = [1.0] * len(tag_ids)
        
        # Pad if necessary
        while len(tag_ids) < max_length:
            tag_ids.append(self.tag_to_id[self.PAD_TOKEN])
            tag_weights.append(0.0)  # Zero weight for padding
            
        return tag_ids, tag_weights
    
    def decode(self, tag_ids: List[int]) -> List[str]:
        """Decode tag IDs back to strings."""
        tags = []
        for tag_id in tag_ids:
            tag = self.id_to_tag.get(tag_id, self.UNK_TOKEN)
            if tag not in [self.PAD_TOKEN, self.EOS_TOKEN]:
                tags.append(tag)
        return tags
    
    def save_vocabulary(self, filepath: str):
        """Save vocabulary to file."""
        import json
        with open(filepath, 'w') as f:
            json.dump({
                'tag_to_id': self.tag_to_id,
                'tag_counts': dict(self.tag_counts.most_common(self.vocab_size))
            }, f, indent=2)
        print(f"Saved vocabulary to {filepath}")
        
    def load_vocabulary(self, filepath: str):
        """Load vocabulary from file."""
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # Handle new vocabulary format with metadata
            if 'tag_to_id' in data:
                self.tag_to_id = data['tag_to_id']
            else:
                # Legacy format
                self.tag_to_id = data
            
            # Rebuild reverse mapping
            self.id_to_tag = {v: k for k, v in self.tag_to_id.items()}
            
            # Load tag counts if available
            if 'tag_counts' in data:
                self.tag_counts = Counter(data['tag_counts'])
            elif 'top_tags' in data:
                # Build counts from top_tags list
                self.tag_counts = Counter({
                    tag['name']: tag['post_count'] 
                    for tag in data['top_tags']
                })
            
        print(f"Loaded vocabulary: {len(self.tag_to_id)} tags")


if __name__ == "__main__":
    # Example usage
    processor = DanbooruTagProcessor(vocab_size=15000)
    
    # Sample tag lists
    sample_tags = [
        ["1girl", "solo", "long_hair", "blue_eyes", "school_uniform"],
        ["1girl", "smile", "outdoors", "cherry_blossoms"],
        ["multiple_girls", "fantasy", "dragon", "battle"],
    ]
    
    # Build vocabulary
    processor.build_vocabulary(sample_tags)
    
    # Encode tags
    test_tags = ["1girl", "solo", "long_hair", "blue_eyes"]
    tag_ids, tag_weights = processor.encode(test_tags)
    
    print(f"\nOriginal tags: {test_tags}")
    print(f"Encoded IDs: {tag_ids[:10]}...")
    print(f"Weights: {tag_weights[:10]}...")
    
    # Decode back
    decoded = processor.decode(tag_ids)
    print(f"Decoded tags: {decoded[:10]}")
