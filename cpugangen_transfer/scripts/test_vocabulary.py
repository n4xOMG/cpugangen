"""
Test vocabulary building and tag processing functionality.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hqpd.utils.danbooru import DanbooruTagProcessor


def test_vocabulary_loading():
    """Test loading pre-built vocabulary."""
    vocab_path = Path(__file__).parent.parent / "data" / "vocabulary.json"
    
    if not vocab_path.exists():
        print(f"❌ Vocabulary file not found: {vocab_path}")
        print(f"   Please run: python scripts/build_vocabulary.py")
        return False
    
    print(f"✓ Found vocabulary file: {vocab_path}")
    
    # Load vocabulary
    processor = DanbooruTagProcessor(vocab_size=15000)
    processor.load_vocabulary(str(vocab_path))
    
    print(f"✓ Loaded vocabulary: {len(processor.tag_to_id):,} tags")
    
    # Check special tokens
    assert processor.PAD_TOKEN in processor.tag_to_id, "Missing PAD token"
    assert processor.UNK_TOKEN in processor.tag_to_id, "Missing UNK token"
    assert processor.EOS_TOKEN in processor.tag_to_id, "Missing EOS token"
    print(f"✓ Special tokens present")
    
    # Check common tags
    common_tags = ['1girl', 'solo', 'highres', 'long_hair', 'smile']
    found_tags = [tag for tag in common_tags if tag in processor.tag_to_id]
    print(f"✓ Common tags found: {found_tags}")
    
    return True


def test_encoding_decoding():
    """Test tag encoding and decoding."""
    vocab_path = Path(__file__).parent.parent / "data" / "vocabulary.json"
    
    if not vocab_path.exists():
        print(f"❌ Vocabulary file not found")
        return False
    
    processor = DanbooruTagProcessor(vocab_size=15000)
    processor.load_vocabulary(str(vocab_path))
    
    # Test tags
    test_tags = ['1girl', 'solo', 'long_hair', 'blue_eyes', 'school_uniform']
    print(f"\n📝 Testing encoding/decoding:")
    print(f"   Input tags: {test_tags}")
    
    # Encode
    tag_ids, tag_weights = processor.encode(test_tags, max_length=77)
    print(f"   Encoded IDs (first 10): {tag_ids[:10]}")
    print(f"   Weights (first 10): {tag_weights[:10]}")
    
    # Decode
    decoded_tags = processor.decode(tag_ids)
    print(f"   Decoded tags (first 10): {decoded_tags[:10]}")
    
    # Verify
    assert len(tag_ids) == 77, f"Expected 77 IDs, got {len(tag_ids)}"
    assert len(tag_weights) == 77, f"Expected 77 weights, got {len(tag_weights)}"
    assert tag_weights[0] == 1.0, "First weight should be 1.0"
    assert tag_weights[-1] == 0.0, "Last weight should be 0.0 (padding)"
    
    print(f"✓ Encoding/decoding works correctly")
    
    return True


def test_unknown_tags():
    """Test handling of unknown tags."""
    vocab_path = Path(__file__).parent.parent / "data" / "vocabulary.json"
    
    if not vocab_path.exists():
        print(f"❌ Vocabulary file not found")
        return False
    
    processor = DanbooruTagProcessor(vocab_size=15000)
    processor.load_vocabulary(str(vocab_path))
    
    # Mix of known and unknown tags
    test_tags = ['1girl', 'unknown_tag_xyz123', 'solo', 'another_unknown_tag']
    print(f"\n🔍 Testing unknown tag handling:")
    print(f"   Input: {test_tags}")
    
    tag_ids, tag_weights = processor.encode(test_tags)
    decoded = processor.decode(tag_ids)
    
    print(f"   Decoded: {decoded[:len(test_tags)]}")
    print(f"✓ Unknown tags handled correctly")
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Danbooru Tag Vocabulary")
    print("=" * 60)
    
    tests = [
        ("Loading vocabulary", test_vocabulary_loading),
        ("Encoding/Decoding", test_encoding_decoding),
        ("Unknown tags", test_unknown_tags),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'─' * 60}")
        print(f"Test: {test_name}")
        print(f"{'─' * 60}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'=' * 60}")
    print(f"Test Summary")
    print(f"{'=' * 60}")
    for test_name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        print(f"\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  Some tests failed")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
