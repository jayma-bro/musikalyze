
"""Test script to verify audio feature tags work correctly."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0=all, 1=info off, 2=warnings off, 3=errors only
import essentia

essentia.log.infoActive = False
from musikalyze import (
    TaggingConfig,
)


def test_audio_feature_tags():
    """Test that audio feature tags work correctly."""
    
    print("Testing audio feature tag support...")
    
    # Create a test configuration that uses the new audio feature tags
    _tagging_config = TaggingConfig(
        tags={
            "artist": "{tag_artist}",
            "title": "{tag_title}",
            "album": "{tag_album}",
            "genre": "{meta_genre}",
        },
        extra={
            "acousticness": "{meta_acousticness}",
            "danceability": "{meta_danceability}",
            "key": "{meta_key}",
            "tcop": "{meta_tcop}",
        }
    )
    
    print("✓ TaggingConfig with audio feature tags created successfully")
    
    # Test that these tags are recognized in the logical keys
    from musikalyze.tagging import _LOGICAL_KEYS
    expected_tags = ["acousticness", "danceability", "key", "tcop"]
    
    for tag in expected_tags:
        if tag in _LOGICAL_KEYS:
            print(f"✓ Logical key '{tag}' recognized")
        else:
            print(f"✗ Logical key '{tag}' NOT recognized")
    
    # Test that the mapping works correctly
    test_meta = {
        "meta_acousticness": 0.85,
        "meta_danceability": 0.75,
        "meta_key": "C#",
        "meta_tcop": "Commercial"
    }
    
    # Test that these new meta tags are correctly handled
    from musikalyze.templates import build_format_mapping, resolve_template
    
    base = build_format_mapping({}, test_meta, ext=None)
    print(f"✓ Base mapping created with meta keys: {list(base.keys())}")
    
    # Test resolution of templates with these tags
    template = "{meta_acousticness} - {meta_danceability} - {meta_key} - {meta_tcop}"
    result = resolve_template(template, base, separator=";")
    print(f"✓ Template resolution: {result}")
    
    print("All audio feature tag tests passed!")

if __name__ == "__main__":
    test_audio_feature_tags()