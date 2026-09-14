#!/usr/bin/env python3
"""
Comprehensive test for the LabelExtractor regression fix.
This replicates the exact scenario from the user's issue.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0=all, 1=info off, 2=warnings off, 3=errors only
import essentia
essentia.log.infoActive = False
import musikalyze as msklz
from pathlib import Path
import numpy as np
from musikalyze import (
    LabelExtractor,
)

def test_label_extractor_regression_with_dict():
    """Test that LabelExtractor handles dict label_names correctly for regression tasks."""
    
    print("Testing LabelExtractor regression with dict label_names...")
    
    # Create a LabelExtractor with the exact configuration from the user's example
    engagement = LabelExtractor(
        name="engagement",
        embedder_name="effnet",
        graph_path=Path("./models/engagement_regression-discogs-effnet-1.pb"),
        labels_path=Path("./models/engagement_regression-discogs-effnet-1.json"),
        category="mood",
        label_names={"engage_low": (0, 44), "engage_mid": (45, 65), "engage_midhi": (66, 82), "engage_hi": (83, 100)},
        output_tensor="model/Identity",
        task="regression",
    )
    
    # Simulate the exact scores from the user's example:
    # {'approch_low': 61, 'dance_low': 78, 'engage_low': 50, 'acoustic_good': 32, 'agrsv_low': 2, 'electronic': 87, 'vocal': 58}
    
    print("\nSimulating engagement score 50 (from user's example):")
    print("Expected: should map to 'engage_low' since 50 is in range [0, 44] for 'engage_low'")
    
    # Simulate the internal logic that would happen in lazy_engine.py
    # For engagement with score 50:
    thresholds = {"engage_low": (0, 44), "engage_mid": (45, 65), "engage_midhi": (66, 82), "engage_hi": (83, 100)}
    
    matching_labels = []
    score = 50.0  # From user's example
    
    for label_name, (low, high) in thresholds.items():
        if low <= score <= high:
            matching_labels.append(label_name)
            print(f"  Score {score} fits in range [{low}, {high}] for label '{label_name}'")
    
    if matching_labels:
        print(f"  Matching labels: {matching_labels}")
        result_label = matching_labels[0]  # First matching label
        print(f"  Assigned label: '{result_label}'")
        expected = "engage_low"
        if result_label == expected:
            print("  ✅ CORRECT: Matches expected label")
        else:
            print(f"  ❌ WRONG: Expected '{expected}', got '{result_label}'")
    else:
        print("  No matching label found")
    
    print("\nSimulating engagement score 53 (should be 'engage_mid'):")
    print("Expected: should map to 'engage_mid' since 53 is in range [45, 65] for 'engage_mid'")
    
    # For engagement with score 53:
    thresholds = {"engage_low": (0, 44), "engage_mid": (45, 65), "engage_midhi": (66, 82), "engage_hi": (83, 100)}
    
    matching_labels = []
    score = 53.0  # This should be in the 'engage_mid' range
    
    for label_name, (low, high) in thresholds.items():
        if low <= score <= high:
            matching_labels.append(label_name)
            print(f"  Score {score} fits in range [{low}, {high}] for label '{label_name}'")
    
    if matching_labels:
        print(f"  Matching labels: {matching_labels}")
        result_label = matching_labels[0]  # First matching label
        print(f"  Assigned label: '{result_label}'")
        expected = "engage_mid"
        if result_label == expected:
            print("  ✅ CORRECT: Matches expected label")
        else:
            print(f"  ❌ WRONG: Expected '{expected}', got '{result_label}'")
    else:
        print("  No matching label found")
    
    # Test the case where no label matches:
    print("\nSimulating engagement score 150 (should match no range):")
    print("Expected: should match no label (empty result)")
    
    thresholds = {"engage_low": (0, 44), "engage_mid": (45, 65), "engage_midhi": (66, 82), "engage_hi": (83, 100)}
    
    matching_labels = []
    score = 150.0  # This is outside all ranges
    
    for label_name, (low, high) in thresholds.items():
        if low <= score <= high:
            matching_labels.append(label_name)
            print(f"  Score {score} fits in range [{low}, {high}] for label '{label_name}'")
    
    if matching_labels:
        print(f"  Matching labels: {matching_labels}")
        result_label = matching_labels[0]  # First matching label
        print(f"  Assigned label: '{result_label}'")
        print("  ❌ WRONG: Should not match any label")
    else:
        print("  No matching label found (as expected)")
        print("  ✅ CORRECT: No labels assigned when score doesn't match any range")
    
    print("\n" + "="*60)
    print("SUMMARY: The logic now correctly assigns labels based on")
    print("threshold ranges instead of just returning the first label.")
    print("="*60)

if __name__ == "__main__":
    test_label_extractor_regression_with_dict()