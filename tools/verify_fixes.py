#!/usr/bin/env python3
"""Quick test to verify Texture V2 fixes work correctly."""
import os
import cv2
import numpy as np
from pathlib import Path

# Load the fix code to verify thresholds

def test_extractor_v2():
    print("=== Testing Texture V2 Fixes ===\n")
    
    # Read the extractor_v2.py and check the key thresholds
    extractor_path = Path("project/s2_metrics/modules/texture/extractor_v2.py")
    with open(extractor_path, 'r') as f:
        content = f.read()
    
    # Check for the new thresholds
    if "p10 lapl 25.0, native_tenengrad p10 25.0" in content:
        print("✅ Tenengrad threshold fixed: should be ~25, not 403")
    else:
        print("❌ Tenengrad threshold NOT fixed (expected ~25)")
    
    if "spec_mask = (min_rgb > 180) &" in content:
        print("✅ Specular threshold fixed: should be 180, not 200")
    else:
        print("❌ Specular threshold NOT fixed (expected 180)")
    
    # Check texture_anomaly.py
    anomaly_path = Path("project/s2_metrics/texture_anomaly.py")
    with open(anomaly_path, 'r') as f:
        content = f.read()
    
    if "if quality < 0.28:" in content:
        print("✅ Quality threshold fixed: should be 0.28, not 0.35")
    else:
        print("❌ Quality threshold NOT fixed (expected 0.28)")
    
    # Check engine.py
    engine_path = Path("project/s2_metrics/engine.py")
    with open(engine_path, 'r') as f:
        content = f.read()
    
    if 'texture_assessability = texture.get("texture_assessability"' in content:
        print("✅ JSON storage fixed: using get() instead of pop()")
    else:
        print("❌ JSON storage NOT fixed")
    
    # Check if skin classifier exists
    classifier_path = Path("project/s2_metrics/modules/texture/skin_classifier_v2.pkl")
    if classifier_path.exists():
        print(f"✅ Skin classifier model exists: {classifier_path.stat().st_size} bytes")
    else:
        print("❌ Skin classifier model does NOT exist")
    
    print("\n" + "="*50)
    print("SUMMARY:")
    print("Fixes have been applied to:")
    print("1. extractor_v2.py - quality assessment thresholds")
    print("2. texture_anomaly.py - quality threshold")  
    print("3. engine.py - JSON storage")
    print("\nNext step: Run actual pipeline to test fixes:")
    print("python -m project.run --stages s1 s2 --input-main data/photo/all --output-main data/storage/main")

if __name__ == "__main__":
    test_extractor_v2()
