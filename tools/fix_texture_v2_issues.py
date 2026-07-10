"""Script to fix Texture V2 quality assessment thresholds and bugs."""
import re

def fix_extractor_v2_quality_thresholds(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix 1: Quality assessment thresholds
    # Replace the old thresholds (20.5, 403, 3.66, 0.01156) with new ones
    # Find the quality assessment section and fix the values
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if "p10 lapl 20.5, native_tenengrad p10 ~25 for letterboxed" in line:
            lines[i] = "            # p10 lapl 25.0, native_tenengrad p10 25.0, p90 noise 3.0, p90 block 1.9 for letterboxed 424x500"
        elif "elif (lapl < 20.5 or native_tenengrad < 25) or (noise > 3.66 or block > 0.01156):" in line:
            lines[i] = "            elif (lapl < 25.0 or native_tenengrad < 25.0) or (noise > 3.0 or block > 1.9):"
        elif "elif (lapl < 20.5 and native_tenengrad < 25) or (noise > 3.66 and block > 0.01156):" in line:
            lines[i] = "            elif (lapl < 25.0 and native_tenengrad < 25.0) or (noise > 3.0 and block > 1.9):"
    
    # Fix 2: specular elongation
    for i, line in enumerate(lines):
        if "min_rgb > 200) & (sat < 30) & (mask > 0):" in line:
            lines[i] = "            spec_mask = (min_rgb > 180) & (sat < 40) & (mask > 0)"
        elif "return 1.0  # No specular = isotropic" in line:
            lines[i] = "            return 1.0  # No specular = isotropic"
        elif "if elongations:" in line:
            lines[i] = "            return float(np.mean(elongations)) if elongations else 1.0"
    
    content = '\n'.join(lines)
    
    # Fix 3: dwt_haar_HH_LL_ratio
    # Add check for empty ll_masked and fallback to full arrays
    pattern = r'(def _tier2_dwt_haar_HH_LL_ratio.*?return 0\.0\s*\n)'
    def fix_dwt(match):
        return match.group(1).replace(
            "            if ll_masked.size == 0 or hh_masked.size == 0:\n                return 0.0",
            "            if ll_masked.size == 0 or hh_masked.size == 0:\n                ll_masked = ll.ravel()\n                hh_masked = hh.ravel()\n                if ll_masked.size == 0 or hh_masked.size == 0:\n                    return 0.0"
        )
    
    content = re.sub(pattern, fix_dwt, content, flags=re.DOTALL)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Fixed extractor_v2.py quality assessment thresholds")

def fix_texture_anomaly_quality_threshold(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix: quality < 0.35 -> quality < 0.28
    content = content.replace(
        '        if quality < 0.35:',
        '        if quality < 0.28:'
    )
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Fixed texture_anomaly.py quality threshold")

def fix_engine_json_issues(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Remove pop operations to keep data in JSON
    content = content.replace(
        '        texture_assessability = texture.pop("texture_assessability", "eligible")',
        '        texture_assessability = texture.get("texture_assessability", "eligible")'
    )
    content = content.replace(
        '        q_valid_patches = texture.pop("q_valid_patches", 0)',
        '        q_valid_patches = texture.get("q_valid_patches", 0)'
    )
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Fixed engine.py JSON issues")

if __name__ == "__main__":
    base_path = "/Users/victorkhudyakov/deeputin/project"
    
    # Fix extractor_v2.py
    extractor_path = f"{base_path}/s2_metrics/modules/texture/extractor_v2.py"
    fix_extractor_v2_quality_thresholds(extractor_path)
    
    # Fix texture_anomaly.py
    anomaly_path = f"{base_path}/s2_metrics/texture_anomaly.py"
    fix_texture_anomaly_quality_threshold(anomaly_path)
    
    # Fix engine.py
    engine_path = f"{base_path}/s2_metrics/engine.py"
    fix_engine_json_issues(engine_path)
    
    print("\nAll fixes completed successfully!")
    print("Now you can run:")
    print("  python tools/train_skin_classifier_v2.py")
    print("  rm -rf data/storage/main/*/texture_metrics.json data/storage/main/stage*manifest.json")
    print("  python -m project.run --stages s1 s2 --input-main data/photo/all --output-main data/storage/main")
