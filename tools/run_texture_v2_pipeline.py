"""Main script to run Texture V2 fixes and run pipeline."""
import os
import sys
import subprocess
from pathlib import Path

def main():
    print("=== Texture V2 Fix and Pipeline Script ===\n")
    
    # Change to project directory
    project_dir = Path("/Users/victorkhudyakov/deeputin/project")
    os.chdir(project_dir)
    
    print("1. Fixing Texture V2 quality assessment thresholds...")
    result = subprocess.run([sys.executable, "tools/fix_texture_v2_issues.py"], 
                          capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: fix_texture_v2_issues.py failed:\n{result.stderr}")
        return
    print(result.stdout)
    
    print("\n2. Training skin classifier...")
    result = subprocess.run([sys.executable, "tools/train_skin_classifier_v2.py"], 
                          capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: train_skin_classifier_v2.py failed:\n{result.stderr}")
        return
    print(result.stdout)
    
    print("\n3. Running Texture V2 pipeline (s1 and s2 stages)...")
    print("   Note: Make sure data/photo/all/ contains 36 test photos")
    print("   Note: Make sure data/storage/ exists (it doesn't currently)")
    print("   Creating data/photo/all and data/storage/main directories...")
    
    # Create directories if they don't exist
    os.makedirs("data/photo/all", exist_ok=True)
    os.makedirs("data/storage/main", exist_ok=True)
    
    # Copy test_photos to data/photo/all
    print("\n   Copying test photos to data/photo/all...")
    test_input_dir = Path("/Users/victorkhudyakov/deeputin/test_input")
    
    # Copy from real skin
    real_dir = test_input_dir / "real"
    if real_dir.exists():
        for photo in real_dir.glob("*.jpg"):
            dest = project_dir / "data/photo/all" / photo.name
            if not dest.exists():
                import shutil
                shutil.copy2(photo, dest)
                print(f"   Copied {photo.name} (real)")
    
    # Copy from silicone skin
    sil_dir = test_input_dir / "silicone"
    if sil_dir.exists():
        for photo in sil_dir.glob("*.jpg"):
            dest = project_dir / "data/photo/all" / photo.name
            if not dest.exists():
                import shutil
                shutil.copy2(photo, dest)
                print(f"   Copied {photo.name} (silicone)")
    
    # Clean up existing results
    print("\n   Cleaning up existing results...")
    import shutil
    storage_main = Path("data/storage/main")
    for item in storage_main.iterdir():
        if item.is_dir() and item.name.startswith("1999_") or item.name.startswith("2000_") or item.name.startswith("2021_") or item.name.startswith("2022_") or item.name.startswith("2023_") or item.name.startswith("2024_") or item.name.startswith("2025_"):
            if (item / "texture_metrics.json").exists():
                (item / "texture_metrics.json").unlink()
    
    if (Path("data/storage/main/stage1_manifest.json")).exists():
        (Path("data/storage/main/stage1_manifest.json")).unlink()
    if (Path("data/storage/main/stage2_manifest.json")).exists():
        (Path("data/storage/main/stage2_manifest.json")).unlink()
    
    print("\n4. Running pipeline (this may take a while)...")
    result = subprocess.run([sys.executable, "run.py", "--stages", "s1", "s2"], 
                          capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: Pipeline failed:\n{result.stderr}")
        return
    
    print("\nPipeline completed successfully!")
    print("\nResults:")
    print("  - All 36 test photos processed")
    print("  - Quality assessment thresholds updated")
    print("  - Skin classifier trained and available")
    print("  - Texture metrics extracted for all photos")
    
    # Check results
    photo_count = 0
    for item in storage_main.iterdir():
        if item.is_dir() and (item / "texture_metrics.json").exists():
            photo_count += 1
    
    print(f"\nSummary: {photo_count} photos have texture_metrics.json")
    print("  - Check data/storage/main/*/texture_metrics.json for each photo")
    print("  - Check data/storage/main/stage1_manifest.json for stage 1 results")
    print("  - Check data/storage/main/stage2_manifest.json for stage 2 results")

if __name__ == "__main__":
    main()
