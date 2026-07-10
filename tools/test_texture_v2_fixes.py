"""Quick test to verify Texture V2 fixes work correctly."""
import sys
import cv2
import numpy as np
from pathlib import Path

# Add project path
sys.path.insert(0, str(Path.cwd() / "project"))

from project.s2_metrics.modules.texture.extractor_v2 import TextureExtractorV2

def create_test_context(image_path):
    """Create a test context from an image file."""
    class TestContext:
        def __init__(self, path):
            self.image_rgb = cv2.imread(str(path), cv2.IMREAD_COLOR)
            # Create a simple mock face mask
            if self.image_rgb is not None and self.image_rgb.size > 0:
                h, w = self.image_rgb.shape[:2]
                mask = np.ones((h, w), dtype=np.uint8)
                # Add some variation for testing
                mask = (mask * 255).astype(np.uint8)
                self.face_mask_path = None
                self._mask = mask
            else:
                self.face_mask_path = None
                self.image_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    return TestContext(image_path)

def test_fixes():
    print("=== Testing Texture V2 Fixes ===\n")
    
    # Use the two test photos
    test_photos = [
        Path("test_input/real/1999_08_16(2).jpg"),
        Path("test_input/silicone/2022_02_01.jpg")
    ]
    
    extractor = TextureExtractorV2()
    
    for photo_path in test_photos:
        print(f"Processing: {photo_path.name}")
        
        # Test with real context
        ctx = create_test_context(photo_path)
        
        # Extract metrics
        result = extractor.extract(ctx)
        
        # Display key results
        print(f"  Quality metrics:")
        print(f"    q_laplacian_var: {result.get('q_laplacian_var', 'N/A')}")
        print(f"    q_tenengrad: {result.get('q_tenengrad', 'N/A')}")
        print(f"    q_noise_sigma: {result.get('q_noise_sigma', 'N/A')}")
        print(f"    q_jpeg_blockiness: {result.get('q_jpeg_blockiness', 'N/A')}")
        
        print(f"  Texture metrics:")
        tier1_keys = [k for k in result.keys() if k not in ['texture_assessability', 'q_valid_patches', 'texture_unreliable']]
        for key in tier1_keys[:5]:
            print(f"    {key}: {result[key]:.4f}")
        
        print(f"  Quality assessment:")
        print(f"    texture_assessability: {result.get('texture_assessability', 'N/A')}")
        print(f"    q_valid_patches: {result.get('q_valid_patches', 'N/A')}")
        print(f"    texture_unreliable: {result.get('texture_unreliable', 'N/A')}")
        
        print(f"  Texture skin hint: {result.get('texture_skin_hint', 'No model available')}")
        print()

if __name__ == "__main__":
    test_fixes()
