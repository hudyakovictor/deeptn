#!/usr/bin/env python3
"""
eyetest.py — Silicon Mask Detection Test Suite
=============================================
Run from 3DDFA-V3 directory. Uses 3DDFA-V3 + MediaPipe to detect
silicone masks with focus on eye/eyelid analysis.

Usage:
    python eyetest.py --inputpath /path/to/real_photos --output eyetest_real.json
    python eyetest.py --inputpath /path/to/mask_photos --output eyetest_mask.json

Then compare the two JSONs to see which methods best separate real vs mask.
"""

import argparse
import cv2
import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ==================== 3DDFA-V3 imports ====================
from face_box import face_box
from model.recon import face_model
from util.preprocess import get_data_path
from util.io import back_resize_ldms

# ==================== MediaPipe ====================
import mediapipe as mp

# ==================== SciPy ====================
from scipy.spatial import cKDTree

# ==================== Head Pose Estimation ====================
_HPE_PATH = "/Users/victorkhudyakov/dutin/core/head-pose-estimation"

if _HPE_PATH not in sys.path:
    sys.path.insert(0, _HPE_PATH)

try:
    from models import SCRFD as _SCRFD, get_model as _get_model
    from utils.general import compute_euler_angles_from_rotation_matrices as _compute_euler
    _HPE_AVAILABLE = True
except ImportError:
    _HPE_AVAILABLE = False
    _SCRFD = None
    _get_model = None
    _compute_euler = None


# =============================================================================
# CONSTANTS
# =============================================================================

# 3DDFA-V3 landmark indices (68-point model)
# Eyes: 36-41 (left), 42-47 (right)
# Brows: 17-21 (left), 22-26 (right)
# Nose: 27-35
# Mouth: 48-67
# Jaw: 0-16

# MediaPipe eye outlines (468 base + 10 iris)
MP_LEFT_EYE_OUTLINE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
MP_RIGHT_EYE_OUTLINE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
MP_LEFT_IRIS = [468, 469, 470, 471, 472]
MP_RIGHT_IRIS = [473, 474, 475, 476, 477]
MP_LEFT_EYE_CORNERS = [33, 133]
MP_RIGHT_EYE_CORNERS = [362, 263]
MP_LEFT_LIDS = [159, 145]  # upper, lower
MP_RIGHT_LIDS = [386, 374]
MP_LEFT_BROW = [70, 63, 105, 66, 107]
MP_RIGHT_BROW = [336, 296, 334, 293, 300]
MP_FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# Segmentation parts (3DDFA-V3): [right_eye, left_eye, right_eyebrow, left_eyebrow, nose, up_lip, down_lip, skin]
SEG_PARTS = ['right_eye', 'left_eye', 'right_eyebrow', 'left_eyebrow', 'nose', 'up_lip', 'down_lip', 'skin']


# =============================================================================
# HELPER: safe float conversion
# =============================================================================
def safe_float(val):
    """Convert to JSON-safe float."""
    if val is None:
        return 0.0
    v = float(val)
    if np.isnan(v) or np.isinf(v):
        return 0.0
    return round(v, 6)


def safe_dict(d):
    """Make all values in dict JSON-safe."""
    return {k: safe_float(v) for k, v in d.items()}


# =============================================================================
# MEDIAPIPE WRAPPER
# =============================================================================
class MediaPipeAnalyzer:
    """Wrapper for MediaPipe Face Mesh + Face Detection."""

    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,  # enables iris tracking
            min_detection_confidence=0.3,
        )
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=0.3,
        )

    def analyze(self, image_rgb):
        """Run MediaPipe on RGB image, return landmarks + detection."""
        h, w = image_rgb.shape[:2]
        result = {
            'landmarks': None,
            'detection_bbox': None,
            'iris_available': False,
        }

        # Face Mesh
        mp_mesh = self.face_mesh.process(image_rgb)
        if mp_mesh.multi_face_landmarks:
            lm = mp_mesh.multi_face_landmarks[0]
            pts = np.zeros((len(lm.landmark), 3))
            for i, lmk in enumerate(lm.landmark):
                pts[i] = [lmk.x * w, lmk.y * h, lmk.z]
            result['landmarks'] = pts
            result['iris_available'] = len(lm.landmark) >= 478

        # Face Detection
        mp_det = self.face_detection.process(image_rgb)
        if mp_det.detections:
            det = mp_det.detections[0]
            bb = det.location_data.relative_bounding_box
            result['detection_bbox'] = [bb.xmin * w, bb.ymin * h, bb.width * w, bb.height * h]

        return result


# =============================================================================
# 3DDFA-V3 WRAPPER
# =============================================================================
class ThreeDDFAAnalyzer:
    """Wrapper for 3DDFA-V3 reconstruction."""

    def __init__(self, args):
        self.args = args
        self.recon_model = face_model(args)
        self.facebox_detector = face_box(args).detector

    def analyze(self, image_pil):
        """Run 3DDFA-V3 on PIL image, return full reconstruction dict."""
        trans_params, im_tensor = self.facebox_detector(image_pil)
        self.recon_model.input_img = im_tensor.to(self.args.device)
        results = self.recon_model.forward()

        # Extract batch=0
        out = {}
        for key, val in results.items():
            if isinstance(val, np.ndarray):
                if val.ndim == 4:
                    out[key] = val[0]
                elif val.ndim == 3 and val.shape[0] == 1:
                    out[key] = val[0]
                else:
                    out[key] = val
            else:
                out[key] = val

        out['trans_params'] = trans_params
        return out


# =============================================================================
# HEAD POSE ESTIMATION + РАКУРС CLASSIFICATION
# =============================================================================

class HeadPoseEstimator:
    """Singleton head pose estimator using SCRFD + MobileNetV3."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.device = torch.device("cpu")
        self.face_detector = None
        self.head_pose = None

        if not _HPE_AVAILABLE:
            print("  WARNING: head-pose-estimation not available, pose will use 3DDFA fallback")
            return

        try:
            det_path = os.path.join(_HPE_PATH, "weights", "det_10g.onnx")
            self.face_detector = _SCRFD(model_path=det_path)
        except Exception as e:
            print(f"  WARNING: Failed to load SCRFD face detector: {e}")

        try:
            weights_path = os.path.join(_HPE_PATH, "weights", "mobilenetv3_large.pt")
            self.head_pose = _get_model("mobilenetv3_large", num_classes=6, pretrained=False)
            state_dict = torch.load(weights_path, map_location=self.device)
            self.head_pose.load_state_dict(state_dict)
            self.head_pose.to(self.device)
            self.head_pose.eval()
        except Exception as e:
            print(f"  WARNING: Failed to load head pose model: {e}")

    def predict(self, image_bgr):
        """Returns {'yaw': deg, 'pitch': deg, 'roll': deg} or None."""
        if self.face_detector is None or self.head_pose is None:
            return None

        import torchvision.transforms as transforms

        frame = image_bgr
        if frame is None:
            return None

        with torch.no_grad():
            bboxes, keypoints = self.face_detector.detect(frame)
            if len(bboxes) == 0:
                return None

            if len(bboxes) > 1:
                areas = [(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in bboxes]
                bbox = bboxes[np.argmax(areas)]
            else:
                bbox = bboxes[0]

            x_min, y_min, x_max, y_max = map(int, bbox[:4])
            h, w = frame.shape[:2]
            x_min = max(0, x_min - int(0.2 * (y_max - y_min)))
            y_min = max(0, y_min - int(0.2 * (x_max - x_min)))
            x_max = min(w, x_max + int(0.2 * (y_max - y_min)))
            y_max = min(h, y_max + int(0.2 * (x_max - x_min)))

            if x_max <= x_min or y_max <= y_min:
                return None

            image_crop = frame[y_min:y_max, x_min:x_max]
            image_crop_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)

            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image_tensor = transform(image_crop_rgb).unsqueeze(0).to(self.device)

            rotation_matrix = self.head_pose(image_tensor)
            euler = np.degrees(_compute_euler(rotation_matrix))

            return {
                "yaw": float(euler[:, 1].cpu()[0]),
                "pitch": float(euler[:, 0].cpu()[0]),
                "roll": float(euler[:, 2].cpu()[0]),
            }


# Ракурс classification — 7 angles (без профилей)
# Пороги из core/utils.py: bucket_from_visible_side_and_yaw_abs()
def classify_rakurs(yaw_deg):
    """
    Classify viewing angle into one of 7 rakurs (no profile).
    Returns (rakurs_id, side) where side is 'L'/'R'/'C'.
    """
    ya = abs(yaw_deg)
    if ya <= 6.0:
        return "frontal", "C"
    elif ya <= 25.0:
        side = "L" if yaw_deg < 0 else "R"
        return "light", side
    elif ya <= 45.0:
        side = "L" if yaw_deg < 0 else "R"
        return "mid", side
    elif ya <= 65.0:
        side = "L" if yaw_deg < 0 else "R"
        return "deep", side
    else:
        return None, None  # profile — skip


def detect_visible_eye(mp_lm):
    """
    Detect which eye is visible based on MediaPipe landmarks.
    Returns 'both', 'left', or 'right'.
    """
    if mp_lm is None:
        return "both"

    has_left = all(i < len(mp_lm) for i in [33, 133, 159, 145])
    has_right = all(i < len(mp_lm) for i in [362, 263, 386, 374])

    if has_left and has_right:
        return "both"
    elif has_left:
        return "left"
    elif has_right:
        return "right"
    else:
        return "none"


# =============================================================================
# EYE ANALYSIS METHODS
# =============================================================================

class EyeMaskDetector:
    """
    Core eye-focused mask detection methods.
    8 methods focused on eyelid, aperture, iris, periocular analysis.
    """

    def __init__(self):
        pass

    def run_all(self, image_rgb, ddffa, mp_data):
        """Run all eye methods, return dict of scores."""
        scores = {}
        details = {}

        # Get data
        mp_lm = mp_data.get('landmarks')
        ldm68 = ddffa.get('ldm68')
        ldm106 = ddffa.get('ldm106')
        seg_vis = ddffa.get('seg_visible')
        v3d = ddffa.get('v3d')
        v2d = ddffa.get('v2d')
        trans_params = ddffa.get('trans_params')

        h, w = image_rgb.shape[:2]

        # Back-project 3DDFA landmarks to original image space
        ldm68_orig = None
        if ldm68 is not None and trans_params is not None:
            ldm68_copy = ldm68.copy()
            ldm68_copy[:, 1] = 224 - 1 - ldm68_copy[:, 1]  # flip Y
            ldm68_orig = back_resize_ldms(ldm68_copy, trans_params)

        ldm106_orig = None
        if ldm106 is not None and trans_params is not None:
            ldm106_copy = ldm106.copy()
            ldm106_copy[:, 1] = 224 - 1 - ldm106_copy[:, 1]
            ldm106_orig = back_resize_ldms(ldm106_copy, trans_params)

        ldm134_orig = None
        ldm134 = ddffa.get('ldm134')
        if ldm134 is not None and trans_params is not None:
            ldm134_copy = ldm134.copy()
            ldm134_copy[:, 1] = 224 - 1 - ldm134_copy[:, 1]
            ldm134_orig = back_resize_ldms(ldm134_copy, trans_params)

        # === METHOD 1: Aperture Shrinkage ===
        scores['aperture_shrinkage'], details['aperture'] = self._aperture_shrinkage(
            mp_lm, ldm68_orig, ldm106_orig, h, w)

        # === METHOD 2: Eyelid Thickness (INVERTED: real>silicone → 1-score) ===
        raw_thick, details['lid_thickness'] = self._eyelid_thickness(
            image_rgb, mp_lm, ldm68_orig, h, w)
        scores['eyelid_thickness'] = safe_float(1.0 - raw_thick)

        # === METHOD 3: Lid Smoothness ===
        scores['lid_smoothness'], details['lid_smooth'] = self._lid_smoothness(
            image_rgb, mp_lm, ldm68_orig, h, w)

        # === METHOD 4: Eyelid Edge Sharpness ===
        scores['eyelid_edge_sharpness'], details['lid_edge'] = self._eyelid_edge_sharpness(
            image_rgb, mp_lm, ldm68_orig, h, w)

        # === METHOD 5: Eye Symmetry (INVERTED: real>silicone → 1-score) ===
        raw_sym, details['eye_sym'] = self._eye_symmetry(
            mp_lm, ldm68_orig, h, w)
        scores['eye_symmetry_anomaly'] = safe_float(1.0 - raw_sym)

        # === METHOD 6: Sclera-Iris Boundary ===
        scores['sclera_iris_boundary'], details['sclera_iris'] = self._sclera_iris_boundary(
            image_rgb, mp_lm, h, w)

        # === METHOD 7: Pupil Apparent Size ===
        scores['pupil_apparent_size'], details['pupil_size'] = self._pupil_apparent_size(
            image_rgb, mp_lm, h, w)

        # === METHOD 8: Eye Orbit Area (segmentation) ===
        scores['orbit_area_ratio'], details['orbit_area'] = self._orbit_area_ratio(
            seg_vis, mp_lm, ddffa, h, w, trans_params)

        # === METHOD 9: Periocular Texture LBP ===
        scores['periocular_lbp_entropy'], details['peri_lbp'] = self._periocular_lbp(
            image_rgb, mp_lm, ldm68_orig, h, w)

        # === METHOD 10: Subsurface Scattering (INVERTED: real>silicone → 1-score) ===
        raw_sss, details['sss'] = self._eyelid_sss(
            image_rgb, mp_lm, ldm68_orig, h, w)
        scores['eyelid_sss'] = safe_float(1.0 - raw_sss)

        # === METHOD 11: Iris Visible Area Ratio (ldm134 eyelid + MP iris) ===
        scores['iris_visible_ratio'], details['iris_vis'] = self._iris_visible_ratio(
            mp_lm, ldm134_orig, h, w)

        # === METHOD 12: Iris Center Offset (iris position in eye opening) ===
        scores['iris_center_offset'], details['iris_offset'] = self._iris_center_offset(
            mp_lm, ldm134_orig, h, w)

        # === METHOD 13: Sclera Asymmetry (visible white area L vs R) ===
        scores['sclera_asymmetry'], details['sclera_asym'] = self._sclera_asymmetry(
            image_rgb, mp_lm, h, w)

        return safe_dict(scores), details

    # --- Individual methods ---

    def _get_iod(self, mp_lm, ldm68_orig):
        """Inter-ocular distance for normalization."""
        if mp_lm is not None and len(mp_lm) > 263:
            left = mp_lm[33, :2]
            right = mp_lm[263, :2]
            return np.linalg.norm(right - left)
        if ldm68_orig is not None and len(ldm68_orig) > 45:
            left = ldm68_orig[36:42].mean(axis=0)
            right = ldm68_orig[42:48].mean(axis=0)
            return np.linalg.norm(right[:2] - left[:2])
        return 100.0  # fallback

    def _get_eye_crop(self, image_rgb, mp_lm, ldm68_orig, side, expand=1.5):
        """Get eye region crop."""
        h, w = image_rgb.shape[:2]
        iod = self._get_iod(mp_lm, ldm68_orig)
        half = int(iod * 0.5 * expand)

        if mp_lm is not None:
            if side == 'left':
                indices = MP_LEFT_EYE_OUTLINE
            else:
                indices = MP_RIGHT_EYE_OUTLINE
            valid_idx = [i for i in indices if i < len(mp_lm)]
            if not valid_idx:
                return None, None
            pts = mp_lm[valid_idx, :2]
        elif ldm68_orig is not None:
            if side == 'left':
                pts = ldm68_orig[36:42, :2]
            else:
                pts = ldm68_orig[42:48, :2]
        else:
            return None, None

        center = pts.mean(axis=0).astype(int)
        y1 = max(0, center[1] - half)
        y2 = min(h, center[1] + half)
        x1 = max(0, center[0] - half)
        x2 = min(w, center[0] + half)

        if y2 <= y1 or x2 <= x1:
            return None, None

        crop = image_rgb[y1:y2, x1:x2]
        return crop, (x1, y1, x2, y2)

    def _aperture_shrinkage(self, mp_lm, ldm68_orig, ldm106_orig, h, w):
        """
        METHOD 1: Eye aperture shrinkage.
        Normal: aperture/IOD ~ 0.25-0.35
        Mask: aperture/IOD < 0.20
        """
        iod = self._get_iod(mp_lm, ldm68_orig)
        details = {}
        apertures = []

        for side in ['left', 'right']:
            ap = self._compute_aperture(mp_lm, ldm68_orig, side, iod)
            if ap is not None:
                apertures.append(ap)
                details[f'aperture_{side}'] = safe_float(ap)

        if not apertures:
            return 0.0, details

        mean_ap = np.mean(apertures)
        details['mean_aperture'] = safe_float(mean_ap)

        # Score: 0 at 0.28 (normal), 1 at 0.15 (mask)
        score = np.clip((0.28 - mean_ap) / 0.13, 0, 1)
        return safe_float(score), details

    def _compute_aperture(self, mp_lm, ldm68_orig, side, iod):
        """Compute aperture/IOD for one eye."""
        if mp_lm is not None and len(mp_lm) > 386:
            if side == 'left':
                upper = mp_lm[159, :2]  # upper lid
                lower = mp_lm[145, :2]  # lower lid
                # Also use additional points
                upper2 = mp_lm[160, :2]
                lower2 = mp_lm[153, :2]
            else:
                upper = mp_lm[386, :2]
                lower = mp_lm[374, :2]
                upper2 = mp_lm[387, :2]
                lower2 = mp_lm[380, :2]

            ap1 = np.linalg.norm(upper - lower)
            ap2 = np.linalg.norm(upper2 - lower2)
            aperture = max(ap1, ap2)
            return aperture / max(iod, 1)

        if ldm68_orig is not None and len(ldm68_orig) > 47:
            if side == 'left':
                upper = (ldm68_orig[37, :2] + ldm68_orig[38, :2]) / 2
                lower = (ldm68_orig[40, :2] + ldm68_orig[41, :2]) / 2
            else:
                upper = (ldm68_orig[43, :2] + ldm68_orig[44, :2]) / 2
                lower = (ldm68_orig[46, :2] + ldm68_orig[47, :2]) / 2
            aperture = np.linalg.norm(upper - lower)
            return aperture / max(iod, 1)

        return None

    def _eyelid_thickness(self, image_rgb, mp_lm, ldm68_orig, h, w):
        """
        METHOD 3: Eyelid thickness (double layer).
        Measure gradient zone width along eyelid contour.
        """
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        details = {}
        thicknesses = []

        for side in ['left', 'right']:
            if mp_lm is None:
                continue

            if side == 'left':
                lid_idx = [159, 160, 158, 157, 173]  # upper lid points
            else:
                lid_idx = [386, 387, 385, 384, 398]

            valid = [i for i in lid_idx if i < len(mp_lm)]
            if len(valid) < 2:
                continue

            pts = mp_lm[valid, :2].astype(int)

            # Sample gradient width at each lid point
            widths = []
            for pt in pts:
                x, y = pt
                if 5 <= y < h - 5 and 5 <= x < w - 5:
                    # Vertical gradient profile through lid
                    col = gray[max(0, y - 10):min(h, y + 10), x]
                    if len(col) > 4:
                        grad = np.abs(np.diff(col))
                        high = grad > np.percentile(grad, 70)
                        if high.any():
                            nz = np.nonzero(high)[0]
                            widths.append(nz[-1] - nz[0] + 1)

            if widths:
                med_w = np.median(widths)
                thicknesses.append(med_w)
                details[f'lid_width_{side}_px'] = safe_float(med_w)

        if not thicknesses:
            return 0.0, details

        mean_thick = np.mean(thicknesses)
        details['mean_lid_thickness_px'] = safe_float(mean_thick)

        # Normal: 8-12 px, Mask: > 12 px (thicker gradient zone from mask edge)
        score = np.clip((mean_thick - 8) / 10, 0, 1)
        return safe_float(score), details

    def _lid_smoothness(self, image_rgb, mp_lm, ldm68_orig, h, w):
        """
        METHOD 4: Lid smoothness (silicone = unnaturally smooth).
        Laplacian variance on eyelid region.
        """
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        details = {}
        lap_vars = []

        for side in ['left', 'right']:
            crop, bbox = self._get_eye_crop(image_rgb, mp_lm, ldm68_orig, side, expand=1.0)
            if crop is None:
                continue

            crop_gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            ch = crop_gray.shape[0]

            # Upper third = lid region
            lid_region = crop_gray[:max(ch // 3, 3), :]
            if lid_region.size < 16:
                continue

            lap_var = float(np.var(cv2.Laplacian(lid_region.astype(np.float32), cv2.CV_32F)))
            lap_vars.append(lap_var)
            details[f'lid_lapvar_{side}'] = safe_float(lap_var)

        if not lap_vars:
            return 0.0, details

        mean_var = np.mean(lap_vars)
        details['mean_lid_lapvar'] = safe_float(mean_var)

        # Real skin: high Laplacian variance (>200), Silicone: low variance (<100)
        score = np.clip((200 - mean_var) / 150, 0, 1)
        return safe_float(score), details

    def _eyelid_edge_sharpness(self, image_rgb, mp_lm, ldm68_orig, h, w):
        """
        METHOD 7: Eyelid edge sharpness.
        Mask: sharp artificial edge. Normal: gradual skin-sclera transition.
        """
        details = {}
        grads = []

        for side in ['left', 'right']:
            crop, bbox = self._get_eye_crop(image_rgb, mp_lm, ldm68_orig, side, expand=1.0)
            if crop is None:
                continue

            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)

            # Edge detection in crop
            edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
            grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

            # Gradient at edges
            edge_grads = grad_mag[edges > 0]
            if len(edge_grads) > 10:
                med_grad = np.median(edge_grads)
                grads.append(med_grad)
                details[f'edge_grad_{side}'] = safe_float(med_grad)

        if not grads:
            return 0.0, details

        mean_grad = np.mean(grads)
        details['mean_edge_gradient'] = safe_float(mean_grad)

        # Normal: 40-70, Mask: > 80
        score = np.clip((mean_grad - 70) / 20, 0, 1)
        return safe_float(score), details

    def _eye_symmetry(self, mp_lm, ldm68_orig, h, w):
        """
        METHOD 9: Eye symmetry (aperture L vs R).
        Mask may be asymmetrically fitted.
        """
        iod = self._get_iod(mp_lm, ldm68_orig)
        details = {}

        ap_L = self._compute_aperture(mp_lm, ldm68_orig, 'left', iod)
        ap_R = self._compute_aperture(mp_lm, ldm68_orig, 'right', iod)

        if ap_L is None or ap_R is None:
            return 0.0, details

        details['aperture_L'] = safe_float(ap_L)
        details['aperture_R'] = safe_float(ap_R)

        asymmetry = abs(ap_L - ap_R) / max(ap_L, ap_R, 1e-6)
        details['asymmetry'] = safe_float(asymmetry)

        # Normal: < 0.08, Mask (crooked): > 0.20
        score = np.clip((asymmetry - 0.08) / 0.12, 0, 1)
        return safe_float(score), details

    def _sclera_iris_boundary(self, image_rgb, mp_lm, h, w):
        """
        METHOD 11: Sclera-iris boundary sharpness.
        Mask: sharper boundary (shadow from mask edge).
        Normal: gradual transition.
        """
        details = {}
        sharpness_vals = []

        if mp_lm is None or not mp_lm.shape[0] > 477:
            return 0.0, details

        for side, iris_idx in [('left', MP_LEFT_IRIS), ('right', MP_RIGHT_IRIS)]:
            iris_center = mp_lm[iris_idx[0], :2].astype(int)
            iris_edges = mp_lm[iris_idx[1:], :2]
            iris_r = int(np.mean([np.linalg.norm(e - iris_center) for e in iris_edges]))

            if iris_r < 3:
                continue

            # Sample radial gradient at iris boundary
            cx, cy = iris_center
            angles = np.linspace(0, 2 * np.pi, 36)
            grads = []
            for angle in angles:
                x = int(cx + iris_r * np.cos(angle))
                y = int(cy + iris_r * np.sin(angle))
                if 2 <= x < w - 2 and 2 <= y < h - 2:
                    patch = image_rgb[y - 2:y + 3, x - 2:x + 3].astype(np.float32)
                    g = np.max(np.abs(np.gradient(patch.mean(axis=2))))
                    grads.append(g)

            if grads:
                med_grad = np.median(grads)
                sharpness_vals.append(med_grad)
                details[f'sclera_iris_grad_{side}'] = safe_float(med_grad)

        if not sharpness_vals:
            return 0.0, details

        mean_sharp = np.mean(sharpness_vals)
        details['mean_sclera_iris_sharpness'] = safe_float(mean_sharp)

        # Normal: ~27-29, Mask: ~26-28 (slightly smoother/blurred boundary)
        score = np.clip((29 - mean_sharp) / 4, 0, 1)
        return safe_float(score), details

    def _pupil_apparent_size(self, image_rgb, mp_lm, h, w):
        """
        METHOD 12: Pupil apparent size.
        Measure dark region (pupil+iris) relative to visible eye area.
        """
        details = {}
        ratios = []

        if mp_lm is None or not mp_lm.shape[0] > 477:
            return 0.0, details

        for side, eye_outline, iris_idx in [
            ('left', MP_LEFT_EYE_OUTLINE, MP_LEFT_IRIS),
            ('right', MP_RIGHT_EYE_OUTLINE, MP_RIGHT_IRIS),
        ]:
            valid = [i for i in eye_outline if i < len(mp_lm)]
            if len(valid) < 4:
                continue

            pts = mp_lm[valid, :2].astype(int)
            iris_center = mp_lm[iris_idx[0], :2]
            iris_edges = mp_lm[iris_idx[1:], :2]
            iris_diam = 2 * np.mean([np.linalg.norm(e - iris_center) for e in iris_edges])

            # Eye area from polygon
            eye_area = cv2.contourArea(pts.astype(np.float32))
            iris_area = np.pi * (iris_diam / 2) ** 2

            if eye_area < 1:
                continue

            ratio = iris_area / eye_area
            ratios.append(ratio)
            details[f'pupil_area_ratio_{side}'] = safe_float(ratio)
            details[f'eye_area_{side}_px'] = safe_float(eye_area)
            details[f'iris_area_{side}_px'] = safe_float(iris_area)

        if not ratios:
            return 0.0, details

        mean_ratio = np.mean(ratios)
        details['mean_pupil_area_ratio'] = safe_float(mean_ratio)

        # Iris-to-eye-area ratio: normal ~0.4-0.55, mask (shrunken aperture) > 0.65
        score = np.clip((mean_ratio - 0.55) / 0.15, 0, 1)
        return safe_float(score), details

    def _orbit_area_ratio(self, seg_vis, mp_lm, ddffa, h, w, trans_params):
        """
        METHOD 13: Orbit area from 3DDFA segmentation.
        Compare visible eye area in segmentation vs expected.
        """
        details = {}

        if seg_vis is None:
            return 0.0, details

        # seg_visible: (224, 224, 8), parts: [right_eye, left_eye, ...]
        right_eye_seg = seg_vis[:, :, 0]
        left_eye_seg = seg_vis[:, :, 1]

        right_area = (right_eye_seg > 0.5).sum()
        left_area = (left_eye_seg > 0.5).sum()

        # Total face area (skin)
        skin_seg = seg_vis[:, :, 7]
        face_area = max((skin_seg > 0.5).sum(), 1)

        # Normalize
        orbit_ratio_R = right_area / face_area
        orbit_ratio_L = left_area / face_area

        details['orbit_area_right'] = safe_float(right_area)
        details['orbit_area_left'] = safe_float(left_area)
        details['face_area'] = safe_float(face_area)
        details['orbit_ratio_R'] = safe_float(orbit_ratio_R)
        details['orbit_ratio_L'] = safe_float(orbit_ratio_L)

        mean_ratio = (orbit_ratio_R + orbit_ratio_L) / 2
        details['mean_orbit_ratio'] = safe_float(mean_ratio)

        # Normal orbit/face ratio: ~0.008-0.015
        # Mask (shrunken orbit): < 0.006
        score = np.clip((0.010 - mean_ratio) / 0.006, 0, 1)
        return safe_float(score), details

    def _periocular_lbp(self, image_rgb, mp_lm, ldm68_orig, h, w):
        """
        METHOD 14: Periocular LBP entropy.
        Mask: low LBP entropy (uniform texture).
        Normal skin: higher entropy (pores, wrinkles).
        """
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        details = {}
        entropies = []

        for side in ['left', 'right']:
            crop, bbox = self._get_eye_crop(image_rgb, mp_lm, ldm68_orig, side, expand=1.8)
            if crop is None:
                continue
            crop_gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

            # LBP on entire periocular region
            ch, cw = crop_gray.shape
            if ch < 5 or cw < 5:
                continue

            center = crop_gray[2:-2, 2:-2].astype(np.uint8)
            lbp = np.zeros_like(center, dtype=np.uint8)
            offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
            for i, (dy, dx) in enumerate(offsets):
                nb = crop_gray[2 + dy:ch - 2 + dy, 2 + dx:cw - 2 + dx]
                lbp |= ((nb >= center).astype(np.uint8) << i)

            hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
            hist = hist.astype(float)
            hist /= hist.sum() + 1e-6
            ent = -np.sum(hist * np.log2(hist + 1e-10))
            entropies.append(ent)
            details[f'peri_lbp_entropy_{side}'] = safe_float(ent)

        if not entropies:
            return 0.0, details

        mean_ent = np.mean(entropies)
        details['mean_peri_lbp_entropy'] = safe_float(mean_ent)

        # Normal skin: entropy ~6.5-6.7, Silicone: lower ~6.3-6.5
        score = np.clip((6.65 - mean_ent) / 0.35, 0, 1)
        return safe_float(score), details

    def _eyelid_sss(self, image_rgb, mp_lm, ldm68_orig, h, w):
        """
        METHOD 15: Subsurface scattering on eyelids.
        Thin skin (eyelids) has high R-B diff.
        Silicone: R-B ≈ 0.
        """
        details = {}
        rb_diffs = []

        for side in ['left', 'right']:
            if mp_lm is None:
                continue

            if side == 'left':
                brow_y = int(mp_lm[105, 1]) if len(mp_lm) > 105 else 0
                eye_y = int(mp_lm[159, 1]) if len(mp_lm) > 159 else 0
                eye_x = int(mp_lm[33, 0]) if len(mp_lm) > 33 else 0
                eye_x2 = int(mp_lm[133, 0]) if len(mp_lm) > 133 else 0
            else:
                brow_y = int(mp_lm[334, 1]) if len(mp_lm) > 334 else 0
                eye_y = int(mp_lm[386, 1]) if len(mp_lm) > 386 else 0
                eye_x = int(mp_lm[362, 0]) if len(mp_lm) > 362 else 0
                eye_x2 = int(mp_lm[263, 0]) if len(mp_lm) > 263 else 0

            y1 = max(0, min(brow_y, eye_y) - 5)
            y2 = min(h, max(brow_y, eye_y) + 5)
            x1 = max(0, min(eye_x, eye_x2) - 5)
            x2 = min(w, max(eye_x, eye_x2) + 5)

            if y2 <= y1 + 2 or x2 <= x1 + 2:
                continue

            patch = image_rgb[y1:y2, x1:x2]
            R = patch[:, :, 0].astype(float).mean()
            B = patch[:, :, 2].astype(float).mean()
            rb_diff = R - B
            rb_diffs.append(rb_diff)
            details[f'lid_RB_diff_{side}'] = safe_float(rb_diff)
            details[f'lid_R_{side}'] = safe_float(R)
            details[f'lid_B_{side}'] = safe_float(B)

        if not rb_diffs:
            return 0.0, details

        mean_rb = np.mean(rb_diffs)
        details['mean_lid_RB_diff'] = safe_float(mean_rb)

        # Normal: R-B ~30-35 (subsurface scattering through thin eyelid skin)
        # Silicone: R-B ~38-45 (more red due to material properties)
        score = np.clip((mean_rb - 28) / 20, 0, 1)
        return safe_float(score), details

    def _iris_visible_ratio(self, mp_lm, ldm134_orig, h, w):
        """
        METHOD 16: Iris visible area ratio.
        Uses ldm134 upper/lower eyelid contours + MediaPipe iris landmarks
        to compute what fraction of the iris circle is visible through the
        eye opening (palpebral fissure).

        Mask: less iris visible (aperture shrunken, eyelids cover more).
        Real: more iris visible (natural eye opening).
        """
        details = {}

        if mp_lm is None or not (mp_lm.shape[0] > 477):
            return 0.0, details
        if ldm134_orig is None or len(ldm134_orig) < 134:
            return 0.0, details

        ratios = []

        for side in ['left', 'right']:
            if side == 'left':
                # ldm134 upper lid: points 0-10 (left eye upper contour)
                upper_pts = ldm134_orig[0:11, :2]
                # ldm134 lower lid: points 11-21 (left eye lower contour)
                lower_pts = ldm134_orig[11:22, :2]
                # MediaPipe iris center + edge points
                iris_center_idx = 468
                iris_edge_idx = [469, 470, 471, 472]
                # MediaPipe eye outline for eye opening width reference
                eye_outline_idx = MP_LEFT_EYE_OUTLINE
            else:
                # ldm134 upper lid: points 22-32 (right eye upper contour)
                upper_pts = ldm134_orig[22:33, :2]
                # ldm134 lower lid: points 33-43 (right eye lower contour)
                lower_pts = ldm134_orig[33:44, :2]
                # MediaPipe iris center + edge points
                iris_center_idx = 473
                iris_edge_idx = [474, 475, 476, 477]
                eye_outline_idx = MP_RIGHT_EYE_OUTLINE

            # Validate iris points exist
            if max(iris_center_idx, max(iris_edge_idx)) >= len(mp_lm):
                continue

            # Iris center and radius from MediaPipe
            iris_center = mp_lm[iris_center_idx, :2]
            iris_edges = mp_lm[iris_edge_idx, :2]
            iris_radius = np.mean([np.linalg.norm(e - iris_center) for e in iris_edges])

            if iris_radius < 2:
                continue

            # Build eye opening polygon from ldm134 eyelid contours
            # Upper lid goes left-to-right, lower lid goes right-to-left → closed polygon
            eye_contour = np.vstack([
                upper_pts,
                lower_pts[::-1]  # reverse lower lid to close polygon
            ]).astype(np.float32)

            if len(eye_contour) < 3:
                continue

            # Compute eye opening height at iris x-position
            # Find upper lid Y and lower lid Y at iris center X
            iris_x = iris_center[0]

            # Interpolate upper lid Y at iris_x
            upper_sorted = upper_pts[np.argsort(upper_pts[:, 0])]
            if iris_x < upper_sorted[0, 0] or iris_x > upper_sorted[-1, 0]:
                continue
            upper_y_at_iris = np.interp(iris_x, upper_sorted[:, 0], upper_sorted[:, 1])

            # Interpolate lower lid Y at iris_x
            lower_sorted = lower_pts[np.argsort(lower_pts[:, 0])]
            if iris_x < lower_sorted[0, 0] or iris_x > lower_sorted[-1, 0]:
                continue
            lower_y_at_iris = np.interp(iris_x, lower_sorted[:, 0], lower_sorted[:, 1])

            eye_opening_height = abs(lower_y_at_iris - upper_y_at_iris)

            # Iris diameter
            iris_diameter = 2 * iris_radius

            # Ratio: how much of iris diameter fits in eye opening
            # > 1.0 means iris is fully visible (opening > iris diameter)
            # < 1.0 means eyelids partially cover iris
            if iris_diameter < 1:
                continue

            vis_ratio = eye_opening_height / iris_diameter
            ratios.append(vis_ratio)
            details[f'iris_vis_ratio_{side}'] = safe_float(vis_ratio)
            details[f'eye_opening_h_{side}'] = safe_float(eye_opening_height)
            details[f'iris_diam_{side}'] = safe_float(iris_diameter)
            details[f'upper_lid_y_{side}'] = safe_float(upper_y_at_iris)
            details[f'lower_lid_y_{side}'] = safe_float(lower_y_at_iris)

        if not ratios:
            return 0.0, details

        mean_ratio = np.mean(ratios)
        details['mean_iris_vis_ratio'] = safe_float(mean_ratio)

        # Real: ~0.95-1.10 (iris largely visible, natural overlap with lids)
        # Mask: < 0.85 (aperture shrunken, much less iris visible)
        # Score: 1 = mask (low ratio), 0 = normal (high ratio)
        score = np.clip((0.95 - mean_ratio) / 0.20, 0, 1)
        return safe_float(score), details

    def _iris_center_offset(self, mp_lm, ldm134_orig, h, w):
        """
        METHOD 15: Iris center offset within eye opening.
        Measures how centered the iris is inside the palpebral fissure.
        Real: iris centered (vertically and horizontally).
        Mask: iris shifted (aperture geometry forces iris off-center).
        """
        details = {}
        if mp_lm is None or not (mp_lm.shape[0] > 477):
            return 0.0, details
        if ldm134_orig is None or len(ldm134_orig) < 44:
            return 0.0, details

        offsets = []

        for side in ['left', 'right']:
            if side == 'left':
                upper_pts = ldm134_orig[0:11, :2]
                lower_pts = ldm134_orig[11:22, :2]
                iris_center_idx = 468
                iris_edge_idx = [469, 470, 471, 472]
            else:
                upper_pts = ldm134_orig[22:33, :2]
                lower_pts = ldm134_orig[33:44, :2]
                iris_center_idx = 473
                iris_edge_idx = [474, 475, 476, 477]

            if max(iris_center_idx, max(iris_edge_idx)) >= len(mp_lm):
                continue

            iris_center = mp_lm[iris_center_idx, :2]
            iris_edges = mp_lm[iris_edge_idx, :2]
            iris_radius = np.mean([np.linalg.norm(e - iris_center) for e in iris_edges])
            if iris_radius < 2:
                continue

            iris_x = iris_center[0]

            # Upper lid Y at iris x
            upper_sorted = upper_pts[np.argsort(upper_pts[:, 0])]
            if iris_x < upper_sorted[0, 0] or iris_x > upper_sorted[-1, 0]:
                continue
            upper_y = np.interp(iris_x, upper_sorted[:, 0], upper_sorted[:, 1])

            # Lower lid Y at iris x
            lower_sorted = lower_pts[np.argsort(lower_pts[:, 0])]
            if iris_x < lower_sorted[0, 0] or iris_x > lower_sorted[-1, 0]:
                continue
            lower_y = np.interp(iris_x, lower_sorted[:, 0], lower_sorted[:, 1])

            # Vertical offset: how far iris center is from eye opening midpoint
            eye_mid_y = (upper_y + lower_y) / 2
            eye_height = abs(lower_y - upper_y)
            if eye_height < 1:
                continue

            vert_offset = abs(iris_center[1] - eye_mid_y) / eye_height
            offsets.append(vert_offset)
            details[f'vert_offset_{side}'] = safe_float(vert_offset)
            details[f'eye_mid_y_{side}'] = safe_float(eye_mid_y)
            details[f'iris_center_y_{side}'] = safe_float(iris_center[1])

        if not offsets:
            return 0.0, details

        mean_offset = np.mean(offsets)
        details['mean_iris_center_offset'] = safe_float(mean_offset)

        # Real: iris centered (offset < 0.15)
        # Mask: iris off-center (offset > 0.20, aperture shape distorts)
        score = np.clip((mean_offset - 0.12) / 0.15, 0, 1)
        return safe_float(score), details

    def _sclera_asymmetry(self, image_rgb, mp_lm, h, w):
        """
        METHOD 16: Sclera visible area asymmetry L vs R.
        Real: roughly symmetric sclera visibility.
        Mask: asymmetric (aperture cut unevenly, one eye more covered).
        """
        details = {}
        if mp_lm is None or not (mp_lm.shape[0] > 477):
            return 0.0, details

        sclera_areas = []

        for side, eye_idx, iris_idx in [
            ('left', MP_LEFT_EYE_OUTLINE, MP_LEFT_IRIS),
            ('right', MP_RIGHT_EYE_OUTLINE, MP_RIGHT_IRIS),
        ]:
            valid_eye = [i for i in eye_idx if i < len(mp_lm)]
            if len(valid_eye) < 4:
                continue

            eye_pts = mp_lm[valid_eye, :2].astype(np.float32)
            eye_area = cv2.contourArea(eye_pts)
            if eye_area < 1:
                continue

            # Iris area
            iris_c = mp_lm[iris_idx[0], :2]
            iris_e = mp_lm[iris_idx[1:], :2]
            iris_r = np.mean([np.linalg.norm(e - iris_c) for e in iris_e])
            iris_area = np.pi * iris_r ** 2

            # Sclera = eye opening minus iris
            sclera = max(eye_area - iris_area, 0)
            sclera_areas.append(sclera)
            details[f'sclera_area_{side}'] = safe_float(sclera)
            details[f'eye_area_{side}'] = safe_float(eye_area)
            details[f'iris_area_{side}'] = safe_float(iris_area)

        if len(sclera_areas) < 2:
            return 0.0, details

        # Asymmetry = |left - right| / max(left, right)
        asymmetry = abs(sclera_areas[0] - sclera_areas[1]) / max(max(sclera_areas), 1)
        details['sclera_asymmetry'] = safe_float(asymmetry)

        # Real: < 0.15 (symmetric)
        # Mask: > 0.25 (asymmetric aperture)
        score = np.clip((asymmetry - 0.12) / 0.18, 0, 1)
        return safe_float(score), details


# =============================================================================
# CROSS-SYSTEM METHODS (3DDFA + MediaPipe combined)
# =============================================================================

class CrossSystemDetector:
    """
    Methods combining 3DDFA-V3 and MediaPipe data.
    """

    def run_all(self, image_rgb, ddffa, mp_data):
        scores = {}
        details = {}

        mp_lm = mp_data.get('landmarks')
        ldm68 = ddffa.get('ldm68')
        trans_params = ddffa.get('trans_params')
        h, w = image_rgb.shape[:2]

        # Back-project 3DDFA ldm68
        ldm68_orig = None
        if ldm68 is not None and trans_params is not None:
            ldm68_copy = ldm68.copy()
            ldm68_copy[:, 1] = 224 - 1 - ldm68_copy[:, 1]
            ldm68_orig = back_resize_ldms(ldm68_copy, trans_params)

        # === METHOD 16: Landmark Discrepancy (INVERTED: real>silicone → 1-score) ===
        raw_disc, details['lm_disc'] = self._landmark_discrepancy(
            mp_lm, ldm68_orig, h, w)
        scores['landmark_discrepancy'] = safe_float(1.0 - raw_disc)

        # === METHOD 17: Specular BRDF (INVERTED: real>silicone → 1-score) ===
        raw_spec, details['specular'] = self._specular_analysis(image_rgb, mp_lm, h, w)
        scores['specular_brdf'] = safe_float(1.0 - raw_spec)

        # === METHOD 18: Eye contour divergence (INVERTED: real>silicone → 1-score) ===
        raw_contour, details['contour'] = self._eye_contour_divergence(
            mp_lm, ldm68_orig, h, w)
        scores['eye_contour_divergence'] = safe_float(1.0 - raw_contour)

        # === METHOD 19: Subsurface Light Transport ===
        scores['subsurface_violation'], details['sss_full'] = self._subsurface_analysis(
            image_rgb, mp_lm, h, w)

        # === METHOD 20: Ear texture cliff ===
        scores['ear_texture_cliff'], details['ear'] = self._ear_cliff(
            image_rgb, mp_lm, h, w)

        # === METHOD 21: Skin tone UV mismatch (INVERTED: real>silicone → 1-score) ===
        raw_tone, details['tone'] = self._skin_tone(
            image_rgb, ddffa, mp_lm, h, w)
        scores['skin_tone_mismatch'] = safe_float(1.0 - raw_tone)

        return safe_dict(scores), details

    def _get_iod(self, mp_lm, ldm68_orig):
        if mp_lm is not None and len(mp_lm) > 263:
            return np.linalg.norm(mp_lm[263, :2] - mp_lm[33, :2])
        if ldm68_orig is not None and len(ldm68_orig) > 45:
            return np.linalg.norm(ldm68_orig[42, :2] - ldm68_orig[36, :2])
        return 100.0

    def _landmark_discrepancy(self, mp_lm, ldm68_orig, h, w):
        """
        METHOD 16: Compare 3DDFA projected landmarks vs MediaPipe landmarks.
        """
        details = {}
        if mp_lm is None or ldm68_orig is None:
            return 0.0, details

        iod = self._get_iod(mp_lm, ldm68_orig)

        # Map 3DDFA 68pts to approximate MP indices
        mapping = {
            36: 33, 37: 160, 38: 159, 39: 133, 40: 145, 41: 153,  # left eye
            42: 362, 43: 386, 44: 385, 45: 263, 46: 374, 47: 380,  # right eye
            30: 1,  # nose tip
            48: 61, 54: 291,  # mouth corners
            8: 152,  # chin
        }

        zone_errors = {'eye_L': [], 'eye_R': [], 'nose': [], 'mouth': [], 'jaw': []}

        for idx_3ddfa, idx_mp in mapping.items():
            if idx_3ddfa >= len(ldm68_orig) or idx_mp >= len(mp_lm):
                continue

            pt_3d = ldm68_orig[idx_3ddfa, :2]
            pt_mp = mp_lm[idx_mp, :2]
            error = np.linalg.norm(pt_3d - pt_mp) / max(iod, 1)

            if 36 <= idx_3ddfa <= 41:
                zone_errors['eye_L'].append(error)
            elif 42 <= idx_3ddfa <= 47:
                zone_errors['eye_R'].append(error)
            elif idx_3ddfa == 30:
                zone_errors['nose'].append(error)
            elif idx_3ddfa in [48, 54]:
                zone_errors['mouth'].append(error)
            elif idx_3ddfa == 8:
                zone_errors['jaw'].append(error)

        for zone, errors in zone_errors.items():
            if errors:
                details[f'disc_{zone}'] = safe_float(np.mean(errors))

        # Eye discrepancy is key
        eye_disc = []
        if zone_errors['eye_L']:
            eye_disc.extend(zone_errors['eye_L'])
        if zone_errors['eye_R']:
            eye_disc.extend(zone_errors['eye_R'])

        non_eye_disc = []
        for z in ['nose', 'mouth', 'jaw']:
            if zone_errors[z]:
                non_eye_disc.extend(zone_errors[z])

        if not eye_disc:
            return 0.0, details

        mean_eye = np.mean(eye_disc)
        mean_other = np.mean(non_eye_disc) if non_eye_disc else mean_eye

        details['mean_eye_discrepancy'] = safe_float(mean_eye)
        details['mean_other_discrepancy'] = safe_float(mean_other)

        # Pattern: eye discrepancy >> other discrepancy = mask
        pattern = max(mean_eye - mean_other, 0)
        details['disc_pattern_score'] = safe_float(pattern)

        score = np.clip(pattern * 15, 0, 1)
        return safe_float(score), details

    def _specular_analysis(self, image_rgb, mp_lm, h, w):
        """
        METHOD 17: Specular highlights analysis.
        Silicone: sharp, bright, elongated specular.
        """
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        details = {}

        # Find specular highlights
        _, bright = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        bright_ratio = bright.sum() / (255 * bright.size)
        details['specular_bright_ratio'] = safe_float(bright_ratio)

        # Analyze specular sharpness
        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sharpness_vals = []
        elongations = []

        for c in contours:
            area = cv2.contourArea(c)
            if area > 20:
                # Sharpness: gradient at edge
                mask = np.zeros_like(gray)
                cv2.drawContours(mask, [c], -1, 255, -1)
                edge = cv2.Canny(mask, 100, 200)
                grad_x = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
                grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
                edge_grads = grad_mag[edge > 0]
                if len(edge_grads) > 5:
                    sharpness_vals.append(np.median(edge_grads))

                # Elongation
                rect = cv2.minAreaRect(c)
                w_box, h_box = rect[1]
                if min(w_box, h_box) > 0:
                    elongations.append(max(w_box, h_box) / min(w_box, h_box))

        if sharpness_vals:
            details['specular_sharpness'] = safe_float(np.mean(sharpness_vals))
        if elongations:
            details['specular_elongation'] = safe_float(np.mean(elongations))

        # Score
        sharp_score = np.clip((details.get('specular_sharpness', 30) - 30) / 30, 0, 1)
        elong_score = np.clip((details.get('specular_elongation', 1.5) - 1.5) / 2.0, 0, 1)
        coverage_score = np.clip(bright_ratio * 20, 0, 1)

        score = sharp_score * 0.4 + elong_score * 0.3 + coverage_score * 0.3
        return safe_float(score), details

    def _eye_contour_divergence(self, mp_lm, ldm68_orig, h, w):
        """
        METHOD 18: Eye contour shape divergence between systems.
        """
        details = {}
        iod = self._get_iod(mp_lm, ldm68_orig)
        divergences = []

        for side in ['left', 'right']:
            if mp_lm is None or ldm68_orig is None:
                continue

            if side == 'left':
                # MP: 6 key points along upper lid
                mp_upper = [mp_lm[i, :2] for i in [159, 160, 158] if i < len(mp_lm)]
                # 3DDFA: upper lid
                dd_upper = [ldm68_orig[i, :2] for i in [37, 38] if i < len(ldm68_orig)]
                # Lower lid
                mp_lower = [mp_lm[i, :2] for i in [145, 153, 154] if i < len(mp_lm)]
                dd_lower = [ldm68_orig[i, :2] for i in [40, 41] if i < len(ldm68_orig)]
            else:
                mp_upper = [mp_lm[i, :2] for i in [386, 387, 385] if i < len(mp_lm)]
                dd_upper = [ldm68_orig[i, :2] for i in [43, 44] if i < len(ldm68_orig)]
                mp_lower = [mp_lm[i, :2] for i in [374, 380, 381] if i < len(mp_lm)]
                dd_lower = [ldm68_orig[i, :2] for i in [46, 47] if i < len(ldm68_orig)]

            # Compare upper lid Y positions
            if mp_upper and dd_upper:
                mp_y = np.mean([p[1] for p in mp_upper])
                dd_y = np.mean([p[1] for p in dd_upper])
                div_upper = abs(mp_y - dd_y) / max(iod, 1)
                divergences.append(div_upper)
                details[f'contour_upper_div_{side}'] = safe_float(div_upper)

            if mp_lower and dd_lower:
                mp_y = np.mean([p[1] for p in mp_lower])
                dd_y = np.mean([p[1] for p in dd_lower])
                div_lower = abs(mp_y - dd_y) / max(iod, 1)
                divergences.append(div_lower)
                details[f'contour_lower_div_{side}'] = safe_float(div_lower)

        if not divergences:
            return 0.0, details

        mean_div = np.mean(divergences)
        details['mean_contour_divergence'] = safe_float(mean_div)

        # Normal: < 0.02, Mask: > 0.05
        score = np.clip((mean_div - 0.02) / 0.04, 0, 1)
        return safe_float(score), details

    def _subsurface_analysis(self, image_rgb, mp_lm, h, w):
        """
        METHOD 19: Subsurface light transport violation.
        """
        details = {}

        # R/B ratio on thin areas
        thin_zones = {
            'eyelid': [159, 386],
            'ear': [234, 454],
            'nose_wing': [129, 358],
        }

        rb_ratios = {}
        for zone, indices in thin_zones.items():
            r_vals, b_vals = [], []
            if mp_lm is None:
                continue
            for idx in indices:
                if idx >= len(mp_lm):
                    continue
                x, y = int(mp_lm[idx, 0]), int(mp_lm[idx, 1])
                r = 5
                patch = image_rgb[max(0, y - r):y + r, max(0, x - r):x + r]
                if patch.size > 0:
                    r_vals.append(patch[:, :, 0].mean())
                    b_vals.append(patch[:, :, 2].mean())

            if r_vals:
                rb = np.mean(r_vals) / max(np.mean(b_vals), 1)
                rb_ratios[zone] = rb
                details[f'RB_ratio_{zone}'] = safe_float(rb)

        # R-G correlation on cheek
        if mp_lm is not None and len(mp_lm) > 117:
            cx = int(mp_lm[117, 0])
            cy = int(mp_lm[117, 1])
            r = 25
            patch = image_rgb[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
            if patch.size > 0:
                R = patch[:, :, 0].ravel().astype(float)
                G = patch[:, :, 1].ravel().astype(float)
                if len(R) > 10:
                    corr = float(np.corrcoef(R, G)[0, 1])
                    details['RG_correlation'] = safe_float(corr)

        # Score
        scores_parts = []
        if 'eyelid' in rb_ratios:
            scores_parts.append(np.clip((1.3 - rb_ratios['eyelid']) / 0.4, 0, 1))
        if 'nose_wing' in rb_ratios:
            scores_parts.append(np.clip((1.4 - rb_ratios['nose_wing']) / 0.4, 0, 1))
        if 'RG_correlation' in details:
            scores_parts.append(np.clip((details['RG_correlation'] - 0.85) / 0.12, 0, 1))

        if not scores_parts:
            return 0.0, details

        return safe_float(np.mean(scores_parts)), details

    def _ear_cliff(self, image_rgb, mp_lm, h, w):
        """
        METHOD 22: Ear region texture cliff.
        """
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        details = {}

        if mp_lm is None:
            return 0.0, details

        cliffs = []
        for ear_idx, face_idx in [(234, 117), (454, 345)]:
            if ear_idx >= len(mp_lm) or face_idx >= len(mp_lm):
                continue

            ex, ey = int(mp_lm[ear_idx, 0]), int(mp_lm[ear_idx, 1])
            fx, fy = int(mp_lm[face_idx, 0]), int(mp_lm[face_idx, 1])

            r = 15
            ear_patch = gray[max(0, ey - r):ey + r, max(0, ex - r):ex + r]
            face_patch = gray[max(0, fy - r):fy + r, max(0, fx - r):fx + r]

            if ear_patch.size > 10 and face_patch.size > 10:
                # Compare variance (texture)
                var_diff = abs(np.var(ear_patch) - np.var(face_patch))
                cliffs.append(var_diff)

        if cliffs:
            mean_cliff = np.mean(cliffs)
            details['ear_texture_cliff'] = safe_float(mean_cliff)
            # Real skin: high cliff (600+) at ear boundary (complex ear anatomy)
            # Silicone mask: low cliff (100-200) uniform material, no ear detail
            score = np.clip(1.0 - mean_cliff / 500, 0, 1)
            return safe_float(score), details

        return 0.0, details

    def _skin_tone(self, image_rgb, ddffa, mp_lm, h, w):
        """
        METHOD 25: Skin tone consistency.
        """
        details = {}

        face_texture = ddffa.get('face_texture')
        if face_texture is None or mp_lm is None:
            return 0.0, details

        # face_texture: (35709, 3) - per-vertex texture from 3DDFA
        # Compare with actual image colors at same positions

        v2d = ddffa.get('v2d')
        if v2d is None:
            return 0.0, details

        # Sample cheek vertices
        iod = self._get_iod(mp_lm, None)

        # Find vertices in cheek zone
        tree = cKDTree(v2d[:, :2])
        if len(mp_lm) > 117:
            cheek_pos = mp_lm[117, :2] * 224.0 / max(w, 1)  # approximate mapping
            _, cheek_idx = tree.query(cheek_pos, k=min(50, len(v2d)))

            # Compare texture
            tex_3ddfa = face_texture[cheek_idx] if isinstance(cheek_idx, (list, np.ndarray)) else face_texture[[cheek_idx]]
            tex_mean = tex_3ddfa.mean(axis=0) if len(tex_3ddfa.shape) > 1 else tex_3ddfa

            # Actual image color at same position
            cx, cy = int(mp_lm[117, 0]), int(mp_lm[117, 1])
            r = 10
            patch = image_rgb[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
            if patch.size > 0:
                img_mean = patch.mean(axis=(0, 1)) / 255.0
                diff = np.linalg.norm(tex_mean - img_mean)
                details['cheek_tone_diff'] = safe_float(diff)
                # Low diff = 3DDFA texture matches image = normal
                # High diff = 3DDFA texture interpolated (mask) = anomaly
                score = np.clip(diff * 3, 0, 1)
                return safe_float(score), details

        return 0.0, details


# =============================================================================
# MAIN TEST SCRIPT
# =============================================================================

def process_image(image_path, ddffa_analyzer, mp_analyzer, eye_detector, cross_detector, args, pose_estimator=None):
    """Process a single image and return all method scores."""
    print(f"  Processing: {os.path.basename(image_path)}")

    # Load image
    im_pil = Image.open(image_path).convert('RGB')
    image_rgb = np.asarray(im_pil)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w = image_rgb.shape[:2]

    result = {
        'filename': os.path.basename(image_path),
        'image_size': [w, h],
    }

    # 3DDFA-V3
    ddffa = None
    try:
        ddffa = ddffa_analyzer.analyze(im_pil)
        result['3ddfa_success'] = True
    except Exception as e:
        print(f"    3DDFA failed: {e}")
        result['3ddfa_success'] = False
        result['error'] = str(e)
        ddffa = {}

    # MediaPipe
    try:
        mp_data = mp_analyzer.analyze(image_rgb)
        result['mediapipe_success'] = mp_data['landmarks'] is not None
        result['iris_available'] = mp_data.get('iris_available', False)
    except Exception as e:
        print(f"    MediaPipe failed: {e}")
        mp_data = {'landmarks': None, 'detection_bbox': None, 'iris_available': False}
        result['mediapipe_success'] = False

    # Head Pose + Ракурс
    mp_lm = mp_data.get('landmarks')
    yaw_deg, pitch_deg, roll_deg = 0.0, 0.0, 0.0
    rakurs = "unknown"
    rakurs_side = "C"
    visible_eye = detect_visible_eye(mp_lm)

    if pose_estimator is not None:
        try:
            pose = pose_estimator.predict(image_bgr)
            if pose is not None:
                yaw_deg = pose['yaw']
                pitch_deg = pose['pitch']
                roll_deg = pose['roll']
        except Exception as e:
            print(f"    Pose estimation failed: {e}")

    rakurs, rakurs_side = classify_rakurs(yaw_deg)

    result['yaw'] = safe_float(yaw_deg)
    result['pitch'] = safe_float(pitch_deg)
    result['roll'] = safe_float(roll_deg)
    result['ракурс'] = rakurs if rakurs else "profile_skipped"
    result['ракурс_сторона'] = rakurs_side
    result['видимый_глаз'] = visible_eye

    print(f"    Pose: yaw={yaw_deg:.1f} pitch={pitch_deg:.1f} roll={roll_deg:.1f} -> {rakurs} ({rakurs_side}), eye={visible_eye}")

    # Profile — MediaPipe не работает, пропускаем анализ
    if rakurs is None:
        print(f"    PROFILE — skipping analysis (|yaw|={abs(yaw_deg):.1f}° > 65°)")
        result['eye_scores'] = {}
        result['cross_scores'] = {}
        result['combined_mean'] = 0.0
        result['combined_max'] = 0.0
        result['eye_mean'] = 0.0
        result['cross_mean'] = 0.0
        return result

    # Pose from 3DDFA
    trans_params = ddffa.get('trans_params')
    if trans_params is not None:
        result['trans_params'] = [safe_float(x) for x in trans_params[:5]]

    # Run eye methods
    try:
        eye_scores, eye_details = eye_detector.run_all(image_rgb, ddffa, mp_data)
        result['eye_scores'] = eye_scores
    except Exception as e:
        print(f"    Eye methods failed: {e}")
        result['eye_scores'] = {}

    # Run cross-system methods
    try:
        cross_scores, cross_details = cross_detector.run_all(image_rgb, ddffa, mp_data)
        result['cross_scores'] = cross_scores
    except Exception as e:
        print(f"    Cross methods failed: {e}")
        result['cross_scores'] = {}

    # Combined score
    all_scores = {}
    all_scores.update(result.get('eye_scores', {}))
    all_scores.update(result.get('cross_scores', {}))

    if all_scores:
        all_vals = list(all_scores.values())
        result['combined_mean'] = safe_float(np.mean(all_vals))
        result['combined_median'] = safe_float(np.median(all_vals))
        result['combined_max'] = safe_float(max(all_vals))
        result['eye_mean'] = safe_float(np.mean(list(result.get('eye_scores', {}).values()))) if result.get('eye_scores') else 0.0
        result['cross_mean'] = safe_float(np.mean(list(result.get('cross_scores', {}).values()))) if result.get('cross_scores') else 0.0
        result['top5_methods'] = sorted(all_scores.items(), key=lambda x: -x[1])[:5]

    return result


def main():
    parser = argparse.ArgumentParser(description='Eye Test - Mask Detection Suite')
    parser.add_argument('-i', '--inputpath', required=True, type=str,
                        help='path to folder with test images')
    parser.add_argument('-o', '--output', default='eyetest.json', type=str,
                        help='output JSON file path')
    parser.add_argument('--device', default='cuda', type=str,
                        help='cuda or cpu')
    parser.add_argument('--detector', default='retinaface', type=str)
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'])
    # 3DDFA options (all enabled)
    parser.add_argument('--ldm68', default=True, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--ldm106', default=True, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--ldm106_2d', default=True, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--ldm134', default=True, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--seg', default=True, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--seg_visible', default=True, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--useTex', default=True, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--extractTex', default=True, type=lambda x: x.lower() in ['true', '1'])

    args = parser.parse_args()

    print("=" * 60)
    print("EYE TEST — Silicon Mask Detection Suite")
    print("=" * 60)
    print(f"Input:  {args.inputpath}")
    print(f"Output: {args.output}")
    print(f"Device: {args.device}")
    print()

    # Initialize
    print("Loading 3DDFA-V3 model...")
    ddffa_analyzer = ThreeDDFAAnalyzer(args)
    print("Loading MediaPipe...")
    mp_analyzer = MediaPipeAnalyzer()

    # Detectors
    eye_detector = EyeMaskDetector()
    cross_detector = CrossSystemDetector()

    # Get image list
    im_paths = get_data_path(args.inputpath)
    print(f"\nFound {len(im_paths)} images")
    print()

    # Process
    results = []
    for i, im_path in enumerate(im_paths):
        print(f"[{i + 1}/{len(im_paths)}]")
        try:
            result = process_image(im_path, ddffa_analyzer, mp_analyzer,
                                   eye_detector, cross_detector, args)
            results.append(result)

            # Print quick summary
            if 'combined_mean' in result:
                print(f"    Combined: mean={result['combined_mean']:.3f}, "
                      f"max={result['combined_max']:.3f}, "
                      f"eye={result['eye_mean']:.3f}, cross={result['cross_mean']:.3f}")
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({
                'filename': os.path.basename(im_path),
                'error': str(e),
            })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_method_names = set()
    for r in results:
        all_method_names.update(r.get('eye_scores', {}).keys())
        all_method_names.update(r.get('cross_scores', {}).keys())

    method_stats = {}
    for method in sorted(all_method_names):
        vals = []
        for r in results:
            v = r.get('eye_scores', {}).get(method)
            if v is None:
                v = r.get('cross_scores', {}).get(method)
            if v is not None:
                vals.append(v)

        if vals:
            method_stats[method] = {
                'mean': safe_float(np.mean(vals)),
                'std': safe_float(np.std(vals)),
                'min': safe_float(np.min(vals)),
                'max': safe_float(np.max(vals)),
                'median': safe_float(np.median(vals)),
                'n_samples': len(vals),
            }

    # Print method stats sorted by mean
    print("\nMethod statistics (sorted by mean score):")
    print(f"{'Method':<35} {'Mean':>7} {'Std':>7} {'Min':>7} {'Max':>7} {'Med':>7}")
    print("-" * 75)
    for method, stats in sorted(method_stats.items(), key=lambda x: -x[1]['mean']):
        print(f"{method:<35} {stats['mean']:>7.3f} {stats['std']:>7.3f} "
              f"{stats['min']:>7.3f} {stats['max']:>7.3f} {stats['median']:>7.3f}")

    # Combined stats
    combined_means = [r.get('combined_mean', 0) for r in results if 'combined_mean' in r]
    if combined_means:
        print(f"\nCombined score: mean={np.mean(combined_means):.3f}, "
              f"std={np.std(combined_means):.3f}")

    # Save
    output_data = {
        'input_path': args.inputpath,
        'device': args.device,
        'n_images': len(results),
        'method_statistics': method_stats,
        'results': results,
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
