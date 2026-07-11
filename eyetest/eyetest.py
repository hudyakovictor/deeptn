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
from util.io import back_resize_ldms, back_resize_crop_img

# ==================== MediaPipe ====================
import mediapipe as mp

# ==================== SciPy ====================
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from scipy.stats import skew as sp_skew


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

        # === METHOD 1: Aperture Shrinkage ===
        scores['aperture_shrinkage'], details['aperture'] = self._aperture_shrinkage(
            mp_lm, ldm68_orig, ldm106_orig, h, w)

        # === METHOD 2: Iris-to-Aperture Ratio ===
        scores['iris_aperture_ratio'], details['iris_ratio'] = self._iris_aperture_ratio(
            image_rgb, mp_lm, ldm68_orig, h, w)

        # === METHOD 3: Eyelid Thickness ===
        scores['eyelid_thickness'], details['lid_thickness'] = self._eyelid_thickness(
            image_rgb, mp_lm, ldm68_orig, h, w)

        # === METHOD 4: Lid Smoothness ===
        scores['lid_smoothness'], details['lid_smooth'] = self._lid_smoothness(
            image_rgb, mp_lm, ldm68_orig, h, w)

        # === METHOD 5: Lash Invisibility ===
        scores['lash_invisibility'], details['lash'] = self._lash_invisibility(
            image_rgb, mp_lm, ldm68_orig, h, w)

        # === METHOD 6: Periocular Discontinuity ===
        scores['periocular_discontinuity'], details['periocular'] = self._periocular_discontinuity(
            image_rgb, mp_lm, ldm68_orig, h, w)

        # === METHOD 7: Eyelid Edge Sharpness ===
        scores['eyelid_edge_sharpness'], details['lid_edge'] = self._eyelid_edge_sharpness(
            image_rgb, mp_lm, ldm68_orig, h, w)

        # === METHOD 8: Eye Depth (3DDFA Z-depth analysis) ===
        scores['eye_depth_anomaly'], details['eye_depth'] = self._eye_depth_anomaly(
            ddffa, mp_lm, h, w)

        # === METHOD 9: Eye Symmetry ===
        scores['eye_symmetry_anomaly'], details['eye_sym'] = self._eye_symmetry(
            mp_lm, ldm68_orig, h, w)

        # === METHOD 10: Brow-Lid Gap ===
        scores['brow_lid_gap'], details['brow_lid'] = self._brow_lid_gap(
            mp_lm, ldm68_orig, h, w)

        # === METHOD 11: Sclera-Iris Boundary ===
        scores['sclera_iris_boundary'], details['sclera_iris'] = self._sclera_iris_boundary(
            image_rgb, mp_lm, h, w)

        # === METHOD 12: Pupil Apparent Size ===
        scores['pupil_apparent_size'], details['pupil_size'] = self._pupil_apparent_size(
            image_rgb, mp_lm, h, w)

        # === METHOD 13: Eye Orbit Area (segmentation) ===
        scores['orbit_area_ratio'], details['orbit_area'] = self._orbit_area_ratio(
            seg_vis, mp_lm, ddffa, h, w, trans_params)

        # === METHOD 14: Periocular Texture LBP ===
        scores['periocular_lbp_entropy'], details['peri_lbp'] = self._periocular_lbp(
            image_rgb, mp_lm, ldm68_orig, h, w)

        # === METHOD 15: Subsurface Scattering (R-B ratio on eyelids) ===
        scores['eyelid_sss'], details['sss'] = self._eyelid_sss(
            image_rgb, mp_lm, ldm68_orig, h, w)

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

    def _iris_aperture_ratio(self, image_rgb, mp_lm, ldm68_orig, h, w):
        """
        METHOD 2: Iris-to-aperture ratio.
        Mask makes iris look bigger relative to visible eye.
        """
        iod = self._get_iod(mp_lm, ldm68_orig)
        details = {}
        ratios = []

        for side, iris_idx, eye_idx in [
            ('left', MP_LEFT_IRIS, MP_LEFT_EYE_OUTLINE),
            ('right', MP_RIGHT_IRIS, MP_RIGHT_EYE_OUTLINE)
        ]:
            if mp_lm is None or not mp_lm.shape[0] > max(iris_idx + eye_idx):
                continue

            # Iris diameter
            iris_center = mp_lm[iris_idx[0], :2]
            iris_edges = mp_lm[iris_idx[1:], :2]
            iris_radii = [np.linalg.norm(e - iris_center) for e in iris_edges]
            iris_diam = 2 * np.mean(iris_radii)

            # Eye height (aperture)
            if side == 'left':
                eye_h = np.linalg.norm(mp_lm[159, :2] - mp_lm[145, :2])
            else:
                eye_h = np.linalg.norm(mp_lm[386, :2] - mp_lm[374, :2])

            if eye_h < 1:
                continue

            ratio = iris_diam / eye_h
            ratios.append(ratio)
            details[f'iris_ap_{side}'] = safe_float(ratio)
            details[f'iris_diam_{side}_px'] = safe_float(iris_diam)
            details[f'eye_height_{side}_px'] = safe_float(eye_h)

        if not ratios:
            return 0.0, details

        mean_ratio = np.mean(ratios)
        details['mean_iris_ap_ratio'] = safe_float(mean_ratio)

        # Normal: 0.5-0.7, Mask: > 0.85
        score = np.clip((mean_ratio - 0.65) / 0.25, 0, 1)
        return safe_float(score), details

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

        # Normal: 3-5 px, Mask: 7-12 px
        score = np.clip((mean_thick - 5) / 7, 0, 1)
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

    def _lash_invisibility(self, image_rgb, mp_lm, ldm68_orig, h, w):
        """
        METHOD 5: Lash invisibility.
        High-frequency energy along upper eyelid edge.
        """
        details = {}
        energies = []

        for side in ['left', 'right']:
            crop, bbox = self._get_eye_crop(image_rgb, mp_lm, ldm68_orig, side, expand=1.0)
            if crop is None:
                continue

            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            ch = gray.shape[0]

            # Upper part (lashes area) - use thin strip at top of eye
            upper = gray[0:max(ch // 6, 2), :]
            if upper.size < 16:
                continue

            # Gabor filter for fine line detection (lashes)
            gabor = cv2.getGaborKernel((11, 11), 3.0, np.pi / 2, 8.0, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(upper.astype(np.float32), cv2.CV_32F, gabor)
            energy = float(np.mean(np.abs(filtered)))
            energies.append(energy)
            details[f'lash_energy_{side}'] = safe_float(energy)

        if not energies:
            return 0.0, details

        mean_energy = np.mean(energies)
        details['mean_lash_energy'] = safe_float(mean_energy)

        # Normal lashes: energy > 100 (visible fine lines), Mask: < 30 (no lashes visible)
        score = np.clip((100 - mean_energy) / 80, 0, 1)
        return safe_float(score), details

    def _periocular_discontinuity(self, image_rgb, mp_lm, ldm68_orig, h, w):
        """
        METHOD 6: Periocular texture discontinuity.
        Compare LBP histogram inside orbit vs outside.
        """
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        details = {}
        diffs = []

        for side in ['left', 'right']:
            if mp_lm is None:
                continue

            if side == 'left':
                eye_center = mp_lm[MP_LEFT_EYE_OUTLINE, :2].mean(axis=0).astype(int)
            else:
                eye_center = mp_lm[MP_RIGHT_EYE_OUTLINE, :2].mean(axis=0).astype(int)

            iod = self._get_iod(mp_lm, ldm68_orig)
            r_inner = int(iod * 0.15)
            r_outer = int(iod * 0.45)

            # Inner ring (close to eye = potential mask edge)
            inner = self._ring_patch(gray, eye_center, r_inner, r_inner + int(iod * 0.1), h, w)
            # Outer ring (further from eye = real skin)
            outer = self._ring_patch(gray, eye_center, r_outer, r_outer + int(iod * 0.1), h, w)

            if inner is not None and outer is not None and inner.size > 16 and outer.size > 16:
                hist_inner = self._quick_lbp_hist(inner)
                hist_outer = self._quick_lbp_hist(outer)
                diff = np.sum((hist_inner - hist_outer) ** 2 / (hist_inner + hist_outer + 1e-6))
                diffs.append(diff)
                details[f'peri_disc_{side}'] = safe_float(diff)

        if not diffs:
            return 0.0, details

        mean_diff = np.mean(diffs)
        details['mean_peri_discontinuity'] = safe_float(mean_diff)

        # Normal: < 0.5, Mask: > 1.0 (texture jump at mask edge)
        score = np.clip((mean_diff - 0.5) / 0.8, 0, 1)
        return safe_float(score), details

    def _ring_patch(self, gray, center, r_inner, r_outer, h, w):
        """Extract ring-shaped patch."""
        cx, cy = center
        size = r_outer * 2 + 1
        y1 = max(0, cy - r_outer)
        y2 = min(h, cy + r_outer)
        x1 = max(0, cx - r_outer)
        x2 = min(w, cx + r_outer)
        if y2 <= y1 or x2 <= x1:
            return None

        patch = gray[y1:y2, x1:x2]
        # Create ring mask
        yy, xx = np.mgrid[y1:y2, x1:x2]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        ring = (dist >= r_inner) & (dist <= r_outer)

        if ring.sum() < 10:
            return None
        return patch[ring]

    def _quick_lbp_hist(self, data):
        """Quick LBP histogram (8 bins)."""
        if data.ndim == 1:
            # Already a flat array of pixel values
            hist, _ = np.histogram(data, bins=8, range=(0, 256))
        else:
            # Compute simple LBP
            h, w = data.shape[:2]
            if h < 3 or w < 3:
                return np.ones(8) / 8
            center = data[1:-1, 1:-1].astype(np.uint8)
            lbp = np.zeros_like(center)
            offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
            for i, (dy, dx) in enumerate(offsets):
                neighbor = data[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx]
                lbp |= ((neighbor >= center).astype(np.uint8) << i)
            hist, _ = np.histogram(lbp.ravel(), bins=8, range=(0, 256))

        hist = hist.astype(np.float32)
        hist /= hist.sum() + 1e-6
        return hist

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

    def _eye_depth_anomaly(self, ddffa, mp_lm, h, w):
        """
        METHOD 8: Eye depth from 3DDFA Z coordinates.
        Mask pushes eyes deeper (several mm behind mask surface).
        """
        v3d = ddffa.get('v3d')
        ldm68 = ddffa.get('ldm68')
        details = {}

        if v3d is None or ldm68 is None:
            return 0.0, details

        # v3d shape: (35709, 3) - 3D vertices in camera space
        # Z = depth (higher = further from camera)
        # ldm68 are indices into v3d... actually ldm68 here is 2D projected

        # Use face_model ldm68 vertex indices if available
        # For now: compare Z of eye region vs nose/forehead
        # We need the vertex indices for landmarks

        # Approximate: use v2d to find vertices near eye positions
        v2d = ddffa.get('v2d')
        if v2d is None:
            return 0.0, details

        # v2d: (35709, 2) in 224x224 space
        # Find vertices near eye landmarks
        if ldm68 is not None:
            # Left eye center in 224 space
            le_center = ldm68[36:42].mean(axis=0)  # (x, y) in 224 space
            re_center = ldm68[42:48].mean(axis=0)
            nose_tip = ldm68[30] if len(ldm68) > 30 else ldm68[27]
            forehead = ldm68[27] if len(ldm68) > 27 else ldm68[19]

            # Find nearest vertices
            tree = cKDTree(v2d[:, :2])

            _, le_idx = tree.query(le_center)
            _, re_idx = tree.query(re_center)
            _, nose_idx = tree.query(nose_tip[:2])
            _, fh_idx = tree.query(forehead[:2])

            z_eye_L = v3d[le_idx, 2]
            z_eye_R = v3d[re_idx, 2]
            z_nose = v3d[nose_idx, 2]
            z_forehead = v3d[fh_idx, 2]

            # Depth differences (Z in 3DDFA: smaller = further from camera)
            depth_eye_nose_L = z_eye_L - z_nose  # positive = eye behind nose
            depth_eye_nose_R = z_eye_R - z_nose
            depth_eye_fh_L = z_eye_L - z_forehead
            depth_eye_fh_R = z_eye_R - z_forehead

            details['z_eye_L'] = safe_float(z_eye_L)
            details['z_eye_R'] = safe_float(z_eye_R)
            details['z_nose'] = safe_float(z_nose)
            details['depth_eye_nose_L'] = safe_float(depth_eye_nose_L)
            details['depth_eye_nose_R'] = safe_float(depth_eye_nose_R)
            details['depth_eye_fh_L'] = safe_float(depth_eye_fh_L)
            details['depth_eye_fh_R'] = safe_float(depth_eye_fh_R)

            # Average depth of eye relative to nose and forehead
            avg_depth = np.mean([depth_eye_nose_L, depth_eye_nose_R,
                                 depth_eye_fh_L, depth_eye_fh_R])
            details['avg_eye_depth_offset'] = safe_float(avg_depth)

            # In 3DDFA: Z ~8.0-8.5, eye is typically slightly closer than nose (negative diff)
            # Normal: eye-nose diff around -0.1 to -0.3 (eye closer to camera)
            # Mask: eye much deeper relative to nose (less negative or positive)
            # We use negative of avg_depth since it's typically negative
            score = np.clip((-avg_depth - 0.15) / 0.15, 0, 1)
            return safe_float(score), details

        return 0.0, details

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

    def _brow_lid_gap(self, mp_lm, ldm68_orig, h, w):
        """
        METHOD 10: Brow-lid depth gap.
        From CSV data: PUT=0.0085, UDMURT=0.0129 (+52%).
        Mask deepens the gap between brow and eyelid.
        """
        iod = self._get_iod(mp_lm, ldm68_orig)
        details = {}
        gaps = []

        if mp_lm is not None and len(mp_lm) > 386:
            for side, brow_idx, lid_idx in [
                ('left', 105, 159),   # brow point, upper lid
                ('right', 334, 386),
            ]:
                if brow_idx < len(mp_lm) and lid_idx < len(mp_lm):
                    brow_y = mp_lm[brow_idx, 1]
                    lid_y = mp_lm[lid_idx, 1]
                    gap = abs(lid_y - brow_y) / max(iod, 1)
                    gaps.append(gap)
                    details[f'brow_lid_gap_{side}'] = safe_float(gap)

        elif ldm68_orig is not None and len(ldm68_orig) > 43:
            for side, brow_idx, lid_idx in [
                ('left', 19, 37),
                ('right', 24, 43),
            ]:
                brow_y = ldm68_orig[brow_idx, 1]
                lid_y = ldm68_orig[lid_idx, 1]
                gap = abs(lid_y - brow_y) / max(iod, 1)
                gaps.append(gap)
                details[f'brow_lid_gap_{side}'] = safe_float(gap)

        if not gaps:
            return 0.0, details

        mean_gap = np.mean(gaps)
        details['mean_brow_lid_gap'] = safe_float(mean_gap)

        # Normal: 0.06-0.10, Mask: > 0.12
        score = np.clip((mean_gap - 0.10) / 0.05, 0, 1)
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

        # Normal: ~25-28, Mask: > 32 (sharp shadow from mask edge)
        score = np.clip((mean_sharp - 28) / 5, 0, 1)
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

        # Normal skin: entropy ~6.4-6.6, Silicone: slightly lower ~6.3-6.5
        score = np.clip((6.4 - mean_ent) / 0.5, 0, 1)
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

        # Normal: R-B ~30-40, Silicone: > 45 (more red)
        score = np.clip((mean_rb - 45) / 15, 0, 1)
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

        # === METHOD 16: Landmark Discrepancy ===
        scores['landmark_discrepancy'], details['lm_disc'] = self._landmark_discrepancy(
            mp_lm, ldm68_orig, h, w)

        # === METHOD 17: Specular BRDF ===
        scores['specular_brdf'], details['specular'] = self._specular_analysis(image_rgb, mp_lm, h, w)

        # === METHOD 18: Eye contour divergence ===
        scores['eye_contour_divergence'], details['contour'] = self._eye_contour_divergence(
            mp_lm, ldm68_orig, h, w)

        # === METHOD 19: BBox expansion ===
        scores['bbox_expansion'], details['bbox'] = self._bbox_expansion(
            ddffa, mp_data, h, w)

        # === METHOD 20: Subsurface Light Transport ===
        scores['subsurface_violation'], details['sss_full'] = self._subsurface_analysis(
            image_rgb, mp_lm, h, w)

        # === METHOD 21: Nasolabial fold suppression ===
        scores['nasolabial_suppression'], details['naso'] = self._nasolabial(
            image_rgb, mp_lm, h, w)

        # === METHOD 22: Ear texture cliff ===
        scores['ear_texture_cliff'], details['ear'] = self._ear_cliff(
            image_rgb, mp_lm, h, w)

        # === METHOD 23: Gaze inconsistency ===
        scores['gaze_inconsistency'], details['gaze'] = self._gaze_inconsistency(
            mp_lm, ldm68_orig, h, w)

        # === METHOD 24: Face mesh curvature ===
        scores['mesh_curvature_anomaly'], details['curvature'] = self._mesh_curvature(ddffa, mp_lm, h, w)

        # === METHOD 25: Skin tone UV mismatch ===
        scores['skin_tone_mismatch'], details['tone'] = self._skin_tone(
            image_rgb, ddffa, mp_lm, h, w)

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

    def _bbox_expansion(self, ddffa, mp_data, h, w):
        """
        METHOD 19: BBox expansion (mask widens face).
        """
        details = {}
        mp_bbox = mp_data.get('detection_bbox')
        trans_params = ddffa.get('trans_params')

        if mp_bbox is None or trans_params is None:
            return 0.0, details

        # 3DDFA face size from trans_params (w0, h0, s, t)
        w0, h0 = trans_params[0], trans_params[1]

        # MP bbox in original image
        mp_w = mp_bbox[2]
        mp_h = mp_bbox[3]

        details['mp_bbox_w'] = safe_float(mp_w)
        details['mp_bbox_h'] = safe_float(mp_h)
        details['3ddfa_face_w'] = safe_float(w0)

        # Width ratio
        ratio = w0 / max(mp_w, 1)
        details['bbox_width_ratio'] = safe_float(ratio)

        # Normal: ~1.0, Mask: > 1.08
        score = np.clip((ratio - 1.0) / 0.10, 0, 1)
        return safe_float(score), details

    def _subsurface_analysis(self, image_rgb, mp_lm, h, w):
        """
        METHOD 20: Subsurface light transport violation.
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

    def _nasolabial(self, image_rgb, mp_lm, h, w):
        """
        METHOD 21: Nasolabial fold suppression.
        """
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        details = {}

        if mp_lm is None:
            return 0.0, details

        contrasts = []
        for fold_idx, cheek_idx in [(165, 117), (394, 345)]:
            if fold_idx >= len(mp_lm) or cheek_idx >= len(mp_lm):
                continue
            fx, fy = int(mp_lm[fold_idx, 0]), int(mp_lm[fold_idx, 1])
            cx, cy = int(mp_lm[cheek_idx, 0]), int(mp_lm[cheek_idx, 1])

            # Compare brightness at fold vs cheek
            r = 5
            fold_patch = gray[max(0, fy - r):fy + r, max(0, fx - r):fx + r]
            cheek_patch = gray[max(0, cy - r):cy + r, max(0, cx - r):cx + r]

            if fold_patch.size > 0 and cheek_patch.size > 0:
                contrast = abs(fold_patch.mean() - cheek_patch.mean())
                contrasts.append(contrast)

        if contrasts:
            mean_c = np.mean(contrasts)
            details['nasolabial_contrast'] = safe_float(mean_c)
            # Normal: > 8, Mask: < 3
            score = np.clip((8 - mean_c) / 6, 0, 1)
            return safe_float(score), details

        return 0.0, details

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
            score = np.clip(mean_cliff / 500, 0, 1)
            return safe_float(score), details

        return 0.0, details

    def _gaze_inconsistency(self, mp_lm, ldm68_orig, h, w):
        """
        METHOD 23: Gaze direction inconsistency.
        """
        details = {}
        if mp_lm is None or not mp_lm.shape[0] > 477:
            return 0.0, details

        # MP gaze: iris center relative to eye center
        gaze_angles = []
        for side, iris_c, inner, outer in [
            ('left', 468, 133, 33),
            ('right', 473, 362, 263),
        ]:
            if max(iris_c, inner, outer) >= len(mp_lm):
                continue

            iris_pt = mp_lm[iris_c, :2]
            inner_pt = mp_lm[inner, :2]
            outer_pt = mp_lm[outer, :2]
            eye_center = (inner_pt + outer_pt) / 2
            eye_width = np.linalg.norm(outer_pt - inner_pt)

            offset = (iris_pt - eye_center) / max(eye_width, 1)
            angle = np.degrees(np.arctan2(offset[1], offset[0]))
            gaze_angles.append(angle)
            details[f'gaze_angle_{side}'] = safe_float(angle)

        if len(gaze_angles) == 2:
            # Both eyes should have similar gaze
            diff = abs(gaze_angles[0] - gaze_angles[1])
            details['gaze_binocular_diff'] = safe_float(diff)
            # Normal: < 10°, Mask: > 20°
            score = np.clip((diff - 10) / 15, 0, 1)
            return safe_float(score), details

        return 0.0, details

    def _mesh_curvature(self, ddffa, mp_lm, h, w):
        """
        METHOD 24: Dense mesh curvature anomaly.
        """
        details = {}
        v3d = ddffa.get('v3d')
        tri = ddffa.get('tri')

        if v3d is None or tri is None:
            return 0.0, details

        # Compute per-vertex curvature (simplified: Laplacian magnitude)
        n_verts = len(v3d)
        if n_verts < 100:
            return 0.0, details

        # Sample subset for speed
        sample_size = min(5000, n_verts)
        sample_idx = np.random.choice(n_verts, sample_size, replace=False)

        curvatures = []
        for v_idx in sample_idx:
            # Find neighbors
            tri_mask = (tri == v_idx).any(axis=1)
            neighbor_tris = tri[tri_mask]
            neighbors = set(neighbor_tris.ravel()) - {v_idx}

            if len(neighbors) > 0:
                nb_pts = v3d[list(neighbors)[:8]]  # max 8 neighbors
                laplacian = nb_pts.mean(axis=0) - v3d[v_idx]
                curv = np.linalg.norm(laplacian)
                curvatures.append(curv)

        if curvatures:
            curv_arr = np.array(curvatures)
            details['mesh_curvature_mean'] = safe_float(curv_arr.mean())
            details['mesh_curvature_std'] = safe_float(curv_arr.std())

            # Low std = smooth = mask
            std_score = np.clip(1.0 - curv_arr.std() / 0.05, 0, 1)
            return safe_float(std_score), details

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

def process_image(image_path, ddffa_analyzer, mp_analyzer, eye_detector, cross_detector, args):
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
    try:
        ddffa = ddffa_analyzer.analyze(im_pil)
        result['3ddfa_success'] = True
    except Exception as e:
        print(f"    3DDFA failed: {e}")
        result['3ddfa_success'] = False
        result['error'] = str(e)
        return result

    # MediaPipe
    try:
        mp_data = mp_analyzer.analyze(image_rgb)
        result['mediapipe_success'] = mp_data['landmarks'] is not None
        result['iris_available'] = mp_data.get('iris_available', False)
    except Exception as e:
        print(f"    MediaPipe failed: {e}")
        mp_data = {'landmarks': None, 'detection_bbox': None, 'iris_available': False}
        result['mediapipe_success'] = False

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
        result['combined_mean'] = safe_float(np.mean(list(all_scores.values())))
        result['combined_max'] = safe_float(max(all_scores.values()))
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
