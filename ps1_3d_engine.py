import argparse
import importlib.abc
import importlib.machinery
import os
import sys
import types
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from scipy.spatial import Delaunay


# 4x4 Hardware PS1 Ordered Dithering Matrix (centered in [-0.5, +0.4375])
PS1_BAYER_4X4 = (
    np.array(
        [
            [0, 8, 2, 10],
            [12, 4, 14, 6],
            [3, 11, 1, 9],
            [15, 7, 13, 5],
        ],
        dtype=np.float32,
    )
    / 16.0
    - 0.5
)


@dataclass
class PS1Config:
    """Configuration for authentic PS1 low-poly 3D subject reconstruction & 2D pre-rendered backdrop."""

    scene_mode: str = "isolated_subject"  # "isolated_subject" (default: 3D subject over flat 2D PS1 backdrop) or "full_scene"
    poly_count: int = 440                 # Target triangle count for the 3D subject mesh
    native_short_side: int = 240          # Authentic PS1 240p resolution along the shorter image dimension
    upscale_factor: int = 3               # Integer nearest-neighbor upscale (e.g. 240p * 3 = 720p)
    tex_resolution: int = 96              # PS1 VRAM body/subject texture page resolution (short side)
    head_tex_resolution: int = 72         # Dedicated PS1 VRAM texture page resolution per detected head/face
    clut_colors: int = 22                 # Colors in foreground subject PS1 CLUT palette (16..32)
    head_clut_colors: int = 24            # Dedicated colors in head/face PS1 CLUT palette
    bg_clut_colors: int = 16              # Colors in static 2D background backdrop CLUT palette
    facet_flatness: float = 0.45          # Body/clothing flatness blend (face prioritizes crisp CLUT texture)
    face_preservation: float = 0.96       # [0.0..1.0] Preserves crisp affine CLUT texture & gentle shading on faces
    face_flatness_max: float = 0.04       # Max flat-color blend on face/head so eyes/lips/stubble stay readable
    lighting_strength: float = 0.52       # 3D facet normal lighting contrast on the subject mesh
    crease_strength: float = 0.18         # Darkening along sharp 3D dihedral polygon folds on clothing/body
    silhouette_simplify: float = 0.009    # Polygonal silhouette coarseness (higher = chunkier low-poly outline)
    camera_yaw_deg: float = 1.2           # Subtle 3D perspective yaw on the subject (tamed to prevent shearing)
    depth_scale: float = 28.0             # Tamed 3D Z relief extrusion scale in native 240p space
    vertex_snap: float = 1.0              # PS1 GTE integer screen-space grid snap (in 240p pixels)
    dither_strength: float = 11.0         # Hardware 4x4 Bayer dithering spread before RGB555
    fog_strength: float = 0.28            # Atmospheric PS1 depth fog on the static 2D backdrop
    scanlines: bool = True                # Subtle CRT scanline pass aligned to native 240p rows
    seed: int = 42                        # Deterministic seed for reproducible meshes


PRESETS: Dict[str, PS1Config] = {
    "mgs1": PS1Config(
        scene_mode="isolated_subject",
        poly_count=420,
        native_short_side=240,
        tex_resolution=92,
        head_tex_resolution=70,
        clut_colors=22,
        head_clut_colors=24,
        bg_clut_colors=16,
        facet_flatness=0.48,
        face_preservation=0.96,
        face_flatness_max=0.04,
        lighting_strength=0.56,
        crease_strength=0.20,
        silhouette_simplify=0.0105,
        camera_yaw_deg=1.2,
        depth_scale=28.0,
        dither_strength=11.0,
        fog_strength=0.28,
        scanlines=True,
    ),
    "balanced": PS1Config(
        scene_mode="isolated_subject",
        poly_count=460,
        native_short_side=240,
        tex_resolution=104,
        head_tex_resolution=72,
        clut_colors=24,
        head_clut_colors=24,
        bg_clut_colors=16,
        facet_flatness=0.42,
        face_preservation=0.96,
        face_flatness_max=0.04,
        lighting_strength=0.50,
        crease_strength=0.18,
        silhouette_simplify=0.0085,
        camera_yaw_deg=1.2,
        depth_scale=28.0,
        dither_strength=10.5,
        fog_strength=0.26,
        scanlines=True,
    ),
    "ultra_lowpoly": PS1Config(
        scene_mode="isolated_subject",
        poly_count=260,
        native_short_side=216,
        tex_resolution=68,
        head_tex_resolution=58,
        clut_colors=16,
        head_clut_colors=20,
        bg_clut_colors=14,
        facet_flatness=0.56,
        face_preservation=0.94,
        face_flatness_max=0.06,
        lighting_strength=0.58,
        crease_strength=0.22,
        silhouette_simplify=0.013,
        camera_yaw_deg=1.4,
        depth_scale=30.0,
        vertex_snap=1.0,
        dither_strength=12.0,
        fog_strength=0.32,
        scanlines=True,
    ),
    "detailed": PS1Config(
        scene_mode="isolated_subject",
        poly_count=780,
        native_short_side=256,
        tex_resolution=128,
        head_tex_resolution=88,
        clut_colors=32,
        head_clut_colors=28,
        bg_clut_colors=20,
        facet_flatness=0.28,
        face_preservation=0.98,
        face_flatness_max=0.02,
        lighting_strength=0.42,
        crease_strength=0.12,
        silhouette_simplify=0.006,
        camera_yaw_deg=1.0,
        depth_scale=24.0,
        dither_strength=9.5,
        fog_strength=0.20,
        scanlines=False,
    ),
}


@dataclass
class FaceRegion:
    """Detected face box & anatomical feature anchors in native 240p coordinates."""

    box: Tuple[int, int, int, int]       # (fx, fy, fw, fh) in native coordinates
    head_box: Tuple[int, int, int, int]  # Expanded head+hair+chin box (hx1, hy1, hx2, hy2)
    landmarks: np.ndarray                # Structured (K, 2) anatomical PS1 head cage vertices
    eye_mouth_boxes: List[Tuple[int, int, int, int]]  # Protected zones (no random clutter vertices inside)


def _install_timm_fallback_if_missing() -> None:
    """Allows MiDaS_small (which uses EfficientNet-Lite3, not timm) to load even if timm isn't installed."""
    try:
        import timm  # noqa: F401
    except ImportError:
        class _TimmStub(types.ModuleType):
            __path__ = []

            def __getattr__(self, key: str):
                if key.startswith("__"):
                    raise AttributeError(key)
                return _TimmStub(f"{self.__name__}.{key}")

            def __call__(self, *args, **kwargs):
                return _TimmStub("call")

        class _TimmFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
            def find_spec(self, fullname: str, path, target=None):
                if fullname == "timm" or fullname.startswith("timm."):
                    return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
                return None

            def create_module(self, spec):
                return _TimmStub(spec.name)

            def exec_module(self, module):
                pass

        sys.meta_path.append(_TimmFinder())


def ensure_rgb_uint8(img: np.ndarray) -> np.ndarray:
    """Normalizes arbitrary grayscale/RGBA/uint16/float images into standard HxWx3 uint8 RGB."""
    if img is None:
        raise ValueError("Input image is None.")
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
        rgb = arr[:, :, :3].astype(np.float32)
        bg = np.full_like(rgb, 32.0)
        arr = (rgb * alpha + bg * (1.0 - alpha)).astype(np.uint8)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)

    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            max_v = float(np.nanmax(arr)) if arr.size > 0 else 1.0
            if max_v <= 1.05:
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


class PS13DEngine:
    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        _install_timm_fallback_if_missing()
        self.midas, self.transform = self._load_midas_model()

        # Built-in OpenCV Haar cascades for fast offline face & eye landmark detection
        haar_dir = cv2.data.haarcascades
        self.face_cascade_alt2 = cv2.CascadeClassifier(os.path.join(haar_dir, "haarcascade_frontalface_alt2.xml"))
        self.face_cascade_def = cv2.CascadeClassifier(os.path.join(haar_dir, "haarcascade_frontalface_default.xml"))
        self.profile_cascade = cv2.CascadeClassifier(os.path.join(haar_dir, "haarcascade_profileface.xml"))
        self.eye_cascade = cv2.CascadeClassifier(os.path.join(haar_dir, "haarcascade_eye.xml"))

    def _load_midas_model(self):
        hub_dir = torch.hub.get_dir()
        midas_local = os.path.join(hub_dir, "intel-isl_MiDaS_master")
        effnet_local = os.path.join(hub_dir, "rwightman_gen-efficientnet-pytorch_master")

        orig_load = torch.hub.load

        def _patched_hub_load(repo, model, *args, **kwargs):
            if "gen-efficientnet-pytorch" in str(repo) and os.path.isdir(effnet_local):
                return orig_load(effnet_local, model, *args, source="local", trust_repo=True, **kwargs)
            return orig_load(repo, model, *args, **kwargs)

        if os.path.isdir(midas_local):
            print("Loading cached MiDaS_small depth model...")
            torch.hub.load = _patched_hub_load
            try:
                midas = orig_load(midas_local, "MiDaS_small", source="local", trust_repo=True).to(self.device).eval()
                midas_transforms = orig_load(midas_local, "transforms", source="local", trust_repo=True)
            finally:
                torch.hub.load = orig_load
        else:
            print("Downloading and loading MiDaS_small depth model (~40MB)...")
            midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).to(self.device).eval()
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)

        return midas, midas_transforms.small_transform

    def compute_depth(self, img_rgb: np.ndarray) -> np.ndarray:
        """Returns normalized inverse depth in [0, 1] (1.0 = closest foreground, 0.0 = farthest background)."""
        input_batch = self.transform(img_rgb).to(self.device)
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth = prediction.cpu().numpy().astype(np.float32)
        p_low, p_high = np.percentile(depth, [1.0, 99.5])
        depth = np.clip((depth - p_low) / (p_high - p_low + 1e-6), 0.0, 1.0).astype(np.float32)
        return depth

    def detect_faces_and_landmarks(
        self, orig_rgb: np.ndarray, full_depth: np.ndarray, ps1_w: int, ps1_h: int
    ) -> List[FaceRegion]:
        """
        Detects faces on the full-resolution image, validates against foreground depth, refines eye/brow/mouth
        heights, and constructs a clean, symmetrical PS1 anatomical head cage in native 240p coordinates.
        """
        orig_h, orig_w = orig_rgb.shape[:2]
        det_scale = min(1.0, 640.0 / float(min(orig_h, orig_w)))
        det_w = max(64, int(round(orig_w * det_scale)))
        det_h = max(64, int(round(orig_h * det_scale)))
        det_gray = cv2.cvtColor(cv2.resize(orig_rgb, (det_w, det_h), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        det_gray_eq = cv2.equalizeHist(det_gray)

        min_face = max(20, int(round(min(det_w, det_h) * 0.05)))
        raw_boxes = []
        for gray_src in (det_gray, det_gray_eq):
            for cascade, scale_f, min_n in (
                (self.face_cascade_alt2, 1.06, 3),
                (self.face_cascade_def, 1.08, 4),
                (self.profile_cascade, 1.08, 4),
            ):
                if cascade.empty():
                    continue
                found = cascade.detectMultiScale(
                    gray_src, scaleFactor=scale_f, minNeighbors=min_n, minSize=(min_face, min_face)
                )
                if len(found) > 0:
                    raw_boxes.extend(found.tolist())

        if not raw_boxes:
            return []

        raw_boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
        merged_boxes = []
        for bx, by, bw, bh in raw_boxes:
            ox1 = int(np.clip(bx / det_scale, 0, orig_w - 1))
            oy1 = int(np.clip(by / det_scale, 0, orig_h - 1))
            ox2 = int(np.clip((bx + bw) / det_scale, ox1 + 1, orig_w))
            oy2 = int(np.clip((by + bh) / det_scale, oy1 + 1, orig_h))
            patch_depth = float(np.median(full_depth[oy1:oy2, ox1:ox2]))
            if patch_depth < 0.35:
                continue

            overlap = False
            for mx, my, mw, mh in merged_boxes:
                ix1, iy1 = max(bx, mx), max(by, my)
                ix2, iy2 = min(bx + bw, mx + mw), min(by + bh, my + mh)
                if ix2 > ix1 and iy2 > iy1:
                    inter = float((ix2 - ix1) * (iy2 - iy1))
                    if inter / float(min(bw * bh, mw * mh)) > 0.35:
                        overlap = True
                        break
            if not overlap:
                merged_boxes.append((bx, by, bw, bh))

        sx = ps1_w / float(det_w)
        sy = ps1_h / float(det_h)
        faces: List[FaceRegion] = []

        filtered_boxes = []
        for bx, by, bw, bh in merged_boxes:
            roi_h = int(round(bh * 0.62))
            face_roi = det_gray_eq[by : by + roi_h, bx : bx + bw]
            min_eye = max(6, int(round(bw * 0.13)))
            max_eye = int(round(bw * 0.42))
            eyes = (
                self.eye_cascade.detectMultiScale(
                    face_roi, scaleFactor=1.06, minNeighbors=3, minSize=(min_eye, min_eye), maxSize=(max_eye, max_eye)
                )
                if not self.eye_cascade.empty() and min_eye < max_eye
                else ()
            )
            # If box is in lower half of image (e.g. knees/hips) and has 0 eyes, reject false positive!
            if len(eyes) == 0 and by > (det_h * 0.45):
                continue
            filtered_boxes.append((bx, by, bw, bh))
        merged_boxes = filtered_boxes if filtered_boxes else merged_boxes[:1]

        # Reject vertically stacked false positive boxes located in torso/chest beneath a higher face
        final_boxes = []
        for i, b1 in enumerate(merged_boxes):
            bx1, by1, bw1, bh1 = b1
            is_torso_duplicate = False
            for j, b2 in enumerate(merged_boxes):
                if i != j:
                    bx2, by2, bw2, bh2 = b2
                    # If b2 is above b1 and horizontally aligned (same body column)
                    if by2 < by1 and (by1 > by2 + bh2 * 0.70):
                        overlap_x = max(0, min(bx1 + bw1, bx2 + bw2) - max(bx1, bx2))
                        if overlap_x > min(bw1, bw2) * 0.50:
                            is_torso_duplicate = True
                            break
            if not is_torso_duplicate:
                final_boxes.append(b1)
        merged_boxes = final_boxes if final_boxes else merged_boxes[:1]

        for bx, by, bw, bh in merged_boxes:
            roi_h = int(round(bh * 0.62))
            face_roi = det_gray_eq[by : by + roi_h, bx : bx + bw]
            min_eye = max(6, int(round(bw * 0.13)))
            max_eye = int(round(bw * 0.42))
            eyes = (
                self.eye_cascade.detectMultiScale(
                    face_roi, scaleFactor=1.06, minNeighbors=3, minSize=(min_eye, min_eye), maxSize=(max_eye, max_eye)
                )
                if not self.eye_cascade.empty() and min_eye < max_eye
                else ()
            )

            fx = float(bx) * sx
            fy = float(by) * sy
            fw = float(bw) * sx
            fh = float(bh) * sy

            eye_v = 0.37
            center_u = 0.50
            left_eye_u, right_eye_u = 0.31, 0.69

            valid_eyes = []
            for ex, ey, ew, eh in eyes:
                ecx = (ex + ew * 0.5) / float(bw)
                ecy = (ey + eh * 0.5) / float(bh)
                if 0.15 <= ecx <= 0.85 and 0.20 <= ecy <= 0.54:
                    valid_eyes.append((ecx, ecy, ew / float(bw)))

            left_cands = [e for e in valid_eyes if e[0] < 0.48]
            right_cands = [e for e in valid_eyes if e[0] > 0.52]
            if left_cands and right_cands:
                le = max(left_cands, key=lambda e: e[2])
                re = max(right_cands, key=lambda e: e[2])
                left_eye_u, right_eye_u = le[0], re[0]
                center_u = 0.5 * (left_eye_u + right_eye_u)
                eye_v = 0.5 * (le[1] + re[1])
            elif left_cands:
                le = max(left_cands, key=lambda e: e[2])
                left_eye_u = le[0]
                eye_v = le[1]
            elif right_cands:
                re = max(right_cands, key=lambda e: e[2])
                right_eye_u = re[0]
                eye_v = re[1]

            mouth_v = 0.78
            m_y1 = int(np.clip(by + bh * 0.68, 0, det_h - 2))
            m_y2 = int(np.clip(by + bh * 0.90, m_y1 + 2, det_h))
            m_x1 = int(np.clip(bx + bw * 0.30, 0, det_w - 2))
            m_x2 = int(np.clip(bx + bw * 0.70, m_x1 + 2, det_w))
            mouth_strip = det_gray[m_y1:m_y2, m_x1:m_x2]
            if mouth_strip.size > 16:
                horiz_edges = np.mean(np.abs(cv2.Sobel(mouth_strip, cv2.CV_32F, 0, 1, ksize=3)), axis=1)
                peak_row = int(np.argmax(horiz_edges))
                mouth_v = float(np.clip(((m_y1 + peak_row) - by) / float(bh), 0.72, 0.85))

            brow_top_v = max(0.14, eye_v - 0.16)
            eye_bot_v = min(0.54, eye_v + 0.11)
            nose_tip_v = 0.5 * (eye_bot_v + mouth_v) + 0.02
            mouth_top_v = max(nose_tip_v + 0.05, mouth_v - 0.065)
            mouth_bot_v = min(0.93, mouth_v + 0.075)

            cu = center_u
            lu_in = 0.5 * (left_eye_u + cu) + 0.02
            ru_in = 0.5 * (right_eye_u + cu) - 0.02
            lu_out = max(0.08, left_eye_u - 0.14)
            ru_out = min(0.92, right_eye_u + 0.14)

            uv_cage = [
                # Crown / Hairline arch
                (0.18, -0.08), (0.36, -0.12), (cu, -0.14), (0.64, -0.12), (0.82, -0.08),
                # Upper forehead
                (0.12, 0.04), (0.32, 0.02), (cu, 0.02), (0.68, 0.02), (0.88, 0.04),
                # Supra-brow ridge (top of orbital quads — sits ABOVE eyebrows)
                (0.06, brow_top_v + 0.03),
                (lu_out, brow_top_v),
                (left_eye_u, brow_top_v - 0.01),
                (lu_in, brow_top_v + 0.01),
                (cu, brow_top_v + 0.02),
                (ru_in, brow_top_v + 0.01),
                (right_eye_u, brow_top_v - 0.01),
                (ru_out, brow_top_v),
                (0.94, brow_top_v + 0.03),
                # Mid-eye bridge & outer temples ONLY (nothing inside the iris/sclera!)
                (0.04, eye_v),
                (lu_in + 0.01, eye_v),
                (cu, eye_v),
                (ru_in - 0.01, eye_v),
                (0.96, eye_v),
                # Infra-orbital / cheekbone ridge (bottom of orbital quads — sits BELOW lower eyelids)
                (0.06, eye_bot_v + 0.02),
                (lu_out, eye_bot_v),
                (left_eye_u, eye_bot_v + 0.01),
                (lu_in, eye_bot_v),
                (cu, eye_bot_v + 0.01),
                (ru_in, eye_bot_v),
                (right_eye_u, eye_bot_v + 0.01),
                (ru_out, eye_bot_v),
                (0.94, eye_bot_v + 0.02),
                # Nose tip, nostril wings & mid-cheek contours
                (0.08, nose_tip_v),
                (cu - 0.14, nose_tip_v + 0.01),
                (cu, nose_tip_v),
                (cu + 0.14, nose_tip_v + 0.01),
                (0.92, nose_tip_v),
                # Upper mouth box & outer jaw
                (0.11, mouth_top_v + 0.02),
                (cu - 0.17, mouth_top_v),
                (cu, mouth_top_v),
                (cu + 0.17, mouth_top_v),
                (0.89, mouth_top_v + 0.02),
                # Lower mouth box & gonial jaw corners
                (0.15, mouth_bot_v),
                (cu - 0.16, mouth_bot_v),
                (cu, mouth_bot_v + 0.01),
                (cu + 0.16, mouth_bot_v),
                (0.85, mouth_bot_v),
                # Chin contour
                (0.28, 0.99),
                (cu, 1.03),
                (0.72, 0.99),
            ]

            pts_native = []
            for u, v in uv_cage:
                px = float(np.clip(fx + u * fw, 1.0, ps1_w - 2.0))
                py = float(np.clip(fy + v * fh, 1.0, ps1_h - 2.0))
                pts_native.append([px, py])

            hx1 = int(np.clip(np.floor(fx - 0.18 * fw), 0, ps1_w - 2))
            hy1 = int(np.clip(np.floor(fy - 0.28 * fh), 0, ps1_h - 2))
            hx2 = int(np.clip(np.ceil(fx + 1.18 * fw), hx1 + 2, ps1_w))
            hy2 = int(np.clip(np.ceil(fy + 1.12 * fh), hy1 + 2, ps1_h))

            prot_boxes = [
                (
                    int(fx + lu_out * fw),
                    int(fy + brow_top_v * fh),
                    int(fx + lu_in * fw),
                    int(fy + eye_bot_v * fh),
                ),
                (
                    int(fx + ru_in * fw),
                    int(fy + brow_top_v * fh),
                    int(fx + ru_out * fw),
                    int(fy + eye_bot_v * fh),
                ),
                (
                    int(fx + (cu - 0.18) * fw),
                    int(fy + mouth_top_v * fh),
                    int(fx + (cu + 0.18) * fw),
                    int(fy + mouth_bot_v * fh),
                ),
            ]

            faces.append(
                FaceRegion(
                    box=(int(round(fx)), int(round(fy)), int(round(fw)), int(round(fh))),
                    head_box=(hx1, hy1, hx2, hy2),
                    landmarks=np.array(pts_native, dtype=np.float32),
                    eye_mouth_boxes=prot_boxes,
                )
            )

        return faces

    @staticmethod
    def build_head_weight_map(
        ps1_h: int,
        ps1_w: int,
        faces: List[FaceRegion],
        poly_mask: Optional[np.ndarray],
        depth: np.ndarray,
        cfg: Optional[PS1Config] = None,
    ) -> np.ndarray:
        """
        Returns a smooth [0, 1] weight mask over head/face regions so `render_ps1_hardware` prioritizes
        crisp CLUT texture clarity & gentle 3D shading on faces while keeping clothing/body chunky.
        """
        weight = np.zeros((ps1_h, ps1_w), dtype=np.float32)
        if faces:
            for face in faces:
                fx, fy, fw, fh = face.box
                cx = int(round(fx + fw * 0.5))
                cy = int(round(fy + fh * 0.46))
                rx = int(round(fw * 0.62))
                ry = int(round(fh * 0.68))
                cv2.ellipse(weight, (cx, cy), (max(4, rx), max(4, ry)), 0, 0, 360, 1.0, -1)
        elif poly_mask is not None and np.any(poly_mask > 0):
            ys, xs = np.where(poly_mask > 0)
            y_min, y_max = int(ys.min()), int(ys.max())
            span_h = y_max - y_min
            if span_h > ps1_h * 0.35:
                head_y2 = y_min + int(span_h * 0.48)
                sub = poly_mask[y_min:head_y2, :] > 0
                if np.any(sub):
                    weight[y_min:head_y2, :][sub] = 0.85

        if np.any(weight > 0):
            sigma = max(2.5, min(ps1_h, ps1_w) * 0.03)
            weight = cv2.GaussianBlur(weight, (0, 0), sigmaX=sigma)
            weight = np.clip(weight * 1.15, 0.0, 1.0)
        if cfg is not None:
            weight *= float(np.clip(cfg.face_preservation, 0.0, 1.0))
        return weight

    @staticmethod
    def _quantize_clut_kmeans(
        img_rgb: np.ndarray,
        n_colors: int,
        seed: int = 42,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Simulates a dedicated PS1 VRAM 4-bit/5-bit CLUT color palette via k-means in RGB space.
        When `mask` is provided, palette entries are learned strictly from masked pixels.
        """
        h, w, _ = img_rgb.shape
        all_pixels = img_rgb.reshape(-1, 3).astype(np.float32)
        if mask is not None and np.any(mask > 0):
            train_pixels = img_rgb[mask > 0].reshape(-1, 3).astype(np.float32)
        else:
            train_pixels = all_pixels

        n_colors = max(4, min(n_colors, len(train_pixels)))
        cv2.setRNGSeed(seed)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.7)
        _, _, centers = cv2.kmeans(train_pixels, n_colors, None, criteria, 2, cv2.KMEANS_PP_CENTERS)

        diffs = all_pixels[:, None, :] - centers[None, :, :]
        nearest = np.argmin(np.sum(diffs * diffs, axis=2), axis=1)
        quant_rgb = centers[nearest].reshape(h, w, 3)
        return np.clip(np.round(quant_rgb), 0, 255).astype(np.uint8)

    def build_stylized_base_and_texture(
        self,
        orig_rgb: np.ndarray,
        native_rgb: np.ndarray,
        cfg: PS1Config,
        faces: List[FaceRegion],
        fg_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Synthesizes authentic PS1 VRAM textures for the 3D subject mesh:
          1. Body / Subject VRAM Texture Page at `cfg.tex_resolution` with foreground CLUT palette
          2. Dedicated Head / Face VRAM Texture Page (`cfg.head_tex_resolution` across the head box)
             sampled from high-res source with its own dedicated Head CLUT (strictly masked to the subject!)
        """
        nh, nw = native_rgb.shape[:2]
        oh, ow = orig_rgb.shape[:2]

        work = cv2.cvtColor(native_rgb, cv2.COLOR_RGB2BGR)
        for _ in range(2):
            work = cv2.bilateralFilter(work, d=5, sigmaColor=30, sigmaSpace=10)
        stylized_rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)

        gray = cv2.cvtColor(native_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        blur_g = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(1.1, min(nh, nw) / 115.0))
        detail = np.clip((gray - blur_g) / 255.0, -0.30, 0.22)
        boosted_body = np.clip(stylized_rgb.astype(np.float32) + detail[:, :, None] * 95.0, 0, 255).astype(np.uint8)

        if nw <= nh:
            tex_w = max(32, cfg.tex_resolution)
            tex_h = max(32, int(round(tex_w * (nh / float(nw)))))
        else:
            tex_h = max(32, cfg.tex_resolution)
            tex_w = max(32, int(round(tex_h * (nw / float(nh)))))

        tiny_body = cv2.resize(boosted_body, (tex_w, tex_h), interpolation=cv2.INTER_AREA)
        tiny_fg_mask = (
            cv2.resize(fg_mask, (tex_w, tex_h), interpolation=cv2.INTER_NEAREST)
            if fg_mask is not None
            else None
        )
        body_clut = self._quantize_clut_kmeans(tiny_body, cfg.clut_colors, seed=cfg.seed, mask=tiny_fg_mask)
        ps1_tex_rgb = cv2.resize(body_clut, (nw, nh), interpolation=cv2.INTER_NEAREST)

        for f_idx, face in enumerate(faces):
            hx1, hy1, hx2, hy2 = face.head_box
            hw_n, hh_n = hx2 - hx1, hy2 - hy1
            if hw_n < 12 or hh_n < 12:
                continue

            ox1 = int(np.clip(round(hx1 * (ow / float(nw))), 0, ow - 2))
            oy1 = int(np.clip(round(hy1 * (oh / float(nh))), 0, oh - 2))
            ox2 = int(np.clip(round(hx2 * (ow / float(nw))), ox1 + 2, ow))
            oy2 = int(np.clip(round(hy2 * (oh / float(nh))), oy1 + 2, oh))
            head_orig = orig_rgb[oy1:oy2, ox1:ox2]

            h_aspect = hw_n / float(hh_n)
            head_short = max(28, min(cfg.head_tex_resolution, min(hw_n, hh_n)))
            if h_aspect <= 1.0:
                ht_w = head_short
                ht_h = max(28, int(round(head_short / h_aspect)))
            else:
                ht_h = head_short
                ht_w = max(28, int(round(head_short * h_aspect)))

            head_small = cv2.resize(head_orig, (ht_w, ht_h), interpolation=cv2.INTER_AREA)
            head_clean = cv2.cvtColor(
                cv2.bilateralFilter(cv2.cvtColor(head_small, cv2.COLOR_RGB2BGR), d=5, sigmaColor=20, sigmaSpace=8),
                cv2.COLOR_BGR2RGB,
            )

            hg = cv2.cvtColor(head_small, cv2.COLOR_RGB2GRAY).astype(np.float32)
            hg_blur = cv2.GaussianBlur(hg, (0, 0), sigmaX=1.0)
            h_detail = np.clip((hg - hg_blur) / 255.0, -0.35, 0.26)
            head_painted = np.clip(head_clean.astype(np.float32) + h_detail[:, :, None] * 115.0, 0, 255).astype(np.uint8)

            head_mask_tiny = None
            if fg_mask is not None:
                hm_crop = fg_mask[hy1:hy2, hx1:hx2]
                head_mask_tiny = cv2.resize(hm_crop, (ht_w, ht_h), interpolation=cv2.INTER_NEAREST)

            head_clut = self._quantize_clut_kmeans(
                head_painted, cfg.head_clut_colors, seed=cfg.seed + 31 + f_idx, mask=head_mask_tiny
            )
            head_upscaled = cv2.resize(head_clut, (hw_n, hh_n), interpolation=cv2.INTER_NEAREST)

            blend = np.ones((hh_n, hw_n), dtype=np.float32)
            pad_x = max(2, int(round(hw_n * 0.10)))
            pad_y = max(2, int(round(hh_n * 0.10)))
            blend[:pad_y, :] *= np.linspace(0.0, 1.0, pad_y, dtype=np.float32)[:, None]
            blend[-pad_y:, :] *= np.linspace(1.0, 0.0, pad_y, dtype=np.float32)[:, None]
            blend[:, :pad_x] *= np.linspace(0.0, 1.0, pad_x, dtype=np.float32)[None, :]
            blend[:, -pad_x:] *= np.linspace(1.0, 0.0, pad_x, dtype=np.float32)[None, :]

            cur_patch = ps1_tex_rgb[hy1:hy2, hx1:hx2].astype(np.float32)
            merged = cur_patch * (1.0 - blend[:, :, None]) + head_upscaled.astype(np.float32) * blend[:, :, None]
            ps1_tex_rgb[hy1:hy2, hx1:hx2] = np.clip(np.round(merged), 0, 255).astype(np.uint8)

        return stylized_rgb, ps1_tex_rgb

    def extract_polygonal_foreground(
        self,
        img_rgb: np.ndarray,
        depth: np.ndarray,
        faces: List[FaceRegion],
        cfg: PS1Config,
    ) -> Tuple[Optional[np.ndarray], Optional[list]]:
        """
        Extracts the foreground character/subject as a straight-edged PS1 polygonal silhouette (`poly_mask`, `poly_list`)
        so that:
          - All 3D low-poly facets, normal lighting, and creases are confined strictly to the subject
          - The background wall, ceiling light, or sky remains a flat, clean 2D PS1 backdrop plate
          - Detected faces/heads are locked as foreground so GrabCut never chops off heads on outdoor scenes
        """
        h, w = depth.shape
        d_u8 = (depth * 255.0).astype(np.uint8)
        otsu_thresh, _ = cv2.threshold(d_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_val = float(otsu_thresh) / 255.0

        fg_init = depth >= thresh_val
        fg_ratio = float(np.mean(fg_init))
        if fg_ratio < 0.08 or fg_ratio > 0.92:
            return None, None
        fg_mean = float(depth[fg_init].mean())
        bg_mean = float(depth[~fg_init].mean())
        if (fg_mean - bg_mean) < 0.20:
            return None, None

        gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
        gc_mask[depth < max(0.06, thresh_val - 0.20)] = cv2.GC_BGD
        gc_mask[depth >= thresh_val] = cv2.GC_PR_FGD
        gc_mask[depth >= min(0.92, thresh_val + 0.16)] = cv2.GC_FGD

        # If image corners indicate a solid background cutout (e.g. pure white or dark backdrop),
        # seed known background pixels to cleanly detach limbs and prevent floor bridging
        tl_corner = img_rgb[0, 0].astype(np.float32)
        tr_corner = img_rgb[0, -1].astype(np.float32)
        if np.linalg.norm(tl_corner - tr_corner) < 15.0 and (np.all(tl_corner > 235) or np.all(tl_corner < 20)):
            bg_target = 0.5 * (tl_corner + tr_corner)
            color_dist = np.linalg.norm(img_rgb.astype(np.float32) - bg_target, axis=-1)
            gc_mask[color_dist < 20.0] = cv2.GC_BGD

        # Seed strict inner core of detected faces (nose/eyes/mouth) as definite foreground so GrabCut
        # never notches cheeks, while letting outer jaw/ear edges snap tightly to true silhouette boundaries!
        for face in faces:
            fx, fy, fw, fh = face.box
            cx = int(round(fx + fw * 0.5))
            cy = int(round(fy + fh * 0.48))
            rx_inner = max(3, int(round(fw * 0.32)))
            ry_inner = max(3, int(round(fh * 0.40)))
            cv2.ellipse(gc_mask, (cx, cy), (rx_inner, ry_inner), 0, 0, 360, int(cv2.GC_FGD), -1)

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        try:
            cv2.grabCut(bgr, gc_mask, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
            refined = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        except cv2.error:
            refined = (fg_init.astype(np.uint8)) * 255

        k_size = max(3, int(round(min(h, w) * 0.02)) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel)
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)

        contours, hierarchy = cv2.findContours(refined, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not contours or hierarchy is None:
            return None, None

        min_area = 0.04 * (h * w)
        min_hole_area = max(16, int(0.002 * (h * w)))

        valid_ext_indices = [
            i for i in range(len(contours))
            if hierarchy[0][i][3] == -1 and cv2.contourArea(contours[i]) >= min_area
        ]
        if not valid_ext_indices:
            return None, None

        poly_mask = np.zeros((h, w), dtype=np.uint8)
        poly_list = []
        for i in valid_ext_indices:
            cnt = contours[i]
            peri = cv2.arcLength(cnt, True)
            eps = cfg.silhouette_simplify * peri
            approx = cv2.approxPolyDP(cnt, eps, True)
            if len(approx) >= 4:
                cv2.fillPoly(poly_mask, [approx], 255)
                poly_list.append(approx.reshape(-1, 2))

        # Carve out interior negative-space holes (e.g. between legs, between bent arms and torso)
        for i in range(len(contours)):
            parent_idx = hierarchy[0][i][3]
            if parent_idx in valid_ext_indices and cv2.contourArea(contours[i]) >= min_hole_area:
                peri = cv2.arcLength(contours[i], True)
                eps = max(1.5, cfg.silhouette_simplify * peri)
                approx = cv2.approxPolyDP(contours[i], eps, True)
                if len(approx) >= 3:
                    cv2.fillPoly(poly_mask, [approx], 0)
                    poly_list.append(approx.reshape(-1, 2))

        if not poly_list:
            return None, None

        final_ratio = float(np.mean(poly_mask > 0))
        if final_ratio < 0.06 or final_ratio > 0.94:
            return None, None

        return poly_mask, poly_list

    @staticmethod
    def _subdivide_polygon_contour(poly_pts: np.ndarray, max_seg_len: float) -> np.ndarray:
        """Subdivides long silhouette segments so Delaunay edges adhere tightly to the polygonal outline."""
        out = []
        n = len(poly_pts)
        for i in range(n):
            p0 = poly_pts[i].astype(np.float32)
            p1 = poly_pts[(i + 1) % n].astype(np.float32)
            dist = float(np.linalg.norm(p1 - p0))
            steps = max(1, int(np.ceil(dist / max_seg_len)))
            for s in range(steps):
                t = s / float(steps)
                out.append(p0 * (1.0 - t) + p1 * t)
        return np.array(out, dtype=np.float32)

    def generate_feature_mesh(
        self,
        stylized_rgb: np.ndarray,
        depth: np.ndarray,
        poly_mask: Optional[np.ndarray],
        poly_contours: Optional[list],
        faces: List[FaceRegion],
        cfg: PS1Config,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Builds an anatomical + feature-aligned low-poly 3D mesh strictly confined to the foreground subject:
          - Locks clean symmetrical PS1 head cage landmarks (orbital quads, nose bridge, cheek planes, mouth box)
          - Suppresses random clutter vertices inside smooth facial planes & eye sockets
          - Places vertices along silhouette & strong clothing/accessory contours
          - Carves out all background regions so walls/ceilings/sky NEVER receive 3D triangular facets!
        """
        rng = np.random.default_rng(cfg.seed)
        depth = np.asarray(depth, dtype=np.float32)
        h, w = depth.shape
        target_verts = max(55, int(round(cfg.poly_count * 0.54)))

        active_area = float(np.sum(poly_mask > 0)) if poly_mask is not None else float(h * w)
        avg_tri_spacing = np.sqrt(active_area / float(max(30, target_verts)))
        min_dist = max(4.0, avg_tri_spacing * 0.55)

        chosen_pts: List[List[float]] = []

        def _try_add_point(px: float, py: float, custom_min_d: float) -> bool:
            px = float(np.clip(px, 0, w - 1))
            py = float(np.clip(py, 0, h - 1))
            if not chosen_pts:
                chosen_pts.append([px, py])
                return True
            arr = np.asarray(chosen_pts, dtype=np.float32)
            d2 = np.sum((arr - np.array([px, py], dtype=np.float32)) ** 2, axis=1)
            if float(np.min(d2)) >= (custom_min_d ** 2):
                chosen_pts.append([px, py])
                return True
            return False

        # 1. Seed outer polygonal silhouette (or frame border if no foreground cutout exists)
        if poly_mask is not None and poly_contours is not None:
            max_sil_seg = max(8.0, avg_tri_spacing * 0.95)
            for cnt in poly_contours:
                for pt in cnt:
                    _try_add_point(pt[0], pt[1], min_dist * 0.45)
                sub_pts = self._subdivide_polygon_contour(cnt, max_sil_seg)
                for pt in sub_pts:
                    _try_add_point(pt[0], pt[1], min_dist * 0.65)
        else:
            border_steps_x = max(8, int(round(w / (avg_tri_spacing * 1.1))))
            border_steps_y = max(8, int(round(h / (avg_tri_spacing * 1.1))))
            for x in np.linspace(0, w - 1, border_steps_x):
                _try_add_point(x, 0, 2.0)
                _try_add_point(x, h - 1, 2.0)
            for y in np.linspace(0, h - 1, border_steps_y):
                _try_add_point(0, y, 2.0)
                _try_add_point(w - 1, y, 2.0)

        # 2. Seed structured anatomical PS1 head cage landmarks & build smooth-face exclusion mask
        face_smooth_exclusion = np.zeros((h, w), dtype=np.uint8)
        orbital_mouth_exclusion = np.zeros((h, w), dtype=np.uint8)

        for face in faces:
            fx, fy, fw, fh = face.box
            face_min_d = max(2.8, min(fw, fh) * 0.085)
            for lm in face.landmarks:
                ix_l, iy_l = int(round(lm[0])), int(round(lm[1]))
                if poly_mask is None or poly_mask[iy_l, ix_l] > 0:
                    _try_add_point(lm[0], lm[1], face_min_d)

            # Sample 5 natural crown arch points along true hat/beanie/hair contour
            if poly_mask is not None:
                mask_n = cv2.resize(poly_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                for u_arch in (0.18, 0.36, 0.50, 0.64, 0.82):
                    col_x = int(np.clip(round(fx + u_arch * fw), 0, w - 1))
                    col_ys = np.where(mask_n[:max(1, int(round(fy + fh * 0.15))), col_x] > 0)[0]
                    if len(col_ys) > 0:
                        top_y = float(col_ys.min())
                        min_lm_y = float(np.min(face.landmarks[:, 1])) if len(face.landmarks) > 0 else fy
                        # Only add extra crown arch vertices if hat/beanie/hair extends significantly above face cage
                        if top_y < min_lm_y - 7.5:
                            _try_add_point(float(col_x), top_y, min_dist * 0.55)

            cx = int(round(fx + fw * 0.5))
            cy = int(round(fy + fh * 0.48))
            cv2.ellipse(
                face_smooth_exclusion,
                (cx, cy),
                (max(4, int(round(fw * 0.46))), max(4, int(round(fh * 0.54)))),
                0, 0, 360, 255, -1,
            )
            for x1, y1, x2, y2 in face.eye_mouth_boxes:
                cv2.rectangle(
                    orbital_mouth_exclusion,
                    (max(0, x1 - 2), max(0, y1 - 2)),
                    (min(w - 1, x2 + 2), min(h - 1, y2 + 2)),
                    255,
                    -1,
                )

        # 3. Seed strong structural Canny contours inside the subject
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        eq_l = clahe.apply(cv2.cvtColor(stylized_rgb, cv2.COLOR_RGB2GRAY))
        edges = cv2.Canny(eq_l, 55, 135)
        if poly_mask is not None:
            erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            edges = cv2.bitwise_and(edges, cv2.erode(poly_mask, erode_k))
        edges[orbital_mouth_exclusion > 0] = 0

        edge_contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        edge_contours = sorted(edge_contours, key=lambda c: cv2.arcLength(c, False), reverse=True)
        edge_budget = int(target_verts * 0.72)
        for ec in edge_contours:
            if len(chosen_pts) >= edge_budget:
                break
            arc_len = cv2.arcLength(ec, False)
            if arc_len < min_dist * 1.6:
                continue
            approx_ec = cv2.approxPolyDP(ec, max(2.0, min_dist * 0.45), False).reshape(-1, 2)
            for pt in approx_ec:
                if len(chosen_pts) >= edge_budget:
                    break
                px_i, py_i = int(round(pt[0])), int(round(pt[1]))
                local_d = min_dist * 1.15 if face_smooth_exclusion[py_i, px_i] > 0 else min_dist * 0.85
                _try_add_point(pt[0], pt[1], local_d)

        # 4. Fill remaining body/subject vertex budget outside the protected face oval
        lab = cv2.cvtColor(stylized_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        gx = cv2.Scharr(eq_l.astype(np.float32), cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(eq_l.astype(np.float32), cv2.CV_32F, 0, 1)
        color_grad = np.hypot(gx, gy)
        color_grad /= (np.percentile(color_grad, 99.0) + 1e-6)

        dgx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
        dgy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
        depth_grad = np.hypot(dgx, dgy)
        depth_grad /= (np.percentile(depth_grad, 99.0) + 1e-6)

        importance = (0.18 + 0.55 * color_grad + 0.27 * depth_grad) * (0.40 + 0.80 * depth)
        if poly_mask is not None:
            importance[poly_mask == 0] = 0.0
        importance[face_smooth_exclusion > 0] = 0.0

        prob = importance.ravel().astype(np.float64)
        prob_sum = prob.sum()
        if prob_sum > 1e-8 and len(chosen_pts) < target_verts:
            prob /= prob_sum
            n_candidates = min(len(prob), max(2500, target_verts * 14))
            flat_indices = rng.choice(len(prob), size=n_candidates, replace=True, p=prob)
            cy_cand, cx_cand = np.divmod(flat_indices, w)
            for cur_scale in (1.0, 0.72, 0.50):
                if len(chosen_pts) >= target_verts:
                    break
                for cx_i, cy_i in zip(cx_cand, cy_cand):
                    if len(chosen_pts) >= target_verts:
                        break
                    jx = cx_i + rng.uniform(-0.35, 0.35)
                    jy = cy_i + rng.uniform(-0.35, 0.35)
                    _try_add_point(jx, jy, min_dist * cur_scale)

        points = np.array(chosen_pts, dtype=np.float32)
        tri_delaunay = Delaunay(points)
        triangles = tri_delaunay.simplices.astype(np.int32)

        # 5. Strictly carve out all triangles outside the foreground subject (`poly_mask`)
        if poly_mask is not None:
            centroids = np.mean(points[triangles], axis=1)
            mid01 = 0.5 * (points[triangles[:, 0]] + points[triangles[:, 1]])
            mid12 = 0.5 * (points[triangles[:, 1]] + points[triangles[:, 2]])
            mid20 = 0.5 * (points[triangles[:, 2]] + points[triangles[:, 0]])

            def _inside(pts_2d: np.ndarray) -> np.ndarray:
                ix = np.clip(np.round(pts_2d[:, 0]).astype(int), 0, w - 1)
                iy = np.clip(np.round(pts_2d[:, 1]).astype(int), 0, h - 1)
                return poly_mask[iy, ix] > 0

            inside_votes = (
                _inside(centroids).astype(int) * 2
                + _inside(mid01).astype(int)
                + _inside(mid12).astype(int)
                + _inside(mid20).astype(int)
            )
            triangles = triangles[inside_votes >= 3]

        # 6. Optimize interior quad diagonals ONLY outside protected eye/mouth quads
        triangles = self._optimize_triangulation_edges(points, triangles, lab, depth, orbital_mouth_exclusion)

        # 7. Smooth, tear-free 3D Z surface across the foreground subject
        clean_depth = depth.copy()
        if poly_mask is not None and np.any(poly_mask == 0):
            bg_u8 = (poly_mask == 0).astype(np.uint8) * 255
            d_u8 = np.clip(np.round(clean_depth * 255.0), 0, 255).astype(np.uint8)
            clean_depth = cv2.inpaint(d_u8, bg_u8, inpaintRadius=5, flags=cv2.INPAINT_TELEA).astype(np.float32) / 255.0

        sigma_d = max(2.5, min(h, w) * 0.025)
        depth_smooth = cv2.GaussianBlur(clean_depth, (0, 0), sigmaX=sigma_d)

        for face in faces:
            fx, fy, fw, fh = face.box
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            nx = (xx - (fx + fw * 0.5)) / (max(1.0, fw) * 0.55)
            ny = (yy - (fy + fh * 0.48)) / (max(1.0, fh) * 0.60)
            r2 = nx * nx + ny * ny
            cranial_dome = np.clip(1.0 - r2, 0.0, 1.0) ** 1.2
            nose_ridge = np.exp(-((nx / 0.22) ** 2)) * np.exp(-(((ny - 0.05) / 0.32) ** 2))
            depth_smooth += 0.08 * cranial_dome + 0.05 * nose_ridge

        ix = np.clip(np.round(points[:, 0]).astype(int), 0, w - 1)
        iy = np.clip(np.round(points[:, 1]).astype(int), 0, h - 1)
        z_raw = np.clip(depth_smooth[iy, ix], 0.0, 1.25)
        z_quant = np.round(z_raw * 64.0) / 64.0
        z_coords = z_quant * cfg.depth_scale

        vertices_3d = np.column_stack([points[:, 0], points[:, 1], z_coords]).astype(np.float32)

        eps_b = 1.5
        border_pin = (
            (points[:, 0] <= eps_b)
            | (points[:, 0] >= (w - 1 - eps_b))
            | (points[:, 1] <= eps_b)
            | (points[:, 1] >= (h - 1 - eps_b))
        )
        return vertices_3d, triangles, border_pin

    @staticmethod
    def _optimize_triangulation_edges(
        points: np.ndarray,
        triangles: np.ndarray,
        lab_img: np.ndarray,
        depth: np.ndarray,
        protected_mask: Optional[np.ndarray] = None,
        passes: int = 2,
    ) -> np.ndarray:
        """Flips interior quad diagonals when doing so reduces color/depth variance without disturbing eye/mouth quads."""
        h, w = depth.shape
        tris = triangles.copy()
        feature_field = np.dstack([lab_img * 0.6, depth[:, :, None] * 60.0]).astype(np.float32)

        def _tri_variance_cost(i0: int, i1: int, i2: int) -> float:
            p0, p1, p2 = points[i0], points[i1], points[i2]
            samples = np.array(
                [
                    0.60 * p0 + 0.20 * p1 + 0.20 * p2,
                    0.20 * p0 + 0.60 * p1 + 0.20 * p2,
                    0.20 * p0 + 0.20 * p1 + 0.60 * p2,
                    (p0 + p1 + p2) / 3.0,
                ],
                dtype=np.float32,
            )
            sx = np.clip(np.round(samples[:, 0]).astype(int), 0, w - 1)
            sy = np.clip(np.round(samples[:, 1]).astype(int), 0, h - 1)
            vals = feature_field[sy, sx]
            return float(np.sum(np.var(vals, axis=0)))

        def _signed_area2(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
            return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

        def _min_cos_ok(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
            v0, v1, v2 = b - a, c - b, a - c
            l0 = np.linalg.norm(v0) + 1e-6
            l1 = np.linalg.norm(v1) + 1e-6
            l2 = np.linalg.norm(v2) + 1e-6
            c0 = abs(float(np.dot(v0, -v2)) / (l0 * l2))
            c1 = abs(float(np.dot(v1, -v0)) / (l1 * l0))
            c2 = abs(float(np.dot(v2, -v1)) / (l2 * l1))
            return max(c0, c1, c2) < 0.955

        for _ in range(passes):
            edge_map = {}
            for t_idx, (v0, v1, v2) in enumerate(tris):
                for a, b, opp in ((v0, v1, v2), (v1, v2, v0), (v2, v0, v1)):
                    key = (min(a, b), max(a, b))
                    edge_map.setdefault(key, []).append((t_idx, a, b, opp))

            existing_edges = set(edge_map.keys())
            locked_tris = set()

            for key, adj in edge_map.items():
                if len(adj) != 2:
                    continue
                (t1, u, v, a), (t2, u2, v2, b) = adj
                if t1 in locked_tris or t2 in locked_tris:
                    continue

                if protected_mask is not None:
                    mid = 0.5 * (points[u] + points[v])
                    mx, my = int(np.clip(round(mid[0]), 0, w - 1)), int(np.clip(round(mid[1]), 0, h - 1))
                    if protected_mask[my, mx] > 0:
                        continue

                new_key = (min(a, b), max(a, b))
                if new_key in existing_edges or a == b:
                    continue

                if _signed_area2(points[u], points[v], points[a]) < 0:
                    u, v = v, u
                pa, pb, pu, pv = points[a], points[b], points[u], points[v]
                if _signed_area2(pu, pb, pa) <= 2.5 or _signed_area2(pb, pv, pa) <= 2.5:
                    continue
                if not (_min_cos_ok(pu, pb, pa) and _min_cos_ok(pb, pv, pa)):
                    continue

                old_cost = _tri_variance_cost(u, v, a) + _tri_variance_cost(v, u, b)
                new_cost = _tri_variance_cost(u, b, a) + _tri_variance_cost(b, v, a)
                if new_cost < old_cost * 0.90:
                    tris[t1] = [u, b, a]
                    tris[t2] = [b, v, a]
                    locked_tris.add(t1)
                    locked_tris.add(t2)
                    existing_edges.add(new_key)

        return tris

    def render_backdrop(
        self,
        img_rgb: np.ndarray,
        depth: np.ndarray,
        poly_mask: Optional[np.ndarray],
        cfg: PS1Config,
    ) -> np.ndarray:
        """
        Renders a clean, flat 2D PS1 pre-rendered backdrop plate with atmospheric depth fog.
        Only inpaints a narrow 3px boundary rim around `poly_mask` (to cover 1-2px camera yaw shifts)
        so background walls, ceiling lights, flags, and hills are NEVER smeared or triangulated!
        """
        h, w = img_rgb.shape[:2]
        work_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        if poly_mask is not None:
            dil_px = max(5, int(round(min(h, w) * 0.045)))
            dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_px * 2 + 1, dil_px * 2 + 1))
            inpaint_mask = cv2.dilate(poly_mask, dil_k)
            work_bgr = cv2.inpaint(work_bgr, inpaint_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

        bg_rgb = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2RGB)
        sky_w = max(48, w // 2)
        sky_h = max(48, h // 2)
        bg_small = cv2.resize(bg_rgb, (sky_w, sky_h), interpolation=cv2.INTER_AREA)
        bg_clut = self._quantize_clut_kmeans(bg_small, cfg.bg_clut_colors, seed=cfg.seed + 7)
        bg_low = cv2.resize(bg_clut, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)

        bg_mask = (poly_mask == 0) if poly_mask is not None else (depth < 0.45)
        if np.any(bg_mask):
            fog_color = np.median(bg_low[bg_mask], axis=0) * 0.80 + np.array([20.0, 24.0, 28.0], dtype=np.float32)
        else:
            fog_color = np.array([60.0, 68.0, 78.0], dtype=np.float32)

        dist_factor = np.clip(1.0 - depth, 0.0, 1.0) ** 1.2
        fog_alpha = np.clip(dist_factor * cfg.fog_strength, 0.0, 0.60)[:, :, None]
        fogged = bg_low * (1.0 - fog_alpha) + fog_color[None, None, :] * fog_alpha
        return np.clip(fogged, 0, 255).astype(np.uint8)

    def render_ps1_hardware(
        self,
        stylized_rgb: np.ndarray,
        ps1_tex_rgb: np.ndarray,
        backdrop_rgb: np.ndarray,
        head_weight_map: np.ndarray,
        vertices: np.ndarray,
        triangles: np.ndarray,
        border_pin: np.ndarray,
        cfg: PS1Config,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Authentic PS1 GTE + GPU software rasterizer running natively at 240p:
          - Background (`backdrop_rgb`) remains a flat, smooth 2D PS1 backdrop plate with ZERO 3D facets
          - Only the foreground character (`triangles`) receives 3D normal lighting, creases, and affine warping
          - Face/head regions prioritize crisp CLUT texture mapping (`face_flatness_max <= 0.04`)
        """
        h, w = stylized_rgb.shape[:2]
        framebuffer = backdrop_rgb.copy()

        if len(triangles) == 0:
            upscaled_rgb = self.apply_ps1_vram_output(framebuffer, cfg)
            return upscaled_rgb, upscaled_rgb.copy()

        centroids_2d = np.mean(vertices[triangles, :2], axis=1)
        c_ix = np.clip(np.round(centroids_2d[:, 0]).astype(int), 0, w - 1)
        c_iy = np.clip(np.round(centroids_2d[:, 1]).astype(int), 0, h - 1)
        tri_head_w = head_weight_map[c_iy, c_ix]

        # 1. Compute 3D triangle surface normals for PS1 GTE flat/faceted lighting
        v0 = vertices[triangles[:, 0]]
        v1 = vertices[triangles[:, 1]]
        v2 = vertices[triangles[:, 2]]
        e1 = v1 - v0
        e2 = v2 - v0
        e1_rh = np.stack([e1[:, 0], -e1[:, 1], e1[:, 2]], axis=1)
        e2_rh = np.stack([e2[:, 0], -e2[:, 1], e2[:, 2]], axis=1)
        normals = np.cross(e1_rh, e2_rh)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-6
        normals *= np.where(normals[:, 2:3] < 0, -1.0, 1.0)

        light_dir = np.array([-0.38, 0.44, 0.81], dtype=np.float32)
        light_dir /= np.linalg.norm(light_dir)
        ndotl = np.sum(normals * light_dir[None, :], axis=1)
        neutral_dot = float(np.median(ndotl))
        raw_delta = (ndotl - neutral_dot) * cfg.lighting_strength * 1.35
        raw_delta = np.round(raw_delta * 12.0) / 12.0

        warm_tint = np.array([1.02, 1.00, 0.97], dtype=np.float32)
        cool_tint = np.array([0.95, 0.98, 1.03], dtype=np.float32)
        facet_light_rgb = np.ones((len(triangles), 3), dtype=np.float32)
        for i, d_val in enumerate(raw_delta):
            hw_i = float(tri_head_w[i])
            max_swing = 0.17 * hw_i + 0.36 * (1.0 - hw_i)
            clamped_d = float(np.clip(d_val * (1.0 - 0.32 * hw_i), -max_swing, max_swing))
            tint = warm_tint if clamped_d >= 0 else cool_tint
            facet_light_rgb[i] = (1.0 + clamped_d) * tint

        edge_normals = {}
        for t_idx, (i0, i1, i2) in enumerate(triangles):
            for a, b in ((i0, i1), (i1, i2), (i2, i0)):
                k = (min(a, b), max(a, b))
                edge_normals.setdefault(k, []).append(t_idx)

        crease_edges = {}
        for k, t_list in edge_normals.items():
            if len(t_list) == 2:
                cos_ang = float(np.clip(np.dot(normals[t_list[0]], normals[t_list[1]]), -1.0, 1.0))
                fold = np.sqrt(max(0.0, 1.0 - cos_ang))
                if fold > 0.24:
                    crease_edges[k] = min(1.0, fold)

        # 2. Tamed 3D Camera Projection + PS1 GTE Integer Vertex Snapping
        cx, cy = w * 0.5, h * 0.5
        z_ref = float(np.median(vertices[:, 2]))
        x_cam = vertices[:, 0] - cx
        y_cam = vertices[:, 1] - cy
        z_rel = vertices[:, 2] - z_ref

        yaw = np.radians(cfg.camera_yaw_deg)
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        x_rot = x_cam * cos_y + z_rel * sin_y
        z_rot = -x_cam * sin_y + z_rel * cos_y

        focal = max(w, h) * 3.8
        persp = focal / np.maximum(focal - z_rot * 0.55, focal * 0.65)
        proj_x = cx + x_rot * persp
        proj_y = cy + y_cam * persp

        proj_x = np.where(border_pin & (vertices[:, 0] <= 1.5), 0.0, proj_x)
        proj_x = np.where(border_pin & (vertices[:, 0] >= w - 2.5), float(w - 1), proj_x)
        proj_y = np.where(border_pin & (vertices[:, 1] <= 1.5), 0.0, proj_y)
        proj_y = np.where(border_pin & (vertices[:, 1] >= h - 2.5), float(h - 1), proj_y)

        if cfg.vertex_snap > 0:
            proj_x = np.round(proj_x / cfg.vertex_snap) * cfg.vertex_snap
            proj_y = np.round(proj_y / cfg.vertex_snap) * cfg.vertex_snap

        snapped_2d = np.column_stack([proj_x, proj_y]).astype(np.float32)

        # 3. PS1 Ordering Table (Painter's Algorithm back-to-front Z-sort)
        tri_depths = np.mean(z_rot[triangles], axis=1)
        sorted_indices = np.argsort(tri_depths)

        for idx in sorted_indices:
            tri = triangles[idx]
            src_tri = vertices[tri, :2].astype(np.float32)
            dst_tri = snapped_2d[tri].astype(np.float32)

            src_signed = (src_tri[1, 0] - src_tri[0, 0]) * (src_tri[2, 1] - src_tri[0, 1]) - (
                src_tri[1, 1] - src_tri[0, 1]
            ) * (src_tri[2, 0] - src_tri[0, 0])
            dst_signed = (dst_tri[1, 0] - dst_tri[0, 0]) * (dst_tri[2, 1] - dst_tri[0, 1]) - (
                dst_tri[1, 1] - dst_tri[0, 1]
            ) * (dst_tri[2, 0] - dst_tri[0, 0])
            if src_signed * dst_signed <= 0:
                dst_tri = np.round(src_tri)

            src_rect = cv2.boundingRect(src_tri)
            dst_rect = cv2.boundingRect(dst_tri)
            if src_rect[2] <= 0 or src_rect[3] <= 0 or dst_rect[2] <= 0 or dst_rect[3] <= 0:
                continue

            src_pts = (src_tri - np.array([src_rect[0], src_rect[1]], dtype=np.float32)).reshape(3, 2)
            dst_pts = (dst_tri - np.array([dst_rect[0], dst_rect[1]], dtype=np.float32)).reshape(3, 2)

            if abs(cv2.contourArea(dst_pts)) < 0.20 or abs(cv2.contourArea(src_pts)) < 0.20:
                continue

            sx, sy, sw, sh = src_rect
            tex_crop = ps1_tex_rgb[sy : sy + sh, sx : sx + sw]
            sty_crop = stylized_rgb[sy : sy + sh, sx : sx + sw]
            if tex_crop.size == 0:
                continue

            hw_i = float(tri_head_w[idx])
            src_mask = np.zeros((sh, sw), dtype=np.uint8)
            cv2.fillConvexPoly(src_mask, np.round(src_pts).astype(np.int32), 255, lineType=0)
            src_pixels = sty_crop[src_mask > 0]
            if len(src_pixels) > 0:
                flat_color = np.median(src_pixels, axis=0).astype(np.float32)
                tri_std = float(np.mean(np.std(src_pixels.astype(np.float32), axis=0)))
            else:
                flat_color = sty_crop.mean(axis=(0, 1)).astype(np.float32)
                tri_std = 0.0

            detail_preserve = np.clip((tri_std - 9.0) / 26.0, 0.0, 1.0)
            body_flatness = cfg.facet_flatness * (1.0 - 0.65 * detail_preserve)
            face_flatness = min(cfg.face_flatness_max, cfg.facet_flatness * 0.08)
            local_flatness = hw_i * face_flatness + (1.0 - hw_i) * body_flatness

            dw, dh = dst_rect[2], dst_rect[3]
            try:
                warp_mat = cv2.getAffineTransform(src_pts, dst_pts)
            except cv2.error:
                continue

            warped_tex = cv2.warpAffine(
                tex_crop,
                warp_mat,
                (dw, dh),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_REFLECT_101,
            ).astype(np.float32)

            facet_patch = (1.0 - local_flatness) * warped_tex + local_flatness * flat_color[None, None, :]
            lit_patch = np.clip(facet_patch * facet_light_rgb[idx][None, None, :], 0, 255).astype(np.uint8)

            dst_pts_i = np.round(dst_pts).astype(np.int32)
            poly_cov = np.zeros((dh, dw), dtype=np.uint8)
            cv2.fillConvexPoly(poly_cov, dst_pts_i, 255, lineType=0)

            local_crease = cfg.crease_strength * (1.0 - 0.85 * hw_i)
            if local_crease > 0.01:
                i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
                for loc_a, loc_b, ga, gb in ((0, 1, i0, i1), (1, 2, i1, i2), (2, 0, i2, i0)):
                    ek = (min(ga, gb), max(ga, gb))
                    if ek in crease_edges:
                        edge_mask = np.zeros((dh, dw), dtype=np.uint8)
                        cv2.line(edge_mask, tuple(dst_pts_i[loc_a]), tuple(dst_pts_i[loc_b]), 255, 1, lineType=0)
                        darken = 1.0 - local_crease * crease_edges[ek] * 0.55
                        em = (edge_mask > 0) & (poly_cov > 0)
                        lit_patch[em] = (lit_patch[em].astype(np.float32) * darken).astype(np.uint8)

            dx, dy = dst_rect[0], dst_rect[1]
            x1, x2 = max(0, dx), min(w, dx + dw)
            y1, y2 = max(0, dy), min(h, dy + dh)
            cx1, cx2 = x1 - dx, x1 - dx + (x2 - x1)
            cy1, cy2 = y1 - dy, y1 - dy + (y2 - y1)

            if cx2 > cx1 and cy2 > cy1:
                fb_view = framebuffer[y1:y2, x1:x2]
                m_view = poly_cov[cy1:cy2, cx1:cx2] > 0
                fb_view[m_view] = lit_patch[cy1:cy2, cx1:cx2][m_view]

        # 4. PS1 Hardware 4x4 Bayer Ordered Dithering & Exact 15-Bit RGB555 Quantization
        upscaled_rgb = self.apply_ps1_vram_output(framebuffer, cfg)

        scale = max(1, int(cfg.upscale_factor))
        wireframe_rgb = (upscaled_rgb.astype(np.float32) * 0.78).astype(np.uint8)
        scaled_pts = np.round(snapped_2d * scale).astype(np.int32)
        for tri in triangles:
            pts_tri = scaled_pts[tri].reshape((-1, 1, 2))
            cv2.polylines(wireframe_rgb, [pts_tri], isClosed=True, color=(90, 245, 150), thickness=1, lineType=cv2.LINE_AA)

        return upscaled_rgb, wireframe_rgb

    @staticmethod
    def apply_ps1_vram_output(native_fb_rgb: np.ndarray, cfg: PS1Config) -> np.ndarray:
        """Applies hardware 4x4 ordered Bayer dithering, bug-free 5-bit RGB555 quantization, & crisp upscale."""
        nh, nw = native_fb_rgb.shape[:2]

        bayer = np.tile(PS1_BAYER_4X4, (nh // 4 + 1, nw // 4 + 1))[:nh, :nw, None]
        dithered = native_fb_rgb.astype(np.float32) + bayer * cfg.dither_strength
        dithered = np.clip(dithered, 0.0, 255.0)

        levels_5bit = np.clip(np.round(dithered * 31.0 / 255.0), 0.0, 31.0)
        rgb555 = np.round(levels_5bit * (255.0 / 31.0)).astype(np.uint8)

        scale = max(1, int(cfg.upscale_factor))
        out_w, out_h = nw * scale, nh * scale
        upscaled = cv2.resize(rgb555, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

        if cfg.scanlines and scale >= 2:
            crt = upscaled.astype(np.float32)
            crt[scale - 1 :: scale, :, :] *= 0.87
            upscaled = np.clip(crt, 0, 255).astype(np.uint8)

        return upscaled

    @staticmethod
    def export_obj(
        filename: str,
        vertices: np.ndarray,
        triangles: np.ndarray,
        img_w: int,
        img_h: int,
        depth_scale: float,
    ) -> None:
        with open(filename, "w", encoding="utf-8") as f:
            f.write("# PS1 Low-Poly 3D Mesh\n")
            scale_norm = float(max(img_w, img_h))
            for v in vertices:
                vx = (v[0] - img_w * 0.5) / scale_norm
                vy = -(v[1] - img_h * 0.5) / scale_norm
                vz = (v[2] / max(1.0, depth_scale)) * 0.35
                f.write(f"v {vx:.5f} {vy:.5f} {vz:.5f}\n")
            for v in vertices:
                u = np.clip(v[0] / float(max(1, img_w - 1)), 0.0, 1.0)
                vt_v = np.clip(1.0 - (v[1] / float(max(1, img_h - 1))), 0.0, 1.0)
                f.write(f"vt {u:.5f} {vt_v:.5f}\n")
            for tri in triangles:
                i1, i2, i3 = int(tri[0]) + 1, int(tri[1]) + 1, int(tri[2]) + 1
                f.write(f"f {i1}/{i1} {i2}/{i2} {i3}/{i3}\n")


def process_image_array(
    orig_rgb: np.ndarray,
    config: Optional[PS1Config] = None,
    engine: Optional[PS13DEngine] = None,
    output_obj_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Processes an in-memory RGB image array and returns `(rendered_rgb, wireframe_rgb, stats_dict)`.
    """
    orig_rgb = ensure_rgb_uint8(orig_rgb)
    cfg = config or PRESETS["mgs1"]
    if engine is None:
        engine = PS13DEngine()

    orig_h, orig_w = orig_rgb.shape[:2]
    max_dim = max(orig_h, orig_w)
    if max_dim > 2400:
        work_scale = 2400.0 / float(max_dim)
        work_rgb = cv2.resize(
            orig_rgb,
            (max(64, int(round(orig_w * work_scale))), max(64, int(round(orig_h * work_scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        work_rgb = orig_rgb

    short_side = max(128, min(360, int(cfg.native_short_side)))
    if orig_w <= orig_h:
        ps1_w = short_side
        ps1_h = max(64, int(round(short_side * (orig_h / float(orig_w)))))
    else:
        ps1_h = short_side
        ps1_w = max(64, int(round(short_side * (orig_w / float(orig_h)))))
    ps1_w += ps1_w % 2
    ps1_h += ps1_h % 2

    full_depth = engine.compute_depth(work_rgb)
    faces = engine.detect_faces_and_landmarks(work_rgb, full_depth, ps1_w, ps1_h)

    native_rgb = cv2.resize(work_rgb, (ps1_w, ps1_h), interpolation=cv2.INTER_AREA)
    native_depth = cv2.resize(full_depth, (ps1_w, ps1_h), interpolation=cv2.INTER_AREA).astype(np.float32)

    poly_mask, poly_contours = engine.extract_polygonal_foreground(native_rgb, native_depth, faces, cfg)
    stylized_rgb, ps1_tex_rgb = engine.build_stylized_base_and_texture(
        work_rgb, native_rgb, cfg, faces, fg_mask=poly_mask
    )
    head_weight_map = engine.build_head_weight_map(ps1_h, ps1_w, faces, poly_mask, native_depth, cfg)

    vertices, triangles, border_pin = engine.generate_feature_mesh(
        stylized_rgb, native_depth, poly_mask, poly_contours, faces, cfg
    )

    backdrop_rgb = engine.render_backdrop(native_rgb, native_depth, poly_mask, cfg)
    rendered_rgb, wireframe_rgb = engine.render_ps1_hardware(
        stylized_rgb,
        ps1_tex_rgb,
        backdrop_rgb,
        head_weight_map,
        vertices,
        triangles,
        border_pin,
        cfg,
    )

    if output_obj_path:
        out_dir = os.path.dirname(os.path.abspath(output_obj_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        engine.export_obj(output_obj_path, vertices, triangles, ps1_w, ps1_h, cfg.depth_scale)

    stats = {
        "native_w": ps1_w,
        "native_h": ps1_h,
        "vertices": int(len(vertices)),
        "triangles": int(len(triangles)),
        "faces_detected": len(faces),
        "scene_mode": cfg.scene_mode,
    }
    return rendered_rgb, wireframe_rgb, stats


def process_image(
    input_path: str,
    output_png_path: str,
    output_obj_path: Optional[str] = None,
    config: Optional[PS1Config] = None,
    engine: Optional[PS13DEngine] = None,
) -> np.ndarray:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image not found: {input_path}")

    cfg = config or PRESETS["mgs1"]
    img_bgr = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img_bgr is None:
        raise ValueError(f"Could not decode image: {input_path}")
    if img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
        orig_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2RGBA)
    elif img_bgr.ndim == 3:
        orig_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        orig_rgb = img_bgr

    if engine is None:
        engine = PS13DEngine()

    print(f"[1/3] Reconstructing PS1 low-poly 3D subject & 2D backdrop for '{input_path}'...")
    rendered_rgb, _, stats = process_image_array(
        orig_rgb, config=cfg, engine=engine, output_obj_path=output_obj_path
    )
    print(
        f"[2/3] Rasterized {stats['triangles']} subject triangles ({stats['vertices']} vertices, "
        f"{stats['faces_detected']} face(s)) at {stats['native_w']}x{stats['native_h']} native PS1 res."
    )

    out_dir = os.path.dirname(os.path.abspath(output_png_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(output_png_path, cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR))
    print(f"[3/3] Saved output to '{output_png_path}'.")
    return rendered_rgb


def main():
    parser = argparse.ArgumentParser(
        description="Recreate images in authentic PS1 Low-Poly 3D style (CLI or interactive Gradio Web UI)."
    )
    parser.add_argument("input", nargs="?", default=None, help="Input image file or folder path (default: foto1.jpg)")
    parser.add_argument("-i", "--input-file", dest="input_flag", default=None, help="Input image file or folder path")
    parser.add_argument("-o", "--output", default=None, help="Output PNG file or folder path (default: ps1_render.png)")
    parser.add_argument(
        "-p",
        "--preset",
        choices=list(PRESETS.keys()),
        default="mgs1",
        help="PS1 visual preset: mgs1 (default), balanced, ultra_lowpoly, detailed",
    )
    parser.add_argument(
        "--mode",
        choices=["full_scene", "isolated_subject"],
        default=None,
        help="Scene meshing mode: isolated_subject (default flat 2D backdrop + 3D subject) or full_scene",
    )
    parser.add_argument("--polys", type=int, default=None, help="Override target polygon count (e.g. 250..1000)")
    parser.add_argument("--tex-res", type=int, default=None, help="Override body PS1 VRAM texture resolution (e.g. 64, 96, 128)")
    parser.add_argument("--head-tex-res", type=int, default=None, help="Override dedicated head PS1 VRAM texture resolution (e.g. 56, 72, 96)")
    parser.add_argument("--colors", type=int, default=None, help="Override CLUT palette color count (e.g. 16, 22, 32)")
    parser.add_argument("--flatness", type=float, default=None, help="Body facet flatness blend 0.0..1.0")
    parser.add_argument("--face-preserve", type=float, default=None, help="Face preservation weight 0.0..1.0 (default 0.96)")
    parser.add_argument("--yaw", type=float, default=None, help="3D camera yaw angle in degrees (default ~1.2)")
    parser.add_argument("--no-scanlines", action="store_true", help="Disable CRT scanline effect")
    parser.add_argument("--obj", default=None, help="Optional path to export .obj 3D mesh")
    parser.add_argument("--gui", action="store_true", help="Launch the interactive Gradio Web UI")
    parser.add_argument("--port", type=int, default=7860, help="Port for Gradio Web UI when using --gui")

    args = parser.parse_args()

    if args.gui:
        from app import launch_ui

        launch_ui(server_port=args.port)
        return

    base = PRESETS[args.preset]
    cfg = PS1Config(**vars(base))
    if args.mode is not None:
        cfg.scene_mode = args.mode
    if args.polys is not None:
        cfg.poly_count = args.polys
    if args.tex_res is not None:
        cfg.tex_resolution = args.tex_res
    if args.head_tex_res is not None:
        cfg.head_tex_resolution = args.head_tex_res
    if args.colors is not None:
        cfg.clut_colors = args.colors
        cfg.head_clut_colors = max(20, args.colors)
    if args.flatness is not None:
        cfg.facet_flatness = float(np.clip(args.flatness, 0.0, 1.0))
    if args.face_preserve is not None:
        cfg.face_preservation = float(np.clip(args.face_preserve, 0.0, 1.0))
    if args.yaw is not None:
        cfg.camera_yaw_deg = args.yaw
    if args.no_scanlines:
        cfg.scanlines = False

    engine = PS13DEngine()
    input_path = args.input_flag or args.input or "foto1.jpg"

    if os.path.isdir(input_path):
        out_folder = args.output or os.path.join(input_path, "ps1_outputs")
        os.makedirs(out_folder, exist_ok=True)
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        files = [
            f for f in sorted(os.listdir(input_path))
            if os.path.splitext(f)[1].lower() in exts and not f.startswith("ps1_")
        ]
        for fname in files:
            in_file = os.path.join(input_path, fname)
            stem = os.path.splitext(fname)[0]
            out_file = os.path.join(out_folder, f"ps1_{stem}.png")
            process_image(in_file, out_file, output_obj_path=None, config=cfg, engine=engine)
    else:
        out_png = args.output or "ps1_render.png"
        obj_path = args.obj
        if obj_path is None and input_path == "foto1.jpg" and args.output is None:
            obj_path = "ps1_character.obj"
        process_image(input_path, out_png, output_obj_path=obj_path, config=cfg, engine=engine)


if __name__ == "__main__":
    main()
