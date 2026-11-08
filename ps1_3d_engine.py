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
from PIL import Image
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
    """Configuration for authentic PS1 low-poly 3D reconstruction & hardware rendering."""

    scene_mode: str = "full_scene"      # "full_scene" (default continuous mesh) or "isolated_subject"
    poly_count: int = 440               # Target triangle count for chunky PS1 3D mesh
    native_short_side: int = 240        # Authentic PS1 240p resolution along the shorter image dimension
    upscale_factor: int = 3             # Integer nearest-neighbor upscale (e.g. 240p * 3 = 720p)
    tex_resolution: int = 128           # Unified PS1 VRAM texture page resolution (short side)
    head_tex_resolution: int = 128      # Retained for CLI/UI backwards compatibility (unified pass used)
    clut_colors: int = 28               # Colors in foreground PS1 CLUT palette (8..48)
    head_clut_colors: int = 28          # Retained for backwards compatibility
    bg_clut_colors: int = 20            # Colors in background/sky CLUT palette
    facet_flatness: float = 0.50        # Body/clothing flatness blend (0.45..0.55 for blocky PS1 facets)
    face_preservation: float = 0.95     # [0.0..1.0] Keeps face local_flatness <= 0.08 so features stay readable
    lighting_strength: float = 0.58     # Strong 3D facet normal lighting contrast on body/limbs
    crease_strength: float = 0.30       # Dihedral polygon fold darkening (0.25..0.35 for sharp jacket/limb folds)
    silhouette_simplify: float = 0.012  # Chunky polygonal silhouette coarseness
    camera_yaw_deg: float = 0.7         # Subtle 3D perspective yaw (attenuated on depth cliffs)
    depth_scale: float = 24.0           # 3D Z relief extrusion scale in native 240p space
    vertex_snap: float = 1.0            # PS1 GTE integer screen-space grid snap (in 240p pixels)
    dither_strength: float = 10.0       # Hardware 4x4 Bayer dithering spread before RGB555
    fog_strength: float = 0.22          # Smooth scalar depth fog on distant background (depth < 0.35)
    scanlines: bool = True              # Subtle CRT scanline pass aligned to native 240p rows
    seed: int = 42                      # Deterministic seed for reproducible meshes


PRESETS: Dict[str, PS1Config] = {
    "mgs1": PS1Config(
        scene_mode="full_scene",
        poly_count=440,
        native_short_side=240,
        tex_resolution=128,
        head_tex_resolution=128,
        clut_colors=28,
        head_clut_colors=28,
        bg_clut_colors=20,
        facet_flatness=0.50,
        face_preservation=0.95,
        lighting_strength=0.58,
        crease_strength=0.30,
        camera_yaw_deg=0.7,
        depth_scale=24.0,
        dither_strength=10.0,
        fog_strength=0.22,
        scanlines=True,
    ),
    "balanced": PS1Config(
        scene_mode="full_scene",
        poly_count=480,
        native_short_side=240,
        tex_resolution=132,
        head_tex_resolution=132,
        clut_colors=28,
        head_clut_colors=28,
        bg_clut_colors=20,
        facet_flatness=0.46,
        face_preservation=0.96,
        lighting_strength=0.54,
        crease_strength=0.26,
        camera_yaw_deg=0.6,
        depth_scale=22.0,
        dither_strength=9.5,
        fog_strength=0.20,
        scanlines=True,
    ),
    "ultra_lowpoly": PS1Config(
        scene_mode="full_scene",
        poly_count=280,
        native_short_side=216,
        tex_resolution=96,
        head_tex_resolution=96,
        clut_colors=20,
        head_clut_colors=20,
        bg_clut_colors=14,
        facet_flatness=0.55,
        face_preservation=0.93,
        lighting_strength=0.64,
        crease_strength=0.35,
        camera_yaw_deg=0.9,
        depth_scale=26.0,
        vertex_snap=1.0,
        dither_strength=11.5,
        fog_strength=0.26,
        scanlines=True,
    ),
    "detailed": PS1Config(
        scene_mode="full_scene",
        poly_count=680,
        native_short_side=256,
        tex_resolution=144,
        head_tex_resolution=144,
        clut_colors=32,
        head_clut_colors=32,
        bg_clut_colors=24,
        facet_flatness=0.45,
        face_preservation=0.97,
        lighting_strength=0.48,
        crease_strength=0.25,
        camera_yaw_deg=0.5,
        depth_scale=20.0,
        dither_strength=8.5,
        fog_strength=0.16,
        scanlines=False,
    ),
}


@dataclass
class FaceRegion:
    """Detected face/head box, subject torso box, & anatomical feature anchors in native 240p coordinates."""

    box: Tuple[int, int, int, int]                    # (fx, fy, fw, fh) face box in native coordinates
    head_box: Tuple[int, int, int, int]               # Expanded head+hair+ears box (hx1, hy1, hx2, hy2)
    body_box: Tuple[int, int, int, int]               # Estimated torso/clothing box below head (bx1, by1, bx2, by2)
    landmarks: np.ndarray                             # Structured (K, 2) anatomical PS1 head + silhouette cage vertices
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
    """Normalizes any input image array (grayscale, RGBA, float, uint16) to a clean 3-channel uint8 RGB array."""
    if img is None:
        raise ValueError("Input image is None.")
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
        bg = np.full_like(arr[:, :, :3], 36, dtype=np.float32)
        arr = np.clip(arr[:, :, :3].astype(np.float32) * alpha + bg * (1.0 - alpha), 0, 255).astype(np.uint8)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)

    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.05:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(arr[:, :, :3])


class PS13DEngine:
    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        _install_timm_fallback_if_missing()
        (
            self.depth_backend,
            self.depth_model,
            self.depth_processor,
            self.midas,
            self.transform,
        ) = self._load_depth_model()

        haar_dir = cv2.data.haarcascades
        self.face_cascade_alt2 = cv2.CascadeClassifier(os.path.join(haar_dir, "haarcascade_frontalface_alt2.xml"))
        self.face_cascade_def = cv2.CascadeClassifier(os.path.join(haar_dir, "haarcascade_frontalface_default.xml"))
        self.profile_cascade = cv2.CascadeClassifier(os.path.join(haar_dir, "haarcascade_profileface.xml"))
        self.eye_cascade = cv2.CascadeClassifier(os.path.join(haar_dir, "haarcascade_eye.xml"))

    def _load_depth_model(self):
        """
        Loads the depth estimation backend.
        Primary: Depth-Anything-V2-Small via Hugging Face transformers (sharp boundaries around limbs, hands, props).
        Automatic Fallback: MiDaS (intel-isl/MiDaS DPT_Hybrid or MiDaS_small).
        """
        # Patch huggingface-hub version metadata for transformers compatibility
        try:
            import importlib.metadata
            orig_meta = importlib.metadata.version
            importlib.metadata.version = lambda pkg: "0.35.0" if pkg == "huggingface-hub" else orig_meta(pkg)
        except Exception:
            pass

        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            print("Attempting to load Depth-Anything-V2-Small model...")
            model_id = "depth-anything/Depth-Anything-V2-Small-hf"
            try:
                proc = AutoImageProcessor.from_pretrained(model_id, local_files_only=True)
                model = AutoModelForDepthEstimation.from_pretrained(model_id, local_files_only=True).to(self.device).eval()
            except Exception:
                proc = AutoImageProcessor.from_pretrained(model_id)
                model = AutoModelForDepthEstimation.from_pretrained(model_id).to(self.device).eval()
            print("Successfully loaded Depth-Anything-V2-Small depth model.")
            return "depth_anything_v2", model, proc, None, None
        except Exception as e:
            print(f"Depth-Anything-V2 not available ({e}). Falling back to MiDaS...")

        # Fallback to MiDaS
        midas, transform = self._load_midas_model()
        return "midas", None, None, midas, transform

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
        if self.depth_backend == "depth_anything_v2":
            pil_img = Image.fromarray(img_rgb)
            inputs = self.depth_processor(images=pil_img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.depth_model(**inputs)
                post = self.depth_processor.post_process_depth_estimation(outputs, target_sizes=[img_rgb.shape[:2]])
                d = post[0]["predicted_depth"].cpu().numpy().astype(np.float32)
            p_low, p_high = np.percentile(d, [1.0, 99.5])
            depth = np.clip((d - p_low) / (p_high - p_low + 1e-6), 0.0, 1.0).astype(np.float32)
            return depth
        else:
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
        Detects faces across close-up portraits, half-body shots, and full-body outdoor photos (down to 5% of image size),
        validates against foreground depth, and builds a complete PS1 anatomical head cage + outer hair/ear protective ring
        so continuous full-scene triangulation never slices sky facets into the head.
        """
        orig_h, orig_w = orig_rgb.shape[:2]
        det_scale = min(1.0, 760.0 / float(min(orig_h, orig_w)))
        det_w = max(64, int(round(orig_w * det_scale)))
        det_h = max(64, int(round(orig_h * det_scale)))
        det_gray = cv2.cvtColor(cv2.resize(orig_rgb, (det_w, det_h), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        det_gray_eq = cv2.equalizeHist(det_gray)

        min_face = max(18, int(round(min(det_w, det_h) * 0.05)))
        raw_boxes = []
        for gray_src in (det_gray, det_gray_eq):
            for cascade, scale_f, min_n in (
                (self.face_cascade_alt2, 1.05, 3),
                (self.face_cascade_def, 1.07, 4),
                (self.profile_cascade, 1.07, 4),
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
        max_area = float(raw_boxes[0][2] * raw_boxes[0][3])
        merged_boxes = []
        for bx, by, bw, bh in raw_boxes:
            ox1 = int(np.clip(bx / det_scale, 0, orig_w - 1))
            oy1 = int(np.clip(by / det_scale, 0, orig_h - 1))
            ox2 = int(np.clip((bx + bw) / det_scale, ox1 + 1, orig_w))
            oy2 = int(np.clip((by + bh) / det_scale, oy1 + 1, orig_h))
            patch_depth = float(np.median(full_depth[oy1:oy2, ox1:ox2]))
            # Reject background detections (curtain wrinkles, walls, floors)
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

        # Full multi-character / multi-person support:
        # Detect ALL valid people across the frame (up to 8 people); do NOT clamp to 1 or 2 dominant subjects.
        # Filter out false-positive curtain/clothing folds when they lack eye confirmation or sufficient scale.
        valid_merged = []
        for bx, by, bw, bh in merged_boxes:
            roi_h = int(round(bh * 0.65))
            face_roi = det_gray_eq[by : by + roi_h, bx : bx + bw]
            min_e = max(5, int(round(bw * 0.12)))
            max_e = int(round(bw * 0.45))
            eyes_chk = self.eye_cascade.detectMultiScale(face_roi, 1.06, 3, minSize=(min_e, min_e), maxSize=(max_e, max_e)) if not self.eye_cascade.empty() else ()
            area = float(bw * bh)
            if len(eyes_chk) > 0 or (area >= 0.18 * max_area):
                valid_merged.append((bx, by, bw, bh))

        if valid_merged:
            merged_boxes = valid_merged[:8]
        else:
            merged_boxes = merged_boxes[:8]

        sx = ps1_w / float(det_w)
        sy = ps1_h / float(det_h)
        faces: List[FaceRegion] = []

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
            lu_out = max(0.06, left_eye_u - 0.14)
            ru_out = min(0.94, right_eye_u + 0.14)

            # Natural anatomical head cage strictly within natural head curve (matching custom2.png):
            # Forehead/hairline, supra-brow ridge, temples, infra-orbital cheekbones, nose, mouth, and tapered chin
            uv_cage = [
                # Angular forehead & hairline (chunky 4-segment top matching custom2.png)
                (0.16, 0.05),
                (0.36, 0.02),
                (cu, 0.01),
                (0.64, 0.02),
                (0.84, 0.05),
                # Supra-brow ridge (top of orbital quads — sits ABOVE eyebrows)
                (0.10, brow_top_v + 0.02),
                (lu_out, brow_top_v),
                (lu_in, brow_top_v + 0.01),
                (cu, brow_top_v + 0.01),
                (ru_in, brow_top_v + 0.01),
                (ru_out, brow_top_v),
                (0.90, brow_top_v + 0.02),
                # Mid-eye bridge & outer temples ONLY (nothing inside the iris/sclera!)
                (0.08, eye_v),
                (lu_in + 0.01, eye_v),
                (cu, eye_v),
                (ru_in - 0.01, eye_v),
                (0.92, eye_v),
                # Infra-orbital / cheekbone ridge (bottom of orbital quads — sits BELOW lower eyelids)
                (0.10, eye_bot_v + 0.02),
                (lu_out, eye_bot_v),
                (lu_in, eye_bot_v),
                (cu, eye_bot_v + 0.01),
                (ru_in, eye_bot_v),
                (ru_out, eye_bot_v),
                (0.90, eye_bot_v + 0.02),
                # Nose tip, nostril wings & angular cheek planes
                (0.12, nose_tip_v),
                (cu - 0.14, nose_tip_v + 0.01),
                (cu, nose_tip_v),
                (cu + 0.14, nose_tip_v + 0.01),
                (0.88, nose_tip_v),
                # Upper mouth box & outer jaw
                (0.15, mouth_top_v + 0.02),
                (cu - 0.17, mouth_top_v),
                (cu, mouth_top_v),
                (cu + 0.17, mouth_top_v),
                (0.85, mouth_top_v + 0.02),
                # Lower mouth box & angular gonial jaw corners
                (0.19, mouth_bot_v),
                (cu - 0.16, mouth_bot_v),
                (cu, mouth_bot_v + 0.01),
                (cu + 0.16, mouth_bot_v),
                (0.81, mouth_bot_v),
                # Angular chin & neck base
                (0.32, 0.96),
                (cu, 0.99),
                (0.68, 0.96),
            ]

            pts_native = []
            for u, v in uv_cage:
                px = float(np.clip(fx + u * fw, 1.0, ps1_w - 2.0))
                py = float(np.clip(fy + v * fh, 1.0, ps1_h - 2.0))
                pts_native.append([px, py])

            hx1 = int(np.clip(np.floor(fx - 0.25 * fw), 0, ps1_w - 2))
            hy1 = int(np.clip(np.floor(fy - 0.52 * fh), 0, ps1_h - 2))
            hx2 = int(np.clip(np.ceil(fx + 1.25 * fw), hx1 + 2, ps1_w))
            hy2 = int(np.clip(np.ceil(fy + 1.15 * fh), hy1 + 2, ps1_h))

            # Estimate torso/body bounding box below the head for higher subject vertex density
            bx1 = int(np.clip(np.floor(fx - 1.35 * fw), 0, ps1_w - 2))
            by1 = int(np.clip(np.floor(fy + 0.85 * fh), 0, ps1_h - 2))
            bx2 = int(np.clip(np.ceil(fx + 2.35 * fw), bx1 + 2, ps1_w))
            by2 = int(np.clip(np.ceil(fy + 6.50 * fh), by1 + 2, ps1_h))

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
                    body_box=(bx1, by1, bx2, by2),
                    landmarks=np.array(pts_native, dtype=np.float32),
                    eye_mouth_boxes=prot_boxes,
                )
            )

        return faces

    @staticmethod
    def build_detail_and_head_weight_map(
        native_rgb: np.ndarray,
        faces: List[FaceRegion],
        depth: np.ndarray,
        cfg: PS1Config,
    ) -> np.ndarray:
        """
        Builds a continuous [0, 1] texture-preservation weight map controlled by `cfg.face_preservation`:
          - Strict depth-gated weight over detected faces so background sky/walls around the head never get haloed
          - Leaves body, limbs, clothing, and terrain at low weight so they exhibit chunky PS1 low-poly facets
        """
        ps1_h, ps1_w = depth.shape
        head_weight = np.zeros((ps1_h, ps1_w), dtype=np.float32)

        if faces:
            for face in faces:
                fx, fy, fw, fh = face.box
                cx = int(round(fx + fw * 0.5))
                cy = int(round(fy + fh * 0.48))
                rx = max(4, int(round(fw * 0.48)))
                ry = max(4, int(round(fh * 0.54)))
                face_ellipse = np.zeros((ps1_h, ps1_w), dtype=np.float32)
                cv2.ellipse(face_ellipse, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
                sigma = max(1.5, min(fw, fh) * 0.08)
                face_ellipse = cv2.GaussianBlur(face_ellipse, (0, 0), sigmaX=sigma)

                # Strict foreground depth mask so `head_weight` NEVER spills onto walls/sky
                bx1 = int(np.clip(round(fx + 0.2 * fw), 0, ps1_w - 1))
                by1 = int(np.clip(round(fy + 0.2 * fh), 0, ps1_h - 1))
                bx2 = int(np.clip(round(fx + 0.8 * fw), bx1 + 1, ps1_w))
                by2 = int(np.clip(round(fy + 0.8 * fh), by1 + 1, ps1_h))
                face_med_d = float(np.median(depth[by1:by2, bx1:bx2]))
                depth_gate = (depth >= max(0.24, face_med_d - 0.28)).astype(np.float32) * np.clip((depth - (face_med_d - 0.20)) / 0.10, 0.0, 1.0)
                head_weight = np.maximum(head_weight, face_ellipse * depth_gate)

            combined = head_weight * float(np.clip(cfg.face_preservation, 0.0, 1.0))
        else:
            # Fallback when no Haar face fires: protect upper foreground focal head region (strictly depth >= 0.35)
            fg_zone = depth >= 0.48
            if np.any(fg_zone):
                ys, xs = np.where(fg_zone)
                y_min, y_max = int(ys.min()), int(ys.max())
                span_h = y_max - y_min
                if span_h > ps1_h * 0.25:
                    head_y2 = y_min + int(span_h * 0.32)
                    sub = fg_zone[y_min:head_y2, :]
                    head_weight[y_min:head_y2, :][sub] = 0.85
            sigma = max(2.0, min(ps1_h, ps1_w) * 0.02)
            head_weight = cv2.GaussianBlur(head_weight, (0, 0), sigmaX=sigma) * (depth >= 0.35).astype(np.float32)
            combined = head_weight * float(np.clip(cfg.face_preservation, 0.0, 1.0))

        return np.clip(combined, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _quantize_clut_kmeans(
        img_rgb: np.ndarray,
        n_colors: int,
        seed: int = 42,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Simulates a dedicated PS1 VRAM 4-bit/5-bit CLUT color palette via k-means in RGB space."""
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
        depth: np.ndarray,
        cfg: PS1Config,
        faces: List[FaceRegion],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Synthesizes authentic Dual-CLUT PS1 VRAM textures + smooth scalar depth fog (ZERO Telea inpainting!):
          1. Foreground CLUT (`depth >= 0.35` or inside subject boxes) + Background/Sky CLUT (`depth < 0.35`)
          2. Smooth atmospheric depth fog applied via continuous depth scalar (`depth < 0.35`)
          3. Dedicated Head / Focal VRAM Texture Page (`cfg.head_tex_resolution`) cropped from full-res source
        """
        nh, nw = native_rgb.shape[:2]
        oh, ow = orig_rgb.shape[:2]

        work = cv2.cvtColor(native_rgb, cv2.COLOR_RGB2BGR)
        for _ in range(2):
            work = cv2.bilateralFilter(work, d=5, sigmaColor=26, sigmaSpace=9)
        stylized_rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)

        gray = cv2.cvtColor(native_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        blur_g = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(1.1, min(nh, nw) / 115.0))
        detail = np.clip((gray - blur_g) / 255.0, -0.28, 0.22)
        boosted_scene = np.clip(stylized_rgb.astype(np.float32) + detail[:, :, None] * 85.0, 0, 255).astype(np.uint8)

        if nw <= nh:
            tex_w = max(32, cfg.tex_resolution)
            tex_h = max(32, int(round(tex_w * (nh / float(nw)))))
        else:
            tex_h = max(32, cfg.tex_resolution)
            tex_w = max(32, int(round(tex_h * (nw / float(nh)))))

        tiny_scene = cv2.resize(boosted_scene, (tex_w, tex_h), interpolation=cv2.INTER_AREA)
        tiny_depth = cv2.resize(depth, (tex_w, tex_h), interpolation=cv2.INTER_AREA)

        # Build continuous depth-based foreground vs. background domain masks (no rectangular box seams!)
        fg_domain = tiny_depth >= 0.35

        if 0.05 < float(np.mean(fg_domain)) < 0.92:
            fg_clut = self._quantize_clut_kmeans(
                tiny_scene, cfg.clut_colors, seed=cfg.seed, mask=fg_domain.astype(np.uint8)
            )
            bg_clut = self._quantize_clut_kmeans(
                tiny_scene, cfg.bg_clut_colors, seed=cfg.seed + 11, mask=(~fg_domain).astype(np.uint8)
            )
            tiny_combined = np.where(fg_domain[:, :, None], fg_clut, bg_clut)
        else:
            tiny_combined = self._quantize_clut_kmeans(tiny_scene, cfg.clut_colors, seed=cfg.seed)

        ps1_tex_rgb = cv2.resize(tiny_combined, (nw, nh), interpolation=cv2.INTER_NEAREST)

        # Apply smooth scalar PS1 distance fog on distant background (`depth < 0.35`) without rectangular box cutouts!
        if cfg.fog_strength > 0.01:
            bg_region = depth < 0.35
            if np.any(bg_region):
                fog_color = np.median(ps1_tex_rgb[bg_region], axis=0).astype(np.float32) * 0.85 + np.array(
                    [22.0, 26.0, 32.0], dtype=np.float32
                )
            else:
                fog_color = np.array([150.0, 160.0, 175.0], dtype=np.float32)

            fog_scalar = np.clip((0.35 - depth) / 0.35, 0.0, 1.0) ** 1.35
            fog_alpha = (fog_scalar * cfg.fog_strength)[:, :, None]
            ps1_tex_rgb = np.clip(
                ps1_tex_rgb.astype(np.float32) * (1.0 - fog_alpha) + fog_color[None, None, :] * fog_alpha, 0, 255
            ).astype(np.uint8)
            stylized_rgb = np.clip(
                stylized_rgb.astype(np.float32) * (1.0 - fog_alpha) + fog_color[None, None, :] * fog_alpha, 0, 255
            ).astype(np.uint8)

        return stylized_rgb, ps1_tex_rgb

    def extract_polygonal_foreground(
        self,
        img_rgb: np.ndarray,
        depth: np.ndarray,
        faces: List[FaceRegion],
        cfg: PS1Config,
    ) -> Tuple[Optional[np.ndarray], Optional[list]]:
        """
        Optional foreground segmentation used ONLY when `cfg.scene_mode == "isolated_subject"`.
        In `full_scene` mode (default), returns `(None, None)` so the entire scene is meshed continuously.
        """
        if cfg.scene_mode != "isolated_subject":
            return None, None

        h, w = depth.shape
        d_u8 = (depth * 255.0).astype(np.uint8)
        otsu_thresh, _ = cv2.threshold(d_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_val = float(otsu_thresh) / 255.0

        fg_init = depth >= thresh_val
        fg_ratio = float(np.mean(fg_init))
        if fg_ratio < 0.08 or fg_ratio > 0.90:
            return None, None
        fg_mean = float(depth[fg_init].mean())
        bg_mean = float(depth[~fg_init].mean())
        if (fg_mean - bg_mean) < 0.22:
            return None, None

        gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
        gc_mask[depth < max(0.06, thresh_val - 0.20)] = cv2.GC_BGD
        gc_mask[depth >= thresh_val] = cv2.GC_PR_FGD
        gc_mask[depth >= min(0.92, thresh_val + 0.16)] = cv2.GC_FGD

        for face in faces:
            fx, fy, fw, fh = face.box
            cx = int(round(fx + fw * 0.5))
            cy = int(round(fy + fh * 0.46))
            cv2.ellipse(
                gc_mask,
                (cx, cy),
                (max(3, int(round(fw * 0.36))), max(3, int(round(fh * 0.44)))),
                0, 0, 360, int(cv2.GC_FGD), -1,
            )

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        try:
            cv2.grabCut(bgr, gc_mask, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
            refined = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        except cv2.error:
            refined = (fg_init.astype(np.uint8)) * 255

        k_size = max(3, int(round(min(h, w) * 0.015)) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel)
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None

        min_area = 0.04 * (h * w)
        valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        if not valid_contours:
            return None, None

        poly_mask = np.zeros((h, w), dtype=np.uint8)
        poly_list = []
        body_eps = float(np.clip(min(h, w) * cfg.silhouette_simplify * 2.0, 1.8, 3.8))
        for cnt in valid_contours:
            approx = cv2.approxPolyDP(cnt, body_eps, True)
            if len(approx) >= 4:
                cv2.fillPoly(poly_mask, [approx], 255)
                poly_list.append(approx.reshape(-1, 2))

        if not poly_list:
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
        Builds a continuous full-scene (or isolated-subject) low-poly 3D mesh:
          - Pins the 4 outer frame borders in `full_scene` mode
          - Places dense anatomical & protective hair/ear silhouette vertices on faces
          - Concentrates vertices along high-contrast structural edges (character silhouette, horizon, ridges)
            and keeps flat open sky regions sparse and chunky
          - Flips quad diagonals to snap triangle edges along silhouettes so sky facets never cut into heads!
        """
        rng = np.random.default_rng(cfg.seed)
        depth = np.asarray(depth, dtype=np.float32)
        h, w = depth.shape
        target_verts = max(65, int(round(cfg.poly_count * 0.54)))

        active_area = float(np.sum(poly_mask > 0)) if poly_mask is not None else float(h * w)
        avg_tri_spacing = np.sqrt(active_area / float(max(30, target_verts)))
        min_dist = max(3.5, avg_tri_spacing * 0.52)

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

        # 1. Seed outer frame borders (in full_scene mode) or polygonal silhouette (in isolated_subject mode)
        if poly_mask is not None and poly_contours is not None:
            max_sil_seg = max(6.5, avg_tri_spacing * 0.85)
            for cnt in poly_contours:
                for pt in cnt:
                    _try_add_point(pt[0], pt[1], min_dist * 0.38)
                sub_pts = self._subdivide_polygon_contour(cnt, max_sil_seg)
                for pt in sub_pts:
                    _try_add_point(pt[0], pt[1], min_dist * 0.55)
        else:
            border_steps_x = max(10, int(round(w / (avg_tri_spacing * 1.05))))
            border_steps_y = max(10, int(round(h / (avg_tri_spacing * 1.05))))
            for x in np.linspace(0, w - 1, border_steps_x):
                _try_add_point(x, 0, 2.0)
                _try_add_point(x, h - 1, 2.0)
            for y in np.linspace(0, h - 1, border_steps_y):
                _try_add_point(0, y, 2.0)
                _try_add_point(w - 1, y, 2.0)

        # 2. Seed structured anatomical PS1 head cage landmarks & trace exact hair/head silhouette edges
        face_smooth_exclusion = np.zeros((h, w), dtype=np.uint8)
        orbital_mouth_exclusion = np.zeros((h, w), dtype=np.uint8)
        head_roi_mask = np.zeros((h, w), dtype=np.uint8)
        body_roi_mask = np.zeros((h, w), dtype=np.uint8)

        for face in faces:
            fx, fy, fw, fh = face.box
            hx1, hy1, hx2, hy2 = face.head_box
            bx1, by1, bx2, by2 = face.body_box
            head_roi_mask[hy1:hy2, hx1:hx2] = 255
            body_roi_mask[by1:by2, bx1:bx2] = 255

            icx1 = int(np.clip(round(fx + 0.2 * fw), 0, w - 1))
            icy1 = int(np.clip(round(fy + 0.2 * fh), 0, h - 1))
            icx2 = int(np.clip(round(fx + 0.8 * fw), icx1 + 1, w))
            icy2 = int(np.clip(round(fy + 0.8 * fh), icy1 + 1, h))
            face_med_d = float(np.median(depth[icy1:icy2, icx1:icx2]))

            face_min_d = max(1.8, min(fw, fh) * 0.08)
            for lm in face.landmarks:
                ix_l = int(np.clip(round(lm[0]), 0, w - 1))
                iy_l = int(np.clip(round(lm[1]), 0, h - 1))
                # Only seed landmark if it lies on the foreground head (never on background sky/wall!)
                if depth[iy_l, ix_l] >= (face_med_d - 0.14):
                    if poly_mask is None or poly_mask[iy_l, ix_l] > 0:
                        _try_add_point(lm[0], lm[1], face_min_d)

            cx = int(round(fx + fw * 0.5))
            cy = int(round(fy + fh * 0.48))
            cv2.ellipse(
                face_smooth_exclusion,
                (cx, cy),
                (max(4, int(round(fw * 0.42))), max(4, int(round(fh * 0.48)))),
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

        # 3. Seed high-contrast structural edges:
        #    Trace true hairline/ear/head silhouette contours inside `head_roi_mask`
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        eq_l = clahe.apply(cv2.cvtColor(stylized_rgb, cv2.COLOR_RGB2GRAY))
        edges_fine = cv2.Canny(eq_l, 45, 115)
        edges_Fine_head = cv2.bitwise_and(edges_fine, head_roi_mask)
        edges_Fine_head[orbital_mouth_exclusion > 0] = 0

        head_contours, _ = cv2.findContours(edges_Fine_head, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        head_contours = sorted(head_contours, key=lambda c: cv2.arcLength(c, False), reverse=True)
        for hc in head_contours:
            if cv2.arcLength(hc, False) < 5.0:
                continue
            approx_hc = cv2.approxPolyDP(hc, 1.8, False).reshape(-1, 2)
            for pt in approx_hc:
                px_i, py_i = int(round(pt[0])), int(round(pt[1]))
                if face_smooth_exclusion[py_i, px_i] == 0:
                    _try_add_point(pt[0], pt[1], max(2.4, min_dist * 0.48))

        # Next seed general scene & character silhouette contours with coarse, blocky PS1 polygonal segments
        edges_scene = cv2.Canny(eq_l, 75, 160)
        d_u8 = (depth * 255.0).astype(np.uint8)
        depth_edges = cv2.Canny(d_u8, 30, 80)
        combined_edges = cv2.bitwise_or(edges_scene, depth_edges)
        if poly_mask is not None:
            erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            combined_edges = cv2.bitwise_and(combined_edges, cv2.erode(poly_mask, erode_k))
        combined_edges[orbital_mouth_exclusion > 0] = 0

        edge_contours, _ = cv2.findContours(combined_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        edge_contours = sorted(edge_contours, key=lambda c: cv2.arcLength(c, False), reverse=True)
        edge_budget = int(target_verts * 0.65)
        for ec in edge_contours:
            if len(chosen_pts) >= edge_budget:
                break
            if cv2.arcLength(ec, False) < min_dist * 1.8:
                continue
            # Higher epsilon on approxPolyDP breaks body/torso contours into chunky, straight-edge polygonal segments
            approx_ec = cv2.approxPolyDP(ec, max(3.6, min_dist * 1.05), False).reshape(-1, 2)
            for pt in approx_ec:
                if len(chosen_pts) >= edge_budget:
                    break
                px_i, py_i = int(round(pt[0])), int(round(pt[1]))
                if face_smooth_exclusion[py_i, px_i] > 0:
                    local_d = min_dist * 1.30
                elif head_roi_mask[py_i, px_i] > 0:
                    local_d = min_dist * 0.62
                elif body_roi_mask[py_i, px_i] > 0:
                    # Larger minimum distance on torso/clothing so the mesh forms broad, blocky geometric quads/prisms
                    local_d = min_dist * 1.32
                else:
                    local_d = min_dist * 1.12
                _try_add_point(pt[0], pt[1], local_d)

        # 4. Adaptive Importance Sampling for remaining vertices:
        lab = cv2.cvtColor(stylized_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        gx = cv2.Scharr(eq_l.astype(np.float32), cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(eq_l.astype(np.float32), cv2.CV_32F, 0, 1)
        color_grad = np.hypot(gx, gy)
        color_grad /= (np.percentile(color_grad, 99.0) + 1e-6)

        dgx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
        dgy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
        depth_grad = np.hypot(dgx, dgy)
        depth_grad /= (np.percentile(depth_grad, 99.0) + 1e-6)

        importance = (0.04 + 0.45 * color_grad + 0.51 * depth_grad) * (0.25 + 0.95 * (depth ** 1.1))
        if np.any(body_roi_mask > 0):
            # Reduce random interior point density on clothing/torso so body facets stay broad and blocky
            importance[body_roi_mask > 0] *= 0.65
        if poly_mask is not None:
            importance[poly_mask == 0] = 0.0
        importance[face_smooth_exclusion > 0] = 0.0

        prob = importance.ravel().astype(np.float64)
        prob_sum = prob.sum()
        if prob_sum > 1e-8 and len(chosen_pts) < target_verts:
            prob /= prob_sum
            n_candidates = min(len(prob), max(2800, target_verts * 15))
            flat_indices = rng.choice(len(prob), size=n_candidates, replace=True, p=prob)
            cy_cand, cx_cand = np.divmod(flat_indices, w)
            for cur_scale in (1.15, 0.90, 0.72):
                if len(chosen_pts) >= target_verts:
                    break
                for cx_i, cy_i in zip(cx_cand, cy_cand):
                    if len(chosen_pts) >= target_verts:
                        break
                    jx = cx_i + rng.uniform(-0.35, 0.35)
                    jy = cy_i + rng.uniform(-0.35, 0.35)
                    body_mult = 1.25 if body_roi_mask[int(cy_i), int(cx_i)] > 0 else 1.0
                    _try_add_point(jx, jy, min_dist * cur_scale * body_mult)

        points = np.array(chosen_pts, dtype=np.float32)
        tri_delaunay = Delaunay(points)
        triangles = tri_delaunay.simplices.astype(np.int32)

        # Subdivide bridging edges across negative space valleys (e.g. between arm and torso or between subjects)
        p_depths = depth[
            np.clip(np.round(points[:, 1]).astype(int), 0, h - 1),
            np.clip(np.round(points[:, 0]).astype(int), 0, w - 1),
        ]
        valley_pts = []
        for tri in triangles:
            for i_a, i_b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                za, zb = float(p_depths[i_a]), float(p_depths[i_b])
                if min(za, zb) >= 0.42:
                    mid_x = 0.5 * (points[i_a, 0] + points[i_b, 0])
                    mid_y = 0.5 * (points[i_a, 1] + points[i_b, 1])
                    mid_z = float(
                        depth[int(np.clip(round(mid_y), 0, h - 1)), int(np.clip(round(mid_x), 0, w - 1))]
                    )
                    if (min(za, zb) - mid_z) > 0.20 and mid_z < 0.35:
                        valley_pts.append([mid_x, mid_y])
        if valley_pts:
            for vp in valley_pts:
                _try_add_point(vp[0], vp[1], min_dist * 0.45)
            points = np.array(chosen_pts, dtype=np.float32)
            tri_delaunay = Delaunay(points)
            triangles = tri_delaunay.simplices.astype(np.int32)

        # 5. If in isolated_subject mode, carve concave silhouette regions
        if poly_mask is not None:
            dil_3 = cv2.dilate(poly_mask, np.ones((3, 3), dtype=np.uint8))

            def _in_mask(pts_2d: np.ndarray, m: np.ndarray) -> np.ndarray:
                ix = np.clip(np.round(pts_2d[:, 0]).astype(int), 0, w - 1)
                iy = np.clip(np.round(pts_2d[:, 1]).astype(int), 0, h - 1)
                return m[iy, ix] > 0

            p0 = points[triangles[:, 0]]
            p1 = points[triangles[:, 1]]
            p2 = points[triangles[:, 2]]
            centroids = (p0 + p1 + p2) / 3.0

            keep = _in_mask(centroids, poly_mask)
            for t_val in (0.25, 0.50, 0.75):
                keep &= _in_mask(p0 * (1.0 - t_val) + p1 * t_val, dil_3)
                keep &= _in_mask(p1 * (1.0 - t_val) + p2 * t_val, dil_3)
                keep &= _in_mask(p2 * (1.0 - t_val) + p0 * t_val, dil_3)

            triangles = triangles[keep]

        # 6. Optimize interior quad diagonals to snap triangle edges along hair/head/horizon silhouettes
        triangles = self._optimize_triangulation_edges(points, triangles, lab, depth, orbital_mouth_exclusion)

        # 7. Edge-preserving 3D Z field with natural facial relief (nose bridge, brow ridge, cheek planes)
        depth_smooth = cv2.bilateralFilter(depth, d=5, sigmaColor=0.10, sigmaSpace=4.0)

        # Allow natural 3D depth extrusion for the nose bridge, brow ridge, and cheekbones relative to temples and ears
        depth_relief = depth_smooth.copy()
        for face in faces:
            fx, fy, fw, fh = face.box
            cx, cy = int(fx + 0.50 * fw), int(fy + 0.48 * fh)
            # Nose bridge subtle forward extrusion (+0.038)
            nose_y1, nose_y2 = int(fy + 0.36 * fh), int(fy + 0.68 * fh)
            nose_x1, nose_x2 = int(fx + 0.44 * fw), int(fx + 0.56 * fw)
            if nose_y2 > nose_y1 and nose_x2 > nose_x1:
                depth_relief[nose_y1:nose_y2, nose_x1:nose_x2] = np.maximum(
                    depth_relief[nose_y1:nose_y2, nose_x1:nose_x2],
                    depth_smooth[nose_y1:nose_y2, nose_x1:nose_x2] + 0.032,
                )
            # Supra-brow ridge forward extrusion (+0.024)
            brow_y1, brow_y2 = int(fy + 0.16 * fh), int(fy + 0.28 * fh)
            brow_x1, brow_x2 = int(fx + 0.22 * fw), int(fx + 0.78 * fw)
            if brow_y2 > brow_y1 and brow_x2 > brow_x1:
                depth_relief[brow_y1:brow_y2, brow_x1:brow_x2] = np.maximum(
                    depth_relief[brow_y1:brow_y2, brow_x1:brow_x2],
                    depth_smooth[brow_y1:brow_y2, brow_x1:brow_x2] + 0.022,
                )

        ix = np.clip(np.round(points[:, 0]).astype(int), 0, w - 1)
        iy = np.clip(np.round(points[:, 1]).astype(int), 0, h - 1)
        z_raw = np.clip(depth_relief[iy, ix], 0.0, 1.0)
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
        passes: int = 3,
    ) -> np.ndarray:
        """
        Flips interior quad diagonals when doing so reduces within-triangle color & depth variance,
        forcing triangle edges to align with hairlines, ears, shoulders, and horizon boundaries.
        """
        h, w = depth.shape
        tris = triangles.copy()
        feature_field = np.dstack([lab_img * 0.65, depth[:, :, None] * 75.0]).astype(np.float32)

        def _tri_variance_cost(i0: int, i1: int, i2: int) -> float:
            p0, p1, p2 = points[i0], points[i1], points[i2]
            samples = np.array(
                [
                    0.65 * p0 + 0.175 * p1 + 0.175 * p2,
                    0.175 * p0 + 0.65 * p1 + 0.175 * p2,
                    0.175 * p0 + 0.175 * p1 + 0.65 * p2,
                    0.50 * (p0 + p1),
                    0.50 * (p1 + p2),
                    0.50 * (p2 + p0),
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
            return max(c0, c1, c2) < 0.96

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
                if _signed_area2(pu, pb, pa) <= 2.0 or _signed_area2(pb, pv, pa) <= 2.0:
                    continue
                if not (_min_cos_ok(pu, pb, pa) and _min_cos_ok(pb, pv, pa)):
                    continue

                old_cost = _tri_variance_cost(u, v, a) + _tri_variance_cost(v, u, b)
                new_cost = _tri_variance_cost(u, b, a) + _tri_variance_cost(b, v, a)
                if new_cost < old_cost * 0.92:
                    tris[t1] = [u, b, a]
                    tris[t2] = [b, v, a]
                    locked_tris.add(t1)
                    locked_tris.add(t2)
                    existing_edges.add(new_key)

        return tris

    def render_backdrop(
        self,
        ps1_tex_rgb: np.ndarray,
        depth: np.ndarray,
        poly_mask: Optional[np.ndarray],
        cfg: PS1Config,
    ) -> np.ndarray:
        """
        Initializes the native 240p framebuffer prior to polygon rasterization.
        In `full_scene` mode (default), simply returns `ps1_tex_rgb` (which already has Dual-CLUT + smooth scalar depth fog)
        with ZERO destructive Telea inpainting!
        """
        if poly_mask is None:
            return ps1_tex_rgb.copy()

        # Only in `isolated_subject` mode: soften background behind cutout
        h, w = ps1_tex_rgb.shape[:2]
        bg = cv2.GaussianBlur(ps1_tex_rgb, (5, 5), 1.2)
        bg_quant = self._quantize_clut_kmeans(bg, cfg.bg_clut_colors, seed=cfg.seed + 19)
        fb = bg_quant.copy()
        fb[poly_mask > 0] = ps1_tex_rgb[poly_mask > 0]
        return fb

    def render_ps1_hardware(
        self,
        stylized_rgb: np.ndarray,
        ps1_tex_rgb: np.ndarray,
        backdrop_rgb: np.ndarray,
        detail_weight_map: np.ndarray,
        vertices: np.ndarray,
        triangles: np.ndarray,
        border_pin: np.ndarray,
        cfg: PS1Config,
        faces: Optional[List[FaceRegion]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Authentic PS1 GTE + GPU software rasterizer running natively at 240p.
        Returns (upscaled_ps1_rgb, wireframe_preview_rgb).
        """
        h, w = stylized_rgb.shape[:2]
        framebuffer = backdrop_rgb.copy()

        # Compute per-triangle detail/face preservation weight from vertices & centroid
        pts_tri_all = vertices[triangles, :2]
        centroids_2d = np.mean(pts_tri_all, axis=1)
        c_ix = np.clip(np.round(centroids_2d[:, 0]).astype(int), 0, w - 1)
        c_iy = np.clip(np.round(centroids_2d[:, 1]).astype(int), 0, h - 1)
        v0_ix = np.clip(np.round(pts_tri_all[:, 0, 0]).astype(int), 0, w - 1)
        v0_iy = np.clip(np.round(pts_tri_all[:, 0, 1]).astype(int), 0, h - 1)
        v1_ix = np.clip(np.round(pts_tri_all[:, 1, 0]).astype(int), 0, w - 1)
        v1_iy = np.clip(np.round(pts_tri_all[:, 1, 1]).astype(int), 0, h - 1)
        v2_ix = np.clip(np.round(pts_tri_all[:, 2, 0]).astype(int), 0, w - 1)
        v2_iy = np.clip(np.round(pts_tri_all[:, 2, 1]).astype(int), 0, h - 1)

        # Use centroid-weighted detail preservation so background triangles sharing 1 head vertex never get face weights
        tri_detail_w = 0.55 * detail_weight_map[c_iy, c_ix] + 0.15 * (
            detail_weight_map[v0_iy, v0_ix] + detail_weight_map[v1_iy, v1_ix] + detail_weight_map[v2_iy, v2_ix]
        )

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

        z_tri = vertices[triangles, 2] / max(1.0, cfg.depth_scale)
        z_span = np.max(z_tri, axis=1) - np.min(z_tri, axis=1)
        z_min = np.min(z_tri, axis=1)
        z_max = np.max(z_tri, axis=1)
        z_cent = np.mean(z_tri, axis=1)

        # Determine foreground subject threshold from detected characters vs background wall/sky
        if faces:
            face_depths = []
            for f in faces:
                cx = int(round(f.box[0] + f.box[2] * 0.5))
                cy = int(round(f.box[1] + f.box[3] * 0.45))
                v_dists = np.hypot(vertices[:, 0] - cx, vertices[:, 1] - cy)
                nearest_v = int(np.argmin(v_dists))
                patch_z = z_tri[np.any(triangles == nearest_v, axis=1)]
                if len(patch_z) > 0:
                    face_depths.append(float(np.median(patch_z)))
            if not face_depths:
                face_depths = [float(np.median(z_tri))]
            subject_depth_thresh = max(0.30, min(face_depths) - 0.28)
            is_depth_cliff = (z_span > 0.16) & (z_min < subject_depth_thresh)
            is_background = (z_max < subject_depth_thresh) | (z_cent < subject_depth_thresh)
        else:
            subject_depth_thresh = 0.22
            is_depth_cliff = (z_span > 0.20) & (z_min < 0.20)
            is_background = (z_max < 0.16)

        light_dir = np.array([-0.38, 0.44, 0.81], dtype=np.float32)
        light_dir /= np.linalg.norm(light_dir)
        ndotl = np.sum(normals * light_dir[None, :], axis=1)
        neutral_dot = float(np.median(ndotl))
        raw_delta = (ndotl - neutral_dot) * cfg.lighting_strength * 1.45
        raw_delta = np.round(raw_delta * 10.0) / 10.0

        warm_tint = np.array([1.02, 1.00, 0.97], dtype=np.float32)
        cool_tint = np.array([0.94, 0.98, 1.04], dtype=np.float32)
        facet_light_rgb = np.ones((len(triangles), 3), dtype=np.float32)
        for i, d_val in enumerate(raw_delta):
            dw_i = float(tri_detail_w[i])
            # Tone down facial normal lighting swing so chin/neck/cheeks don't fall into deep black shadows (matching custom2.png)
            max_swing = 0.12 * dw_i + 0.38 * (1.0 - dw_i)
            clamped_d = float(np.clip(d_val * (1.0 - 0.55 * dw_i), -max_swing, max_swing))
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
                if fold > 0.15:
                    crease_edges[k] = min(1.0, fold * 1.25)

        # 2. Tamed 3D Camera Projection + Yaw Parallax Attenuation on High Depth-Gradient Transition Edges
        cx, cy = w * 0.5, h * 0.5
        z_norm = vertices[:, 2] / max(1.0, cfg.depth_scale)
        # Estimate local depth gradient around each vertex from incident triangles to tame boundary parallax stretching
        v_max_span = np.zeros(len(vertices), dtype=np.float32)
        np.maximum.at(v_max_span, triangles[:, 0], z_span)
        np.maximum.at(v_max_span, triangles[:, 1], z_span)
        np.maximum.at(v_max_span, triangles[:, 2], z_span)
        parallax_atten = np.clip(1.0 - (v_max_span - 0.05) / 0.08, 0.0, 1.0)

        z_ref = float(np.median(vertices[:, 2]))
        x_cam = vertices[:, 0] - cx
        y_cam = vertices[:, 1] - cy
        z_rel = (vertices[:, 2] - z_ref) * parallax_atten

        yaw = np.radians(cfg.camera_yaw_deg)
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        x_rot = x_cam * cos_y + z_rel * sin_y
        z_rot = -x_cam * sin_y + z_rel * cos_y

        focal = max(w, h) * 4.2
        persp = focal / np.maximum(focal - z_rot * 0.48, focal * 0.68)
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
            # 1. Flat Background (No 3D Facets on Walls/Sky) & No Capes / Depth-Cliff Triangles:
            # If triangle belongs to the background or is an occlusion depth cliff bridging
            # character to background, SKIP drawing so the clean static 2D backdrop plate
            # (with circular ceiling light and uncreased walls) remains completely intact!
            if is_background[idx] or is_depth_cliff[idx]:
                continue

            tri = triangles[idx]
            src_tri = vertices[tri, :2].astype(np.float32)
            dst_tri = snapped_2d[tri].astype(np.float32)

            src_signed = (src_tri[1, 0] - src_tri[0, 0]) * (src_tri[2, 1] - src_tri[0, 1]) - (
                src_tri[1, 1] - src_tri[0, 1]
            ) * (src_tri[2, 0] - src_tri[0, 0])
            dst_signed = (dst_tri[1, 0] - dst_tri[0, 0]) * (dst_tri[2, 1] - dst_tri[0, 1]) - (
                dst_tri[1, 1] - dst_tri[0, 1]
            ) * (dst_tri[2, 0] - dst_tri[0, 0])
            # Prevent affine silhouette stretching on depth-cliff triangles or flipped triangles
            if src_signed * dst_signed <= 0 or float(z_span[idx]) > 0.10:
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

            dw_i = float(tri_detail_w[idx])
            src_mask = np.zeros((sh, sw), dtype=np.uint8)
            cv2.fillConvexPoly(src_mask, np.round(src_pts).astype(np.int32), 255, lineType=0)
            src_pixels = sty_crop[src_mask > 0]
            if len(src_pixels) > 0:
                # Sample centroid color and take median of pixels close to centroid color so boundary
                # triangles never blend sky blue into dark hair/clothing!
                c_local_x = int(np.clip(round(np.mean(src_pts[:, 0])), 0, sw - 1))
                c_local_y = int(np.clip(round(np.mean(src_pts[:, 1])), 0, sh - 1))
                center_col = sty_crop[c_local_y, c_local_x].astype(np.float32)
                px_f = src_pixels.astype(np.float32)
                col_dists = np.linalg.norm(px_f - center_col[None, :], axis=1)
                inliers = px_f[col_dists <= max(28.0, np.percentile(col_dists, 65.0))]
                flat_color = np.median(inliers if len(inliers) > 0 else px_f, axis=0).astype(np.float32)
                tri_std = float(np.mean(np.std(px_f, axis=0)))
            else:
                flat_color = sty_crop.mean(axis=(0, 1)).astype(np.float32)
                tri_std = 0.0

            # Rebalance local_flatness: 0.22 - 0.32 on the head so each broad planar facet
            # receives distinct 3D normal lighting and planar shading while the affine texture details remain readable
            edge_contrast_preserve = np.clip((tri_std - 24.0) / 36.0, 0.0, 1.0)
            base_body_flat = float(np.clip(cfg.facet_flatness, 0.45, 0.55)) if cfg.facet_flatness >= 0.35 else float(cfg.facet_flatness)
            body_flatness = base_body_flat * (1.0 - 0.30 * edge_contrast_preserve)
            face_flatness = float(np.clip(0.26 + 0.10 * (cfg.facet_flatness - 0.50), 0.22, 0.32))
            local_flatness = dw_i * face_flatness + (1.0 - dw_i) * body_flatness

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

            # Emphasize dihedral crease fold lines across jawline, cheek planes, and jacket folds (0.18 - 0.24 on face)
            local_crease = float(np.clip(cfg.crease_strength, 0.25, 0.35)) * (1.0 - 0.30 * dw_i)
            if local_crease > 0.01:
                i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
                for loc_a, loc_b, ga, gb in ((0, 1, i0, i1), (1, 2, i1, i2), (2, 0, i2, i0)):
                    ek = (min(ga, gb), max(ga, gb))
                    if ek in crease_edges:
                        edge_mask = np.zeros((dh, dw), dtype=np.uint8)
                        cv2.line(edge_mask, tuple(dst_pts_i[loc_a]), tuple(dst_pts_i[loc_b]), 255, 1, lineType=0)
                        darken = 1.0 - local_crease * crease_edges[ek] * 0.78
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

        # 4. Apply PS1 Hardware 4x4 Bayer Ordered Dithering & Exact 15-Bit RGB555 Quantization
        upscaled_rgb = self.apply_ps1_vram_output(framebuffer, cfg)

        # Build wireframe topology preview at upscaled resolution
        scale = max(1, int(cfg.upscale_factor))
        wireframe_rgb = (upscaled_rgb.astype(np.float32) * 0.78).astype(np.uint8)
        wf_native = np.zeros((h, w, 3), dtype=np.uint8)
        for idx in range(len(triangles)):
            if is_background[idx] or is_depth_cliff[idx]:
                continue
            tri = triangles[idx]
            p_tri = np.round(snapped_2d[tri]).astype(np.int32)
            cv2.polylines(wf_native, [p_tri], True, (0, 255, 128), 1, cv2.LINE_4)
        wf_up = cv2.resize(wf_native, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
        wf_mask = np.any(wf_up > 0, axis=2)
        wireframe_rgb[wf_mask] = wf_up[wf_mask]
        scaled_pts = np.round(snapped_2d * scale).astype(np.int32)
        for idx in range(len(triangles)):
            if is_background[idx] or is_depth_cliff[idx]:
                continue
            tri = triangles[idx]
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
        faces: Optional[List[FaceRegion]] = None,
    ) -> None:
        z_tri = vertices[triangles, 2] / max(1.0, depth_scale)
        z_span = np.max(z_tri, axis=1) - np.min(z_tri, axis=1)
        z_min = np.min(z_tri, axis=1)
        z_max = np.max(z_tri, axis=1)
        z_cent = np.mean(z_tri, axis=1)

        if faces:
            face_depths = []
            for f in faces:
                cx = int(round(f.box[0] + f.box[2] * 0.5))
                cy = int(round(f.box[1] + f.box[3] * 0.45))
                v_dists = np.hypot(vertices[:, 0] - cx, vertices[:, 1] - cy)
                nearest_v = int(np.argmin(v_dists))
                patch_z = z_tri[np.any(triangles == nearest_v, axis=1)]
                if len(patch_z) > 0:
                    face_depths.append(float(np.median(patch_z)))
            if not face_depths:
                face_depths = [float(np.median(z_tri))]
            subject_depth_thresh = max(0.30, min(face_depths) - 0.28)
            is_depth_cliff = (z_span > 0.16) & (z_min < subject_depth_thresh)
            is_background = (z_max < subject_depth_thresh) | (z_cent < subject_depth_thresh)
            valid_tri_mask = ~is_background & ~is_depth_cliff
            export_triangles = triangles[valid_tri_mask]
        else:
            export_triangles = triangles

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
            for tri in export_triangles:
                i1, i2, i3 = int(tri[0]) + 1, int(tri[1]) + 1, int(tri[2]) + 1
                f.write(f"f {i1}/{i1} {i2}/{i2} {i3}/{i3}\n")


def process_image_array(
    orig_rgb: np.ndarray,
    config: Optional[PS1Config] = None,
    engine: Optional[PS13DEngine] = None,
    output_obj_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Processes any in-memory RGB image array and returns (rendered_rgb, wireframe_rgb, stats_dict).
    Safely handles arbitrary aspect ratios, outdoor landscapes, portraits, and non-face images.
    """
    orig_rgb = ensure_rgb_uint8(orig_rgb)
    orig_h, orig_w = orig_rgb.shape[:2]
    if orig_h < 16 or orig_w < 16:
        raise ValueError(f"Image is too small ({orig_w}x{orig_h}); minimum supported size is 16x16.")

    max_dim = max(orig_h, orig_w)
    if max_dim > 1600:
        cap_scale = 1600.0 / float(max_dim)
        orig_rgb = cv2.resize(
            orig_rgb,
            (int(round(orig_w * cap_scale)), int(round(orig_h * cap_scale))),
            interpolation=cv2.INTER_AREA,
        )
        orig_h, orig_w = orig_rgb.shape[:2]

    cfg = config or PRESETS["mgs1"]
    if engine is None:
        engine = PS13DEngine()

    short_side = max(96, int(cfg.native_short_side))
    if orig_w <= orig_h:
        ps1_w = short_side
        ps1_h = max(96, min(640, int(round(short_side * (orig_h / float(orig_w))))))
    else:
        ps1_h = short_side
        ps1_w = max(96, min(640, int(round(short_side * (orig_w / float(orig_h))))))
    ps1_w += ps1_w % 2
    ps1_h += ps1_h % 2

    full_depth = engine.compute_depth(orig_rgb)
    faces = engine.detect_faces_and_landmarks(orig_rgb, full_depth, ps1_w, ps1_h)

    native_rgb = cv2.resize(orig_rgb, (ps1_w, ps1_h), interpolation=cv2.INTER_AREA)
    native_depth = cv2.resize(full_depth, (ps1_w, ps1_h), interpolation=cv2.INTER_AREA)

    poly_mask, poly_contours = engine.extract_polygonal_foreground(native_rgb, native_depth, faces, cfg)
    stylized_rgb, ps1_tex_rgb = engine.build_stylized_base_and_texture(
        orig_rgb, native_rgb, native_depth, cfg, faces
    )
    detail_weight_map = engine.build_detail_and_head_weight_map(native_rgb, faces, native_depth, cfg)

    vertices, triangles, border_pin = engine.generate_feature_mesh(
        stylized_rgb, native_depth, poly_mask, poly_contours, faces, cfg
    )
    backdrop_rgb = engine.render_backdrop(ps1_tex_rgb, native_depth, poly_mask, cfg)
    rendered_rgb, wireframe_rgb = engine.render_ps1_hardware(
        stylized_rgb, ps1_tex_rgb, backdrop_rgb, detail_weight_map, vertices, triangles, border_pin, cfg, faces=faces
    )

    if output_obj_path:
        out_obj_dir = os.path.dirname(os.path.abspath(output_obj_path))
        if out_obj_dir:
            os.makedirs(out_obj_dir, exist_ok=True)
        engine.export_obj(output_obj_path, vertices, triangles, ps1_w, ps1_h, cfg.depth_scale, faces=faces)

    stats = {
        "triangles": int(len(triangles)),
        "vertices": int(len(vertices)),
        "faces_detected": int(len(faces)),
        "native_w": int(ps1_w),
        "native_h": int(ps1_h),
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

    img_bgr = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img_bgr is None:
        raise ValueError(f"Could not decode image: {input_path}")
    if img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
        orig_rgb = ensure_rgb_uint8(cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2RGBA))
    elif img_bgr.ndim == 3:
        orig_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        orig_rgb = ensure_rgb_uint8(img_bgr)

    cfg = config or PRESETS["mgs1"]
    print(f"[1/3] Reconstructing PS1 low-poly 3D mesh ({cfg.scene_mode}) for '{input_path}'...")
    rendered_rgb, _, stats = process_image_array(
        orig_rgb, config=cfg, engine=engine, output_obj_path=output_obj_path
    )
    print(
        f"[2/3] Rasterized {stats['triangles']} triangles ({stats['vertices']} vertices, "
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
        help="Scene meshing mode: full_scene (default continuous mesh) or isolated_subject",
    )
    parser.add_argument("--polys", type=int, default=None, help="Override target polygon count (e.g. 250..1000)")
    parser.add_argument("--tex-res", type=int, default=None, help="Override body PS1 VRAM texture resolution (e.g. 64, 104, 128)")
    parser.add_argument("--head-tex-res", type=int, default=None, help="Override dedicated head PS1 VRAM texture resolution (e.g. 56, 76, 96)")
    parser.add_argument("--colors", type=int, default=None, help="Override CLUT palette color count (e.g. 16, 24, 32)")
    parser.add_argument("--flatness", type=float, default=None, help="Body facet flatness blend 0.0..1.0")
    parser.add_argument("--face-preserve", type=float, default=None, help="Face preservation weight 0.0..1.0 (default 0.95)")
    parser.add_argument("--yaw", type=float, default=None, help="3D camera yaw angle in degrees (default ~0.8)")
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
