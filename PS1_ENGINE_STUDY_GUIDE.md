# Comprehensive Study Guide: Mastering `ps1_3d_engine.py` & `app.py`

This guide breaks down [`ps1_3d_engine.py`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py) and [`app.py`](file:///c:/Users/yavuz/Ps1/app.py) from end to end—combining **Computer Vision** (monocular depth estimation, Haar cascades, GrabCut, Canny/Sobel/Scharr edge detection, k-means clustering) with **1994 Sony PlayStation (PS1) 3D Hardware Emulation** (Geometry Transformation Engine vertex snapping, affine texture mapping, flat/faceted normal lighting, ordering-table painter's algorithm, 4×4 Bayer ordered dithering, and 15-bit RGB555 color quantization).

---

## 1. High-Level Architecture & Pipeline Map

### Chronological Execution Flow
When you run [`ps1_3d_engine.py`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py), execution flows through [`main`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L1516-L1600) -> [`process_image`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L1478-L1513) -> [`process_image_array`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L1403-L1475), which orchestrates the 8 core stages of [`PS13DEngine`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L205-L1401):

```
[Raw Input Image (H x W x C)]
       │
       ▼
1. Input Normalization & Native 240p Resolution Setup
   └── ensure_rgb_uint8() + aspect-ratio scaling to (ps1_w, ps1_h) where short side = 240px
       │
       ├──► 2. Monocular Depth Estimation: PS13DEngine.compute_depth()
       │       └── MiDaS_small neural network -> full_depth (H, W) float32 in [0.0, 1.0]
       │
       ├──► 3. Face & Anatomical Cage Detection: PS13DEngine.detect_faces_and_landmarks()
       │       └── Haar Cascades + Sobel mouth finder -> List[FaceRegion] (42-pt UV cage + protected boxes)
       │
       ▼
   Downsample to PS1 Native Space: native_rgb (ps1_h, ps1_w, 3) & native_depth (ps1_h, ps1_w)
       │
       ├──► 4. Optional Subject Silhouette Extraction: PS13DEngine.extract_polygonal_foreground()
       │       └── (In isolated_subject mode) Otsu Depth + GrabCut + approxPolyDP -> poly_mask, poly_contours
       │
       ├──► 5. VRAM Texture & CLUT Synthesis: PS13DEngine.build_stylized_base_and_texture()
       │       └── Bilateral filter + k-means CLUT quantization + depth fog -> stylized_rgb, ps1_tex_rgb
       │
       ├──► 6. Facial Detail Weight Map: PS13DEngine.build_detail_and_head_weight_map()
       │       └── Gaussian ellipse gated by foreground depth -> detail_weight_map (ps1_h, ps1_w) in [0, 1]
       │
       ├──► 7. 3D Low-Poly Mesh Generation: PS13DEngine.generate_feature_mesh()
       │       └── Importance sampling + Delaunay + quad edge flipping -> vertices (N, 3), triangles (M, 3)
       │
       ├──► 8. 2D Backdrop Initialization: PS13DEngine.render_backdrop()
       │       └── Static 2D pre-rendered PS1 background plate -> backdrop_rgb (ps1_h, ps1_w, 3)
       │
       ▼
9. PS1 GTE + GPU Hardware Rasterization: PS13DEngine.render_ps1_hardware()
   ├── 3D surface normals + Lambertian dot-product lighting + dihedral crease detection
   ├── 3D camera yaw rotation + perspective projection + integer grid vertex snapping
   ├── Painter's Algorithm (back-to-front Z sort) + per-triangle affine warp (cv2.warpAffine)
   └── PS13DEngine.apply_ps1_vram_output(): 4x4 Bayer dithering -> 15-bit RGB555 -> Nearest-neighbor upscale + CRT scanlines
```

### Data Transformation Summary Table

| Pipeline Stage | Primary Data Structure | Shape & Dtype | Semantic Meaning |
| :--- | :--- | :--- | :--- |
| **Input** | `orig_rgb` | `(H, W, 3)` `uint8` | Clean 3-channel RGB image (capped at 1600px max dimension) |
| **Depth Map** | `full_depth` / `native_depth` | `(ps1_h, ps1_w)` `float32` | Normalized inverse depth: `1.0` = closest foreground, `0.0` = farthest background |
| **Face Anchors** | `faces` (`List[FaceRegion]`) | `landmarks`: `(42, 2)` `float32` | 2D screen coordinates of forehead, brow ridge, eye sockets, nose, lips, jaw |
| **VRAM Texture** | `ps1_tex_rgb` | `(ps1_h, ps1_w, 3)` `uint8` | Blocky, low-res (`tex_resolution`), color-quantized (CLUT) texture sheet |
| **3D Vertices** | `vertices` | `(N, 3)` `float32` | Columns `[x, y, z]` where `x, y` are 240p pixel coords and `z` is quantized depth `[0..depth_scale]` |
| **Mesh Topology** | `triangles` | `(M, 3)` `int32` | Triplets of vertex indices `[i0, i1, i2]` forming counter-clockwise triangles |
| **Framebuffer** | `framebuffer` -> `upscaled_rgb` | `(ps1_h*scale, ps1_w*scale, 3)` `uint8` | Final 15-bit RGB555 dithered & upscaled PS1 render |

---

## 2. Core Mathematical & Algorithmic Concepts to Research

### A. Computer Vision & Image Processing

1. **Monocular Depth Estimation (MiDaS)**
   - **What it is**: A deep convolutional neural network (`MiDaS_small` with an EfficientNet-Lite3 backbone) trained on multiple datasets to predict relative inverse depth from a single 2D RGB photograph.
   - **Why this script uses it**: A 2D photo has no 3D `Z` coordinate. MiDaS gives us a continuous `Z` relief field so we can extrude the 2D vertices into a 3D mesh, separate foreground subjects from background walls/sky, apply distance fog, and compute 3D surface normals for polygon lighting.
   - **Keywords to search**: *Monocular Depth Estimation, MiDaS Ranftl et al., Relative Inverse Depth, Scale-Invariant Depth Loss*.

2. **Viola-Jones Haar Cascade Classifiers**
   - **What it is**: A classical, ultra-fast object detection algorithm that slides rectangular wavelet filters (Haar-like features) across an integral image at multiple scales to detect faces and eyes.
   - **Why this script uses it**: Random triangles across an eye or mouth destroy human facial recognition. Detecting the face and eyes lets us place a structured 42-vertex anatomical cage (`uv_cage`) around the sockets and lips, and dial down flat polygon shading on the face.
   - **Keywords to search**: *Viola-Jones Object Detection, Haar-like Features, Integral Image, AdaBoost Cascade, `cv2.CascadeClassifier`*.

3. **Otsu's Bimodal Thresholding & GrabCut Segmentation**
   - **What it is**: Otsu's method automatically finds the threshold that minimizes intra-class variance in a bimodal histogram; GrabCut models foreground and background color distributions using Gaussian Mixture Models (GMMs) and solves for the minimum-energy cut on a pixel graph.
   - **Why this script uses it**: In `isolated_subject` mode, Otsu finds the initial depth split between the foreground person and the background wall, and GrabCut refines that mask along true color edges (hair, ears, shoulders).
   - **Keywords to search**: *Otsu's Thresholding Method, Rother et al. GrabCut, Gaussian Mixture Model (GMM) Graph Cut, Markov Random Fields*.

4. **Bilateral Filtering & Unsharp Masking**
   - **What it is**: Unlike a standard Gaussian blur that blurs across boundaries, a bilateral filter multiplies a spatial Gaussian kernel by a color-range Gaussian kernel so edges stay sharp while flat interiors become smooth.
   - **Why this script uses it**: Smoothing skin and clothing textures before k-means quantization prevents noisy "confetti" pixels in the PS1 texture, while smoothed depth (`depth_smooth`) prevents jagged 3D spikes.
   - **Keywords to search**: *Tomasi and Manduchi Bilateral Filter, Edge-Preserving Smoothing, Unsharp Masking*.

5. **Image Gradients (Sobel & Scharr Operators) & Canny Edge Detection**
   - **What it is**: Sobel and Scharr convolve the image with $3 \times 3$ derivative kernels to approximate horizontal ($\frac{\partial I}{\partial x}$) and vertical ($\frac{\partial I}{\partial y}$) intensity/depth gradients; Canny applies non-maximum suppression and hysteresis thresholding to extract 1-pixel-wide structural contours.
   - **Why this script uses it**: Low-poly meshes look terrible if vertices are placed on a uniform grid. By sampling vertices with probability proportional to gradient magnitude $\sqrt{G_x^2 + G_y^2}$, vertices naturally snap to silhouettes, folds, and ridges.
   - **Keywords to search**: *Sobel-Feldman Operator, Scharr Operator, Canny Edge Detector, Non-Maximum Suppression, Hysteresis Thresholding*.

---

### B. Computational Geometry & 3D Mesh Generation

6. **Ramer-Douglas-Peucker Polygon Simplification (`cv2.approxPolyDP`)**
   - **What it is**: A recursive curve-decimation algorithm that approximates a dense pixel contour with a coarse polygon using the fewest vertices possible within a maximum perpendicular distance tolerance $\varepsilon$.
   - **Why this script uses it**: Gives characters and clothing contours their signature chunky, straight-edged 1998 PlayStation silhouette instead of smooth, high-res curves.
   - **Keywords to search**: *Ramer-Douglas-Peucker Algorithm, Polyline Simplification, Hausdorff Distance*.

7. **Poisson-Disk / Minimum-Distance Importance Sampling**
   - **What it is**: A point-placement strategy where candidate points are drawn from a non-uniform probability distribution (`importance`), but each new point $(x, y)$ is accepted only if its Euclidean distance to all existing points is at least $d_{\min}$ (`_try_add_point`).
   - **Why this script uses it**: Prevents degenerate sliver triangles caused by vertices bunching up in high-contrast areas while still concentrating polygon density on important features.
   - **Keywords to search**: *Blue Noise Sampling, Poisson-Disk Sampling, Rejection Sampling, Adaptive Mesh Tessellation*.

8. **Delaunay Triangulation & Edge Flipping**
   - **What it is**: A triangulation of a 2D point set such that no vertex lies inside the circumcircle of any triangle, maximizing the minimum angle of all triangles.
   - **Why this script uses it**: `scipy.spatial.Delaunay` connects our scattered 2D points into a valid triangle mesh. Afterwards, [`_optimize_triangulation_edges`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L990-L1082) performs data-dependent edge flips on interior quads so triangle edges align with color and depth boundaries.
   - **Keywords to search**: *Delaunay Triangulation, Empty Circumcircle Property, Lawson's Edge Flip Algorithm, Data-Dependent Triangulation (Dyn, Levin, Rippa)*.

---

### C. Authentic PS1 Hardware Graphics (GTE & GPU)

9. **Surface Normals, Cross Products, & Lambertian Cosine Lighting**
   - **What it is**: Given a triangle's 3D vertices $V_0, V_1, V_2$, the cross product of two edges $\vec{e}_1 = V_1 - V_0$ and $\vec{e}_2 = V_2 - V_0$ yields the vector perpendicular to the triangle's face ($\vec{N} = \frac{\vec{e}_1 \times \vec{e}_2}{\|\vec{e}_1 \times \vec{e}_2\|}$). The dot product $\vec{N} \cdot \vec{L}$ measures how directly the face points toward the light source $\vec{L}$.
   - **Why this script uses it**: Produces flat/faceted polygon shading where each triangle on a jacket or limb has a distinct brightness based on its 3D tilt, plus dihedral crease darkening where two adjacent triangles meet at a sharp angle ($\cos\theta = \vec{N}_1 \cdot \vec{N}_2$).
   - **Keywords to search**: *3D Vector Cross Product, Surface Normal, Lambertian Reflectance, Flat Shading vs. Gouraud Shading, Dihedral Angle*.

10. **PS1 GTE Perspective Projection & Integer Vertex Snapping**
    - **What it is**: The PlayStation 1 CPU had no floating-point unit (FPU); its Geometry Transformation Engine (GTE) coprocessor used 16-bit fixed-point math and truncated projected 2D screen coordinates to integer pixel positions (`proj_x = round(proj_x)`).
    - **Why this script uses it**: Snapping projected vertices to the low-res 240p integer grid reproduces the unmistakable PS1 polygon "jitter" and crisp pixel-aligned triangle edges.
    - **Keywords to search**: *PlayStation 1 GTE (Geometry Transformation Engine), Fixed-Point Arithmetic, Subpixel Precision Lack, Vertex Jitter*.

11. **Affine Texture Mapping (No Perspective Correction) & Painter's Algorithm**
    - **What it is**: Because the PS1 GPU lacked a Z-buffer and lacked perspective-correct $1/z$ texture interpolation, it sorted polygons back-to-front using an Ordering Table (Painter's Algorithm) and interpolated texture coordinates $(u, v)$ linearly (affinely) in screen space.
    - **Why this script uses it**: Sorting triangles by mean rotated depth `z_rot` and warping each triangle patch via a $2 \times 3$ affine matrix (`cv2.getAffineTransform` + `cv2.warpAffine(..., flags=cv2.INTER_NEAREST)`) emulates how the PS1 GPU rasterized textured polygons.
    - **Keywords to search**: *Affine Texture Mapping vs. Perspective-Correct Texture Mapping, Painter's Algorithm, PS1 Ordering Table (OT)*.

12. **CLUT (Color Look-Up Table), 4×4 Bayer Ordered Dithering, & 15-Bit RGB555**
    - **What it is**: PS1 textures were stored in VRAM as 4-bit (16-color) or 8-bit (256-color) indexed palettes (CLUTs). The PS1 framebuffer stored 15-bit color (5 bits per channel = 32 levels per R, G, B instead of 256). Before truncating 8-bit shading to 5-bit, the hardware added a repeating $4 \times 4$ Bayer threshold matrix to break up color banding.
    - **Why this script uses it**: K-means palette quantization + `PS1_BAYER_4X4` ordered dithering + 31-level RGB555 quantization gives the exact grainy, stippled color texture of real PS1 hardware.
    - **Keywords to search**: *PS1 VRAM CLUT, Ordered Dithering, Bayer Matrix, 15-bit High Color (RGB555), Color Quantization via K-Means*.

---

## 3. Detailed Function-by-Function Walkthrough

### Global Constants & Data Structures

#### [`PS1_BAYER_4X4`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L17-L29)
- **Shape/Type**: `(4, 4)` `float32` array, values in `[-0.5, +0.4375]`.
- **Math Intuition**: The hardware PS1 $4 \times 4$ dither matrix contains integers $0 \dots 15$. Dividing by $16.0$ and subtracting $0.5$ centers the offsets around zero so dithering adds structured spatial noise without changing average image brightness.

#### [`PS1Config`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L33-L57) & [`PRESETS`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L59-L137)
- **Purpose**: Holds all tunable hyperparameters (`scene_mode`, `poly_count`, `native_short_side`, `tex_resolution`, `clut_colors`, `facet_flatness`, `face_preservation`, `lighting_strength`, `crease_strength`, `camera_yaw_deg`, `depth_scale`, `dither_strength`, `fog_strength`, `scanlines`).
- **Presets**: `"mgs1"`, `"balanced"`, `"ultra_lowpoly"`, and `"detailed"` instantiate pre-tuned [`PS1Config`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L33-L57) objects.

#### [`FaceRegion`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L140-L149)
- **Fields**:
  - `box`: `(fx, fy, fw, fh)` in native 240p coordinates.
  - `head_box`: `(hx1, hy1, hx2, hy2)` expanded by 8% horizontally and 14% vertically to cover hair and ears.
  - `body_box`: `(bx1, by1, bx2, by2)` estimated torso region below the head (`fx - 1.35*fw` to `fx + 2.35*fw`, extending `6.5*fh` downward).
  - `landmarks`: `(42, 2)` `float32` array of 2D anatomical vertices.
  - `eye_mouth_boxes`: List of 3 `[x1, y1, x2, y2]` bounding boxes protecting the left eye, right eye, and mouth from random interior vertices or edge flips.

#### [`_install_timm_fallback_if_missing`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L151-L180)
- **Purpose**: `intel-isl/MiDaS` imports `timm` at the top of its hub file even when loading `MiDaS_small` (which only uses `rwightman/gen-efficientnet-pytorch`, not `timm`). If `timm` isn't installed in your Python environment, this registers a lightweight `MetaPathFinder` stub so `MiDaS_small` loads cleanly without requiring extra dependencies.

#### [`ensure_rgb_uint8`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L182-L202)
- **Input**: `img: np.ndarray` (any shape: 2D grayscale, 3D single-channel, 4-channel RGBA, `float32` in `[0, 1]`, or `uint16`).
- **Return**: `(H, W, 3)` contiguous `uint8` RGB array.
- **Edge Cases Handled**: Alpha-blends RGBA images onto a dark charcoal (`RGB(36, 36, 36)`) background instead of dropping the alpha channel (which often contains garbage white/black pixels in transparent PNGs).

---

### Class [`PS13DEngine`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L205-L1401)

#### 1. [`__init__`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L206-L220) & [`_load_midas_model`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L221-L246)
- **What it does**: Selects CUDA if available (otherwise CPU), loads `MiDaS_small` and `small_transform` from local Torch Hub cache (`~/.cache/torch/hub/intel-isl_MiDaS_master`) if present or downloads it once, and initializes 4 OpenCV Haar cascades (`frontalface_alt2`, `frontalface_default`, `profileface`, and `eye`).

#### 2. [`compute_depth`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L248-L262)
- **Input**: `img_rgb: np.ndarray` `(H, W, 3)` `uint8`.
- **Return**: `depth: np.ndarray` `(H, W)` `float32` in `[0.0, 1.0]`.
- **Math Intuition**:
  - Runs `self.midas(input_batch)` under `torch.no_grad()` and bicubically interpolates the output tensor back to `(H, W)`.
  - Normalizes using robust percentiles (`1.0%` and `99.5%`) rather than `min()` and `max()`:
    $$\text{depth}_{\text{norm}} = \text{clip}\left(\frac{D - P_{1.0}}{P_{99.5} - P_{1.0} + 10^{-6}}, 0, 1\right)$$
  - **Pitfall Handled**: Using `np.percentile` prevents a single hot pixel or specular highlight from compressing the entire scene's depth range.

#### 3. [`detect_faces_and_landmarks`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L264-L493)
- **Input**: `orig_rgb (H, W, 3)`, `full_depth (H, W)`, `ps1_w: int`, `ps1_h: int`.
- **Return**: `List[FaceRegion]` (up to 3 faces sorted largest-first).
- **Step-by-Step Breakdown**:
  1. **Multi-Cascade Detection (lines 272–294)**: Runs `frontalface_alt2`, `frontalface_default`, and `profileface` on both raw grayscale and histogram-equalized grayscale (`cv2.equalizeHist`) so faces in dark/backlit photos are still found down to `5%` of image size.
  2. **Depth Validation & Non-Maximum Suppression (lines 298–321)**: Rejects false-positive face detections in background wallpaper/clouds if median depth `< 0.24`, and merges overlapping boxes with Intersection-over-Minimum-Area `> 0.35`.
  3. **Eye & Sobel Mouth Localization (lines 326–388)**: Detects left/right eyes inside the upper 62% of the face box to calibrate `left_eye_u`, `right_eye_u`, `center_u`, and `eye_v`. In the lower face strip (`0.68..0.90` of `bh`), computes horizontal Sobel derivatives `cv2.Sobel(..., 0, 1)` and takes `argmax` of row-wise edge energy to lock onto the lip parting line `mouth_v`.
  4. **42-Point Anatomical Cage (`uv_cage`, lines 396–443)**: Constructs normalized `(u, v)` coordinates forming orbital quads *around* the eyes (supra-brow ridge at `brow_top_v`, infra-orbital cheekbone ridge at `eye_bot_v`) and a mouth box *around* the lips, with **zero** vertices inside the pupils/sclera and **zero** vertices outside the head silhouette (`0.08 <= u <= 0.92`, `0.01 <= v <= 0.99`).

#### 4. [`build_detail_and_head_weight_map`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L495-L547)
- **Input**: `native_rgb (ps1_h, ps1_w, 3)`, `faces: List[FaceRegion]`, `depth (ps1_h, ps1_w)`, `cfg: PS1Config`.
- **Return**: `combined: np.ndarray` `(ps1_h, ps1_w)` `float32` in `[0.0, 1.0]`.
- **Math Intuition**:
  - Draws a filled ellipse (`1.0`) over each detected face, blurs it with `cv2.GaussianBlur`, and multiplies by a strict depth gate:
    $$\text{depth\_gate} = \mathbb{I}(\text{depth} \ge 0.35) \cdot \text{clip}\left(\frac{\text{depth} - (\text{face\_med\_d} - 0.12)}{0.07}, 0, 1\right)$$
  - **Pitfall Handled**: Without `depth_gate`, the Gaussian blur around the head ellipse would bleed onto the background wall/sky behind the ears, causing a circular halo artifact.

#### 5. [`_quantize_clut_kmeans`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L549-L572)
- **Input**: `img_rgb (H, W, 3)`, `n_colors: int`, `seed: int`, `mask: Optional[np.ndarray]`.
- **Return**: Quantized `(H, W, 3)` `uint8` image containing at most `n_colors` distinct RGB colors.
- **Math Intuition**: Runs `cv2.kmeans` with `KMEANS_PP_CENTERS` (Arthur & Vassilvitskii initialization) on `train_pixels` (restricted to `mask > 0` if provided), then assigns every pixel to its nearest cluster center in Euclidean RGB space:
  $$\text{nearest}(p) = \arg\min_{k \in \{1..K\}} \| I(p) - C_k \|_2^2$$

#### 6. [`build_stylized_base_and_texture`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L574-L646)
- **Input**: `orig_rgb`, `native_rgb`, `depth`, `cfg`, `faces`.
- **Return**: `(stylized_rgb, ps1_tex_rgb)` — both `(ps1_h, ps1_w, 3)` `uint8`.
- **Step-by-Step Breakdown**:
  1. Applies 2 passes of `cv2.bilateralFilter` (`d=5, sigmaColor=26, sigmaSpace=9`) to create `stylized_rgb` (used later to sample clean flat polygon colors).
  2. Applies an unsharp high-frequency detail boost (`detail * 85.0`) so eyes, eyebrows, and mouth lines survive downsampling.
  3. Downsamples to PS1 VRAM texture dimensions `(tex_w, tex_h)` (e.g., `128px` on the short side) using `cv2.INTER_AREA`.
  4. **Dual-CLUT Quantization (lines 612–624)**: Splits `tiny_scene` at `tiny_depth >= 0.35` so foreground subjects get their own `cfg.clut_colors` palette (e.g. 28 colors for skin/clothing) and background gets `cfg.bg_clut_colors` (e.g. 20 colors), preventing bright sky or wall colors from stealing palette slots from the character's face.
  5. Upsamples `tiny_combined` back to `(ps1_w, ps1_h)` with `cv2.INTER_NEAREST` so texels stay crisp and blocky, then blends smooth atmospheric distance fog on distant pixels (`depth < 0.35`).

#### 7. [`extract_polygonal_foreground`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L648-L726) & [`_subdivide_polygon_contour`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L728-L741)
- **Input**: `img_rgb`, `depth`, `faces`, `cfg`.
- **Return**: `(poly_mask, poly_list)` in `isolated_subject` mode, or `(None, None)` in `full_scene` mode.
- **Step-by-Step Breakdown**:
  - Computes Otsu threshold on `depth`, seeds `cv2.grabCut` with `GC_BGD`, `GC_PR_BGD`, `GC_PR_FGD`, and `GC_FGD` (locking the inner facial ellipse as definite foreground), cleans the mask with morphological closing/opening, and simplifies the outer contour into a chunky polygon via `cv2.approxPolyDP`.
  - `_subdivide_polygon_contour` linearly interpolates intermediate vertices along long straight polygon edges (`steps = ceil(dist / max_seg_len)`) so Delaunay triangulation doesn't bridge across long silhouette edges.

#### 8. [`generate_feature_mesh`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L743-L987)
- **Input**: `stylized_rgb`, `depth`, `poly_mask`, `poly_contours`, `faces`, `cfg`.
- **Return**: `vertices_3d (N, 3) float32`, `triangles (M, 3) int32`, `border_pin (N,) bool`.
- **Step-by-Step Breakdown**:
  1. **Target Vertex Budget (line 763)**: By Euler's formula for planar triangulations ($T \approx 2V - V_{\text{hull}} - 2$), setting `target_verts = round(cfg.poly_count * 0.54)` produces almost the exact requested `cfg.poly_count` triangles.
  2. **Minimum Distance Guard (`_try_add_point`, lines 771–782)**: Rejects any candidate point closer than `custom_min_d` to existing points.
  3. **Stage 1 — Boundary Seeding (lines 785–801)**: Seeds either the polygonal subject contour (`isolated_subject`) or the 4 outer image borders (`full_scene`).
  4. **Stage 2 — Face Landmarks & Exclusion Masks (lines 803–846)**: Adds the 42 `face.landmarks` (if they pass the foreground depth check) and builds `face_smooth_exclusion` (to keep random vertices off smooth cheeks/forehead) and `orbital_mouth_exclusion` (to protect eyes and lips).
  5. **Stage 3 — Canny Structural & Depth Contours (lines 848–900)**: Runs CLAHE + Canny on luminance and depth, simplifies contours with `cv2.approxPolyDP`, and enforces a *larger* minimum vertex spacing (`min_dist * 1.32`) on `body_roi_mask` so torso/jacket polygons stay broad and chunky.
  6. **Stage 4 — Importance Sampling (lines 902–939)**: Computes color gradient (`cv2.Scharr`) and depth gradient (`cv2.Sobel`):
     $$\text{importance} = (0.04 + 0.45 \cdot \nabla_{\text{color}} + 0.51 \cdot \nabla_{\text{depth}}) \cdot (0.25 + 0.95 \cdot \text{depth}^{1.1})$$
     Draws random candidates weighted by `prob = importance / sum(importance)` across 3 progressively tighter distance passes (`1.15`, `0.90`, `0.72`).
  7. **Stage 5 & 6 — Delaunay + Silhouette Carving + Edge Flipping (lines 940–967)**: Runs `Delaunay(points)`, removes exterior triangles in `isolated_subject` mode, and calls `_optimize_triangulation_edges`.
  8. **Stage 7 — 64-Step Quantized 3D Z Extrusion (lines 970–978)**: Smooths depth with a bilateral filter, quantizes $Z$ to 64 discrete steps (`round(z_raw * 64.0) / 64.0`) to mimic PS1 fixed-point Z coordinates, and scales by `cfg.depth_scale`.

#### 9. [`_optimize_triangulation_edges`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L990-L1082)
- **Input**: `points (N, 2)`, `triangles (M, 3)`, `lab_img`, `depth`, `protected_mask`, `passes=3`.
- **Return**: Optimized `tris (M, 3)` `int32`.
- **Math Intuition**:
  - Builds an adjacency table `edge_map` mapping every interior shared edge `(u, v)` to its two incident triangles `(u, v, a)` and `(v, u, b)`.
  - Checks if flipping the diagonal from `(u, v)` to `(a, b)` keeps both new triangles counter-clockwise (`_signed_area2 > 2.0` — i.e., the quad is strictly convex) and non-degenerate (`_min_cos_ok`).
  - Evaluates a 7-point sample variance of LAB color + depth inside each triangle (`_tri_variance_cost`). If flipping the diagonal reduces total within-triangle variance by at least 8% (`new_cost < old_cost * 0.92`), it flips the edge so the triangle boundary snaps onto the hairline, shoulder, or horizon edge.

#### 10. [`render_backdrop`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L1084-L1105)
- **Input**: `ps1_tex_rgb`, `depth`, `poly_mask`, `cfg`.
- **Return**: `framebuffer` base `(ps1_h, ps1_w, 3)` `uint8`.
- **What it does**: In `full_scene` mode (`poly_mask is None`), returns a copy of `ps1_tex_rgb`. In `isolated_subject` mode, softly blurs and quantizes the background region into `cfg.bg_clut_colors` so the background acts as a clean 2D pre-rendered backdrop plate.

#### 11. [`render_ps1_hardware`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L1107-L1352)
- **Input**: `stylized_rgb`, `ps1_tex_rgb`, `backdrop_rgb`, `detail_weight_map`, `vertices`, `triangles`, `border_pin`, `cfg`.
- **Return**: `(upscaled_rgb, wireframe_rgb)`.
- **Step-by-Step Breakdown**:
  1. **Surface Normals & Depth-Cliff Attenuation (lines 1142–1163)**:
     - Flips the $Y$ axis (`-e1[:, 1]`) because image $Y$ points downward while right-handed 3D $Y$ points upward, then takes the cross product `np.cross(e1_rh, e2_rh)` and normalizes to unit length.
     - Computes `z_span = max(z_tri) - min(z_tri)`. If a triangle bridges across a steep occlusion boundary (`z_span > 0.06`), `cliff_atten` smoothly drops to `0.0` so silhouette-bridging triangles never cast dark 3D shadows onto background walls or sky.
  2. **Lambertian Facet Lighting & Warm/Cool Tint (lines 1164–1179)**:
     - Computes `ndotl = normals @ light_dir` with `light_dir = normalize([-0.38, 0.44, 0.81])` (top-left-front key light).
     - Quantizes lighting delta to 10 discrete steps (`np.round(raw_delta * 10.0) / 10.0`) and damps lighting swing on facial triangles (`dw_i` from `detail_weight_map`).
  3. **Dihedral Crease Detection (lines 1181–1197)**:
     - For every shared edge between two coherent triangles $T_1, T_2$, computes the cosine of their dihedral angle $\cos\theta = \vec{N}_1 \cdot \vec{N}_2$ and fold intensity $\sqrt{1 - \cos\theta}$. Edges with `fold > 0.15` are recorded in `crease_edges` to darken jacket/limb folds.
  4. **3D Camera Yaw Projection & GTE Vertex Snapping (lines 1199–1233)**:
     - Rotates centered coordinates $(x_{\text{cam}}, z_{\text{rel}})$ around the $Y$-axis by `yaw = radians(cfg.camera_yaw_deg)`:
       $$x_{\text{rot}} = x_{\text{cam}}\cos\theta + z_{\text{rel}}\sin\theta, \quad z_{\text{rot}} = -x_{\text{cam}}\sin\theta + z_{\text{rel}}\cos\theta$$
     - Applies perspective division `persp = focal / (focal - 0.48 * z_rot)` and snaps to integer pixels (`np.round(proj_x / vertex_snap) * vertex_snap`).
  5. **Painter's Algorithm & Affine Triangle Rasterization (lines 1235–1340)**:
     - Sorts triangles back-to-front by `np.mean(z_rot[triangles], axis=1)`.
     - For each triangle, crops its bounding box, computes `flat_color` from the median of inlier pixels close to the triangle centroid color (preventing sky colors from bleeding into hair facets), and computes `warp_mat = cv2.getAffineTransform(src_pts, dst_pts)`.
     - Blends the nearest-neighbor affine-warped texture `warped_tex` with `flat_color` using `local_flatness` (`<= 0.08` on faces so eyes/lips stay sharp; `0.45..0.55` on body/jacket for chunky low-poly facets), multiplies by `facet_light_rgb[idx]`, draws dihedral crease lines, and composites into `framebuffer`.

#### 12. [`apply_ps1_vram_output`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L1354-L1375) & [`export_obj`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L1377-L1401)
- **VRAM Output**: Tiles [`PS1_BAYER_4X4`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L17-L29) across the `(nh, nw)` native framebuffer, adds `bayer * cfg.dither_strength`, quantizes each RGB channel to 32 discrete 5-bit levels (`round(dithered * 31.0 / 255.0) * (255.0 / 31.0)`), upscales by integer `cfg.upscale_factor` using `cv2.INTER_NEAREST`, and dims every $N$-th row by `0.87` for CRT scanlines.
- **Wavefront `.obj` Export**: Writes normalized 3D vertex positions `v x y z`, UV texture coordinates `vt u v` (flipping $v \to 1 - v$ for 3D software convention), and 1-indexed faces `f i1/i1 i2/i2 i3/i3` so you can open the mesh in Blender or MeshLab.

---

## 4. Deep-Dive Mathematical Derivations (Key Formulas Explained)

### A. Right-Handed 3D Surface Normals from 2D Image + Depth ([lines 1143–1152](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L1143-L1152))
In standard 2D image arrays, $x$ increases to the right and $y$ increases **downward**. In a right-handed 3D coordinate system where $+Z$ points toward the camera, $+Y$ must point **upward**.
Given triangle vertices $V_0, V_1, V_2$ with coordinates $(x_i, y_i, z_i)$:
1. We negate the $y$-component of the edge vectors:
   $$\vec{e}_1 = (x_1 - x_0,\; -(y_1 - y_0),\; z_1 - z_0), \quad \vec{e}_2 = (x_2 - x_0,\; -(y_2 - y_0),\; z_2 - z_0)$$
2. Their cross product is:
   $$\vec{N}_{\text{raw}} = \vec{e}_1 \times \vec{e}_2 = \begin{vmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \\ e_{1x} & e_{1y} & e_{1z} \\ e_{2x} & e_{2y} & e_{2z} \end{vmatrix}$$
3. Because all front-facing camera relief triangles must face the viewer ($N_z > 0$), line 1152 enforces consistent orientation:
   $$\hat{N} = \text{sign}(N_z) \frac{\vec{N}_{\text{raw}}}{\|\vec{N}_{\text{raw}}\|_2 + 10^{-6}}$$

### B. Signed Area & Convex Quad Diagonal Flipping ([lines 1025–1077](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L1025-L1077))
When two triangles $T_1 = (u, v, a)$ and $T_2 = (v, u, b)$ share an interior edge $(u, v)$, they form a 4-sided polygon $(u, b, v, a)$.
- You can only flip the diagonal from $(u, v)$ to $(a, b)$ if the quad is **strictly convex**; if the quad is concave (shaped like a dart/chevron), flipping the diagonal creates two overlapping, inverted triangles!
- The 2D cross product (`_signed_area2(p, q, r)`) computes twice the signed area of $\triangle pqr$:
  $$2 A_{\text{signed}}(p, q, r) = (q_x - p_x)(r_y - p_y) - (q_y - p_y)(r_x - p_x)$$
- Requiring `_signed_area2(pu, pb, pa) > 2.0` and `_signed_area2(pb, pv, pa) > 2.0` guarantees that both new triangles have positive counter-clockwise area and the quad is strictly convex.

### C. Affine Texture Mapping Matrix ([lines 1299–1309](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L1299-L1309))
Given 3 source UV coordinates $(u_i, v_i)$ in the texture crop and 3 destination screen coordinates $(x'_i, y'_i)$ after GTE projection and integer vertex snapping, `cv2.getAffineTransform` solves the $6 \times 6$ linear system for the unique $2 \times 3$ matrix $M$:
$$\begin{bmatrix} x'_i \\ y'_i \end{bmatrix} = \begin{bmatrix} m_{00} & m_{01} & m_{02} \\ m_{10} & m_{11} & m_{12} \end{bmatrix} \begin{bmatrix} u_i \\ v_i \\ 1 \end{bmatrix}, \quad i \in \{0, 1, 2\}$$
Unlike modern GPUs that divide $(u, v)$ by $z$ at every pixel, this affine matrix interpolates texels linearly across screen space—reproducing the exact affine texture warping of the PS1 GPU.

---

## 5. Companion Walkthrough: Gradio Web UI ([`app.py`](file:///c:/Users/yavuz/Ps1/app.py))

[`app.py`](file:///c:/Users/yavuz/Ps1/app.py) wraps [`ps1_3d_engine.py`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py) in an interactive browser interface:
- **[`get_engine`](file:///c:/Users/yavuz/Ps1/app.py#L21-L25)**: Lazily initializes a singleton [`PS13DEngine`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L205) instance (`_ENGINE_SINGLETON`) so the `MiDaS_small` neural network and Haar cascades stay loaded in GPU/CPU memory across slider tweaks.
- **[`on_preset_change`](file:///c:/Users/yavuz/Ps1/app.py#L28-L42)**: When you select `"mgs1"`, `"balanced"`, `"ultra_lowpoly"`, or `"detailed"`, it returns the 9 matching slider/dropdown values from [`PRESETS`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L59-L137).
- **[`run_ps1_pipeline`](file:///c:/Users/yavuz/Ps1/app.py#L45-L109)**: Copies the selected preset into a fresh [`PS1Config`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L33-L57), applies the UI slider overrides, calls [`process_image_array`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L1403-L1475), writes downloadable `.png` and `.obj` files to the system temp folder, and returns the rendered image, before/after comparison tuple `(orig_matched, rendered_rgb)`, and wireframe topology preview.

---

## 6. Recommended Learning Roadmap & Hands-On Experiments

### Step-by-Step Reading Order
1. **Pass 1 — Data & Orchestration (15 mins)**:
   Read [`PS1Config`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L33-L57), [`FaceRegion`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L140-L149), and [`process_image_array`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L1403-L1475). Keep [`process_image_array`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L1403-L1475) open in a split pane—it is the "table of contents" showing how all 8 engine methods connect.
2. **Pass 2 — The PS1 Hardware Post-Processor (20 mins)**:
   Read [`PS1_BAYER_4X4`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L17-L29), [`apply_ps1_vram_output`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L1354-L1375), [`_quantize_clut_kmeans`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L549-L572), and [`build_stylized_base_and_texture`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L574-L646). These are self-contained 2D pixel pipelines that are easy to visualize.
3. **Pass 3 — Depth & Facial Anatomy (30 mins)**:
   Read [`compute_depth`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L248-L262), [`detect_faces_and_landmarks`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L264-L493), and [`build_detail_and_head_weight_map`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L495-L547). Pay special attention to how normalized `(u, v)` coordinates in `uv_cage` map onto `fx + u * fw, fy + v * fh`.
4. **Pass 4 — 3D Mesh Generation & Edge Flipping (45 mins)**:
   Read [`extract_polygonal_foreground`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L648-L726), [`generate_feature_mesh`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L743-L987), and [`_optimize_triangulation_edges`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L990-L1082).
5. **Pass 5 — The Software 3D Rasterizer (45 mins)**:
   Study [`render_ps1_hardware`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L1107-L1352) line by line: normal calculation -> lighting -> camera yaw projection -> painter's sort -> per-triangle affine warp.

### Quick Interactive Debug Snippet to Inspect Intermediate Arrays
You can save and run this snippet in `c:\Users\yavuz\Ps1` to dump every intermediate stage of the pipeline as a visual PNG image:

```python
import cv2
import numpy as np
from ps1_3d_engine import PS13DEngine, PRESETS, ensure_rgb_uint8

engine = PS13DEngine()
cfg = PRESETS["mgs1"]
orig_rgb = ensure_rgb_uint8(cv2.cvtColor(cv2.imread("foto2.jpg"), cv2.COLOR_BGR2RGB))

# 1. Inspect MiDaS Depth Map
depth = engine.compute_depth(orig_rgb)
cv2.imwrite("debug_1_depth.png", (depth * 255).astype(np.uint8))

# 2. Inspect Native 240p + Face Landmarks Cage
ps1_w, ps1_h = 240, int(round(240 * orig_rgb.shape[0] / orig_rgb.shape[1]))
ps1_h += ps1_h % 2
faces = engine.detect_faces_and_landmarks(orig_rgb, depth, ps1_w, ps1_h)
native_rgb = cv2.resize(orig_rgb, (ps1_w, ps1_h), interpolation=cv2.INTER_AREA)
native_depth = cv2.resize(depth, (ps1_w, ps1_h), interpolation=cv2.INTER_AREA)

lm_vis = native_rgb.copy()
for f in faces:
    for (x, y) in f.landmarks:
        cv2.circle(lm_vis, (int(round(x)), int(round(y))), 1, (0, 255, 0), -1)
    for (x1, y1, x2, y2) in f.eye_mouth_boxes:
        cv2.rectangle(lm_vis, (x1, y1), (x2, y2), (255, 80, 80), 1)
cv2.imwrite("debug_2_landmarks.png", cv2.cvtColor(lm_vis, cv2.COLOR_RGB2BGR))

# 3. Inspect PS1 CLUT Texture & Head Detail Weight Map
stylized_rgb, ps1_tex_rgb = engine.build_stylized_base_and_texture(orig_rgb, native_rgb, native_depth, cfg, faces)
detail_w = engine.build_detail_and_head_weight_map(native_rgb, faces, native_depth, cfg)
cv2.imwrite("debug_3_ps1_tex.png", cv2.cvtColor(ps1_tex_rgb, cv2.COLOR_RGB2BGR))
cv2.imwrite("debug_4_head_weight.png", (detail_w * 255).astype(np.uint8))

# 4. Inspect Wireframe Mesh
verts, tris, pin = engine.generate_feature_mesh(stylized_rgb, native_depth, None, None, faces, cfg)
backdrop = engine.render_backdrop(ps1_tex_rgb, native_depth, None, cfg)
rendered, wireframe = engine.render_ps1_hardware(stylized_rgb, ps1_tex_rgb, backdrop, detail_w, verts, tris, pin, cfg)
cv2.imwrite("debug_5_wireframe.png", cv2.cvtColor(wireframe, cv2.COLOR_RGB2BGR))
```

### 3 High-Impact Experiments to Try While Studying
1. **See Why Orbital Quads Matter**: In [`detect_faces_and_landmarks`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L295-L296), temporarily force `return []` so no face landmarks or detail weights are used. Re-render `foto2.jpg` and compare `debug_5_wireframe.png`—you will immediately see Delaunay triangles slicing diagonally across the pupils and `facet_flatness` turning the eyes into flat geometric blocks.
2. **Exaggerate 3D Perspective & Vertex Jitter**: Run `python ps1_3d_engine.py foto2.jpg -o exp_yaw.png --yaw 8.0` and set `vertex_snap = 3.0` in [`PS1Config`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L52). You will see how the 2.5D relief mesh rotates in 3D space and how coarse GTE integer snapping warps polygon corners.
3. **Compare Edge Flipping On vs. Off**: In [`generate_feature_mesh`](file:///c:/Users/yavuz/Ps1/ps1_3d_engine.py#L967), comment out `triangles = self._optimize_triangulation_edges(...)` and inspect `debug_5_wireframe.png` around the hairline and shoulders to see how data-dependent edge flipping snaps triangle diagonals onto high-contrast boundaries.

