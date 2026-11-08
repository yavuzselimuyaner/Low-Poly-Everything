"""
PS1 3D Engine — Interactive Study & Inspection Script
Run this script to step through the pipeline stages, print intermediate tensor shapes,
and save visual diagnostic images into the 'study_visualizations/' folder.
"""

import os
import cv2
import numpy as np
from ps1_3d_engine import (
    PS13DEngine,
    PRESETS,
    ensure_rgb_uint8,
    PS1_BAYER_4X4,
)

def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "study_visualizations")
    os.makedirs(out_dir, exist_ok=True)
    input_path = "foto2.jpg"

    print("=" * 70)
    print("PS1 3D ENGINE — STEP-BY-STEP STUDY WALKTHROUGH")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # STAGE 0: Load and Normalize Raw Image
    # -------------------------------------------------------------------------
    raw_bgr = cv2.imread(input_path)
    if raw_bgr is None:
        raise FileNotFoundError(f"Could not find '{input_path}'. Place an image in this directory.")
    orig_rgb = ensure_rgb_uint8(cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB))
    orig_h, orig_w = orig_rgb.shape[:2]
    print(f"\n[Stage 0] Input Loaded: {orig_w}x{orig_h} (dtype={orig_rgb.dtype}, range=[{orig_rgb.min()}, {orig_rgb.max()}])")

    # -------------------------------------------------------------------------
    # STAGE 1: Engine Initialization & Depth Estimation (MiDaS)
    # -------------------------------------------------------------------------
    engine = PS13DEngine()
    cfg = PRESETS["mgs1"]

    full_depth = engine.compute_depth(orig_rgb)
    print(f"\n[Stage 1] Depth Computed: shape={full_depth.shape}, min={full_depth.min():.4f}, max={full_depth.max():.4f}")
    
    # Save depth visualization (scaled to 0..255)
    depth_vis = (full_depth * 255.0).astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
    cv2.imwrite(os.path.join(out_dir, "01_depth_gray.png"), depth_vis)
    cv2.imwrite(os.path.join(out_dir, "01_depth_magma.png"), depth_colormap)
    print("  -> Saved: 01_depth_gray.png & 01_depth_magma.png")

    # -------------------------------------------------------------------------
    # STAGE 2: Aspect-Ratio Scaling to 240p Native Resolution
    # -------------------------------------------------------------------------
    short_side = max(96, int(cfg.native_short_side))
    if orig_w <= orig_h:
        ps1_w = short_side
        ps1_h = max(96, min(640, int(round(short_side * (orig_h / float(orig_w))))))
    else:
        ps1_h = short_side
        ps1_w = max(96, min(640, int(round(short_side * (orig_w / float(orig_h))))))
    ps1_w += ps1_w % 2
    ps1_h += ps1_h % 2
    print(f"\n[Stage 2] Native 240p Grid: {ps1_w}x{ps1_h} (short_side={short_side})")

    native_rgb = cv2.resize(orig_rgb, (ps1_w, ps1_h), interpolation=cv2.INTER_AREA)
    native_depth = cv2.resize(full_depth, (ps1_w, ps1_h), interpolation=cv2.INTER_AREA)

    # -------------------------------------------------------------------------
    # STAGE 3: Face Detection & Anatomical Head Cage
    # -------------------------------------------------------------------------
    faces = engine.detect_faces_and_landmarks(orig_rgb, full_depth, ps1_w, ps1_h)
    print(f"\n[Stage 3] Faces Detected: {len(faces)}")

    landmarks_canvas = cv2.cvtColor(native_rgb.copy(), cv2.COLOR_RGB2BGR)
    for i, face in enumerate(faces):
        fx, fy, fw, fh = face.box
        print(f"  Face #{i+1}: box=({fx}, {fy}, {fw}, {fh}), landmarks={len(face.landmarks)} points")
        
        # Draw face bounding box (yellow)
        cv2.rectangle(landmarks_canvas, (fx, fy), (fx + fw, fy + fh), (0, 230, 255), 1)
        # Draw torso estimate (cyan)
        bx1, by1, bx2, by2 = face.body_box
        cv2.rectangle(landmarks_canvas, (bx1, by1), (bx2, by2), (255, 200, 0), 1)
        # Draw protected eye & mouth boxes (red)
        for (x1, y1, x2, y2) in face.eye_mouth_boxes:
            cv2.rectangle(landmarks_canvas, (x1, y1), (x2, y2), (0, 0, 255), 1)
        # Draw anatomical cage landmark points (green)
        for pt in face.landmarks:
            cv2.circle(landmarks_canvas, (int(round(pt[0])), int(round(pt[1]))), 1, (0, 255, 0), -1)

    cv2.imwrite(os.path.join(out_dir, "02_face_anatomical_cage.png"), landmarks_canvas)
    print("  -> Saved: 02_face_anatomical_cage.png")

    # -------------------------------------------------------------------------
    # STAGE 4: Foreground Mask (if isolated_subject mode) & Base Textures
    # -------------------------------------------------------------------------
    poly_mask, poly_contours = engine.extract_polygonal_foreground(native_rgb, native_depth, faces, cfg)
    if poly_mask is not None:
        cv2.imwrite(os.path.join(out_dir, "03_polygonal_silhouette.png"), poly_mask)
        print(f"\n[Stage 4] Polygonal Silhouette Extracted: active_pixels={np.sum(poly_mask > 0)}")
    else:
        print(f"\n[Stage 4] Full-Scene Mode Active (poly_mask is None, whole scene will be meshed)")

    stylized_rgb, ps1_tex_rgb = engine.build_stylized_base_and_texture(
        orig_rgb, native_rgb, native_depth, cfg, faces
    )
    detail_weight_map = engine.build_detail_and_head_weight_map(native_rgb, faces, native_depth, cfg)

    cv2.imwrite(os.path.join(out_dir, "04_stylized_bilateral_rgb.png"), cv2.cvtColor(stylized_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, "04_ps1_clut_texture.png"), cv2.cvtColor(ps1_tex_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, "04_face_detail_weight_map.png"), (detail_weight_map * 255.0).astype(np.uint8))
    print("  -> Saved: 04_stylized_bilateral_rgb.png, 04_ps1_clut_texture.png, 04_face_detail_weight_map.png")

    # -------------------------------------------------------------------------
    # STAGE 5: 3D Low-Poly Mesh Generation & Edge Optimization
    # -------------------------------------------------------------------------
    vertices, triangles, border_pin = engine.generate_feature_mesh(
        stylized_rgb, native_depth, poly_mask, poly_contours, faces, cfg
    )
    print(f"\n[Stage 5] Low-Poly 3D Mesh Generated:")
    print(f"  Vertices:  {len(vertices)} points [x, y, z]")
    print(f"  Triangles: {len(triangles)} faces (Delaunay + Quad Edge Flipped)")
    print(f"  Pinned:    {np.sum(border_pin)} outer border vertices")

    # -------------------------------------------------------------------------
    # STAGE 6: PS1 Hardware Rasterization (GTE Snapping, Affine Warp, Lighting)
    # -------------------------------------------------------------------------
    backdrop_rgb = engine.render_backdrop(ps1_tex_rgb, native_depth, poly_mask, cfg)
    rendered_rgb, wireframe_rgb = engine.render_ps1_hardware(
        stylized_rgb,
        ps1_tex_rgb,
        backdrop_rgb,
        detail_weight_map,
        vertices,
        triangles,
        border_pin,
        cfg,
    )
    cv2.imwrite(os.path.join(out_dir, "05_rendered_ps1_final.png"), cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, "05_mesh_wireframe_overlay.png"), cv2.cvtColor(wireframe_rgb, cv2.COLOR_RGB2BGR))
    print(f"\n[Stage 6] Hardware Rasterization Complete:")
    print(f"  Output Canvas: {rendered_rgb.shape[1]}x{rendered_rgb.shape[0]} (Upscale Factor={cfg.upscale_factor}x)")
    print("  -> Saved: 05_rendered_ps1_final.png & 05_mesh_wireframe_overlay.png")

    # -------------------------------------------------------------------------
    # STAGE 7: 3D Wavefront .OBJ Export
    # -------------------------------------------------------------------------
    obj_path = os.path.join(out_dir, "ps1_character_study.obj")
    engine.export_obj(obj_path, vertices, triangles, ps1_w, ps1_h, cfg.depth_scale)
    print(f"\n[Stage 7] Wavefront 3D Mesh Exported:")
    print(f"  -> File: {obj_path} (can be imported into Blender/MeshLab)")

    print("\n" + "=" * 70)
    print(f"All 8 study diagnostics written to: {out_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()

