import os
import tempfile
from typing import Optional

import cv2
import gradio as gr
import numpy as np

from ps1_3d_engine import PRESETS, PS13DEngine, PS1Config, ensure_rgb_uint8, process_image_array


SCENE_MODE_LABELS = {
    "Full Scene (Recommended for Landscapes/Outdoors)": "full_scene",
    "Isolated Subject (Studio/Dark Backgrounds)": "isolated_subject",
}
SCENE_MODE_REVERSE = {v: k for k, v in SCENE_MODE_LABELS.items()}

_ENGINE_SINGLETON: Optional[PS13DEngine] = None


def get_engine() -> PS13DEngine:
    global _ENGINE_SINGLETON
    if _ENGINE_SINGLETON is None:
        _ENGINE_SINGLETON = PS13DEngine()
    return _ENGINE_SINGLETON


def on_preset_change(preset_name: str):
    """Updates UI controls to match the chosen preset."""
    cfg = PRESETS.get(preset_name, PRESETS["mgs1"])
    mode_label = SCENE_MODE_REVERSE.get(cfg.scene_mode, "Full Scene (Recommended for Landscapes/Outdoors)")
    return (
        mode_label,
        cfg.poly_count,
        cfg.tex_resolution,
        cfg.crease_strength,
        cfg.clut_colors,
        cfg.facet_flatness,
        cfg.face_preservation,
        cfg.camera_yaw_deg,
        cfg.scanlines,
    )


def run_ps1_pipeline(
    input_img: Optional[np.ndarray],
    preset_name: str,
    scene_mode_label: str,
    poly_count: int,
    tex_res: int,
    crease_strength: float,
    clut_colors: int,
    facet_flatness: float,
    face_preservation: float,
    camera_yaw: float,
    scanlines: bool,
):
    if input_img is None:
        raise gr.Error("Please upload or select an input image first.")

    try:
        orig_rgb = ensure_rgb_uint8(input_img)
    except Exception as exc:
        raise gr.Error(f"Could not read input image: {exc}") from exc

    base_cfg = PRESETS.get(preset_name, PRESETS["mgs1"])
    cfg = PS1Config(**vars(base_cfg))
    cfg.scene_mode = SCENE_MODE_LABELS.get(scene_mode_label, "full_scene")
    cfg.poly_count = int(poly_count)
    cfg.tex_resolution = int(tex_res)
    cfg.crease_strength = float(np.clip(crease_strength, 0.0, 0.60))
    cfg.clut_colors = int(clut_colors)
    cfg.head_clut_colors = max(20, int(clut_colors))
    cfg.facet_flatness = float(np.clip(facet_flatness, 0.0, 1.0))
    cfg.face_preservation = float(np.clip(face_preservation, 0.0, 1.0))
    cfg.camera_yaw_deg = float(camera_yaw)
    cfg.scanlines = bool(scanlines)

    out_dir = os.path.join(tempfile.gettempdir(), "ps1_gradio_exports")
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, "ps1_render.png")
    obj_path = os.path.join(out_dir, "ps1_mesh.obj")

    try:
        engine = get_engine()
        rendered_rgb, wireframe_rgb, stats = process_image_array(
            orig_rgb, config=cfg, engine=engine, output_obj_path=obj_path
        )
        cv2.imwrite(png_path, cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR))
    except Exception as exc:
        raise gr.Error(f"PS1 rendering failed: {exc}") from exc

    out_h, out_w = rendered_rgb.shape[:2]
    orig_matched = cv2.resize(orig_rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)

    status_md = (
        f"**Render Complete (`{cfg.scene_mode}`)** — `{stats['triangles']}` triangles | "
        f"`{stats['vertices']}` 3D vertices | `{stats['native_w']}x{stats['native_h']}` native PS1 res "
        f"(`{out_w}x{out_h}` upscaled) | `{stats['faces_detected']}` face(s) preserved"
    )

    return (
        rendered_rgb,
        (orig_matched, rendered_rgb),
        wireframe_rgb,
        png_path,
        obj_path,
        status_md,
    )


def build_interface() -> gr.Blocks:
    default_cfg = PRESETS["mgs1"]
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    example_files = [
        os.path.join(workspace_dir, fname)
        for fname in ("foto1.jpg", "foto2.jpg", "foto4.jpg", "foto5.jpg", "foto6.jpg")
        if os.path.exists(os.path.join(workspace_dir, fname))
    ]

    with gr.Blocks(title="PS1 Low-Poly 3D Engine") as demo:
        gr.Markdown(
            """
            # PS1 Low-Poly 3D Character & Scene Engine
            Recreate portraits, outdoor photos, characters, and scenes in authentic **1998 PlayStation 1 (32-bit)** low-poly 3D style:
            Depth-Anything-V2 / MiDaS 3D continuous depth meshing, anatomical face & cranial cage preservation, Dual-CLUT VRAM textures,
            smooth scalar distance fog, 4x4 hardware Bayer dithering, and 15-bit RGB555 color quantization.
            """
        )

        with gr.Row():
            with gr.Column(scale=5):
                input_image = gr.Image(
                    label="Input Image (Drag & Drop or Click to Upload)",
                    type="numpy",
                    height=320,
                )

                with gr.Row():
                    preset_dropdown = gr.Dropdown(
                        choices=list(PRESETS.keys()),
                        value="mgs1",
                        label="PS1 Hardware Preset",
                    )
                    scene_mode_dropdown = gr.Dropdown(
                        choices=list(SCENE_MODE_LABELS.keys()),
                        value="Full Scene (Recommended for Landscapes/Outdoors)",
                        label="Scene Mode",
                    )

                render_btn = gr.Button("⚡ Render PS1 Low-Poly", variant="primary")

                with gr.Group():
                    with gr.Row():
                        poly_slider = gr.Slider(
                            minimum=120,
                            maximum=1200,
                            step=20,
                            value=default_cfg.poly_count,
                            label="Target Polygons (Triangles)",
                            info="Lower = chunkier 3D geometry; Higher = finer mesh subdivision.",
                        )
                        face_preserve_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            step=0.05,
                            value=default_cfg.face_preservation,
                            label="Face Preservation Weight",
                            info="Retains crisp affine CLUT texture over faces regardless of scene complexity.",
                        )
                    with gr.Row():
                        tex_res_slider = gr.Slider(
                            minimum=48,
                            maximum=160,
                            step=4,
                            value=default_cfg.tex_resolution,
                            label="Unified VRAM Texture Res",
                            info="Pixel resolution of the unified PS1 VRAM texture page.",
                        )
                        crease_slider = gr.Slider(
                            minimum=0.05,
                            maximum=0.50,
                            step=0.05,
                            value=default_cfg.crease_strength,
                            label="Dihedral Crease Strength",
                            info="Darkens sharp 3D low-poly fold lines across jackets, arms, and legs.",
                        )
                    with gr.Row():
                        colors_slider = gr.Slider(
                            minimum=8,
                            maximum=48,
                            step=2,
                            value=default_cfg.clut_colors,
                            label="CLUT Palette Colors",
                            info="Number of quantized colors in the PS1 VRAM CLUT palette.",
                        )
                        flatness_slider = gr.Slider(
                            minimum=0.0,
                            maximum=0.85,
                            step=0.05,
                            value=default_cfg.facet_flatness,
                            label="Facet Flatness (Body & Terrain)",
                            info="Flat polygon shading blend on non-face surfaces.",
                        )
                    with gr.Row():
                        yaw_slider = gr.Slider(
                            minimum=-4.0,
                            maximum=4.0,
                            step=0.2,
                            value=default_cfg.camera_yaw_deg,
                            label="3D Camera Yaw (°)",
                            info="Subtle 3D perspective rotation angle.",
                        )
                        scanlines_toggle = gr.Checkbox(
                            value=default_cfg.scanlines,
                            label="CRT Scanlines",
                            info="Apply authentic 240p horizontal CRT scanline shading.",
                        )

                if example_files:
                    gr.Examples(
                        examples=[[f] for f in example_files],
                        inputs=[input_image],
                        label="Quick-Load Workspace Sample Images",
                    )

            with gr.Column(scale=6):
                status_box = gr.Markdown("*Upload an image and click **Render PS1 Low-Poly** to begin.*")

                with gr.Tabs():
                    with gr.TabItem("🖼️ PS1 Render Output"):
                        output_image = gr.Image(label="Authentic PS1 240p Render", type="numpy", height=460)
                    with gr.TabItem("↔️ Before / After Comparison"):
                        comparison_slider = gr.ImageSlider(
                            label="Original vs. PS1 Low-Poly (Drag Divider)",
                            type="numpy",
                            height=460,
                        )
                    with gr.TabItem("🕸️ 3D Low-Poly Mesh Wireframe"):
                        wireframe_image = gr.Image(
                            label="Continuous Delaunay + Anatomical Face Cage Topology",
                            type="numpy",
                            height=460,
                        )

                with gr.Row():
                    download_png = gr.File(label="📥 Download Rendered PNG (.png)")
                    download_obj = gr.File(label="📦 Download 3D Low-Poly Mesh (.obj)")

        preset_dropdown.change(
            fn=on_preset_change,
            inputs=[preset_dropdown],
            outputs=[
                scene_mode_dropdown,
                poly_slider,
                tex_res_slider,
                crease_slider,
                colors_slider,
                flatness_slider,
                face_preserve_slider,
                yaw_slider,
                scanlines_toggle,
            ],
        )

        render_inputs = [
            input_image,
            preset_dropdown,
            scene_mode_dropdown,
            poly_slider,
            tex_res_slider,
            crease_slider,
            colors_slider,
            flatness_slider,
            face_preserve_slider,
            yaw_slider,
            scanlines_toggle,
        ]
        render_outputs = [
            output_image,
            comparison_slider,
            wireframe_image,
            download_png,
            download_obj,
            status_box,
        ]

        render_btn.click(fn=run_ps1_pipeline, inputs=render_inputs, outputs=render_outputs)
        input_image.upload(fn=run_ps1_pipeline, inputs=render_inputs, outputs=render_outputs)

    return demo


def launch_ui(server_name: str = "127.0.0.1", server_port: int = 7860, share: bool = False):
    demo = build_interface()
    demo.launch(server_name=server_name, server_port=server_port, share=share)


if __name__ == "__main__":
    launch_ui()
