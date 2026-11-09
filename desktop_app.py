"""
PS1 Low-Poly 3D Studio - Native Desktop Application (PySide6)
Reconstructs photos into authentic 1998 PlayStation 1 3D character meshes and pre-rendered 2D backdrops.
"""

import os
import sys
import tempfile
import traceback
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QSize, Qt, QThread, Signal
from PySide6.QtGui import QAction, QColor, QDragEnterEvent, QDropEvent, QFont, QIcon, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ps1_3d_engine import PRESETS, PS13DEngine, PS1Config, ensure_rgb_uint8, process_image_array


def np_to_qpixmap(rgb: np.ndarray) -> QPixmap:
    """Converts a numpy RGB uint8 array to a QPixmap."""
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def load_image_unicode_safe(path: str) -> np.ndarray:
    """
    Safely loads any image from any folder path on Windows, fully supporting
    Turkish characters (Masaüstü, İndirilenler, vb.), Unicode accents, and spaces.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dosya bulunamadı: {path}")

    # 1. Native binary read + cv2.imdecode (bypasses Windows non-ASCII path limitations)
    try:
        with open(path, "rb") as f:
            file_bytes = np.frombuffer(f.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        if img_bgr is not None:
            if img_bgr.ndim == 2:
                return cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
            elif img_bgr.shape[2] == 4:
                return cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2RGB)
            else:
                return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        pass

    # 2. Robust fallback via PIL (handles unusual color spaces, EXIF orientations, TIFF, WEBP)
    try:
        from PIL import Image, ImageOps
        with Image.open(path) as pil_img:
            pil_img = ImageOps.exif_transpose(pil_img)
            return np.array(pil_img.convert("RGB"))
    except Exception as exc:
        raise ValueError(f"Görsel açılamadı veya desteklenmeyen format: {os.path.basename(path)}") from exc


def save_image_unicode_safe(path: str, rgb: np.ndarray) -> bool:
    """
    Safely saves an RGB image to any folder path on Windows, fully supporting
    Turkish characters and Unicode paths.
    """
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ext = os.path.splitext(path)[1].lower()
    if not ext:
        ext = ".png"
        path = path + ".png"
    success, buffer = cv2.imencode(ext, bgr)
    if not success:
        return False
    with open(path, "wb") as f:
        f.write(buffer)
    return True


class RenderWorker(QThread):
    """Background worker thread to run PS1 3D reconstruction without blocking the desktop UI."""

    progress_signal = Signal(str, int)
    finished_signal = Signal(object, object, dict, str)
    error_signal = Signal(str)

    def __init__(
        self,
        img_rgb: np.ndarray,
        config: PS1Config,
        engine: PS13DEngine,
        temp_obj_path: Optional[str] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.img_rgb = img_rgb
        self.config = config
        self.engine = engine
        self.temp_obj_path = temp_obj_path

    def run(self):
        try:
            self.progress_signal.emit("Analyzing image and estimating monocular depth...", 20)
            rendered_rgb, wireframe_rgb, stats = process_image_array(
                self.img_rgb,
                config=self.config,
                engine=self.engine,
                output_obj_path=self.temp_obj_path,
            )
            self.progress_signal.emit("Finalizing 15-bit RGB555 & Bayer dithering...", 90)
            self.finished_signal.emit(rendered_rgb, wireframe_rgb, stats, self.temp_obj_path or "")
        except Exception as exc:
            err_msg = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            self.error_signal.emit(err_msg)


class ScalableImageLabel(QLabel):
    """A QLabel that scales its pixmap smoothly to fit its container while preserving aspect ratio."""

    def __init__(self, placeholder_text="No Image Loaded", parent=None):
        super().__init__(parent)
        self.placeholder_text = placeholder_text
        self.original_pixmap: Optional[QPixmap] = None
        self.setAlignment(Qt.AlignCenter)
        self.setText(self.placeholder_text)
        self.setStyleSheet("color: #7a8299; font-size: 14px; font-weight: 500;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(280, 280)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        win = self.window()
        if win and hasattr(win, "dropEvent"):
            win.dropEvent(event)

    def set_image(self, pixmap: QPixmap):
        self.original_pixmap = pixmap
        self.setText("")
        self._update_display()

    def clear_image(self):
        self.original_pixmap = None
        self.setText(self.placeholder_text)
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.original_pixmap is not None:
            self._update_display()

    def _update_display(self):
        if self.original_pixmap is None:
            return
        scaled = self.original_pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.FastTransformation,  # FastTransformation preserves pixelated PS1 nearest-neighbor aesthetic!
        )
        self.setPixmap(scaled)


class PS1DesktopApp(QMainWindow):
    """Main Window for the PS1 3D Reconstruction Desktop Application."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PS1 3D Reconstruction Studio (1998 32-Bit Aesthetics)")
        self.resize(1280, 860)
        self.setMinimumSize(960, 680)
        self.setAcceptDrops(True)

        self.engine: Optional[PS13DEngine] = None
        self.current_input_rgb: Optional[np.ndarray] = None
        self.current_rendered_rgb: Optional[np.ndarray] = None
        self.current_wireframe_rgb: Optional[np.ndarray] = None
        self.current_obj_path: Optional[str] = None
        self.worker: Optional[RenderWorker] = None

        self._setup_retro_styling()
        self._build_ui()
        self._load_default_preset("mgs1")

        # Load default sample image if present
        sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "foto2.jpg")
        if os.path.exists(sample_path):
            self.load_image_from_path(sample_path)

    def _setup_retro_styling(self):
        """Applies a modern dark retro theme inspired by the classic PlayStation 1 development systems."""
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #0f1115;
            }
            QWidget {
                color: #e1e4ed;
                font-family: 'Segoe UI', 'San Francisco', 'DejaVu Sans', sans-serif;
                font-size: 13px;
            }
            QGroupBox {
                border: 1px solid #272c3d;
                border-radius: 6px;
                margin-top: 18px;
                padding-top: 14px;
                font-weight: 600;
                color: #8da4c4;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px;
            }
            QLabel {
                color: #c5cad6;
            }
            QPushButton {
                background-color: #1e2433;
                border: 1px solid #333c52;
                border-radius: 5px;
                padding: 7px 14px;
                font-weight: 600;
                color: #ffffff;
            }
            QPushButton:hover {
                background-color: #2b3447;
                border-color: #0084ff;
            }
            QPushButton:pressed {
                background-color: #171c26;
            }
            QPushButton#PrimaryBtn {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #0056b3, stop:1 #0084ff);
                border: 1px solid #0094ff;
                font-size: 14px;
                font-weight: bold;
                padding: 10px 18px;
            }
            QPushButton#PrimaryBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #006ce0, stop:1 #249eff);
            }
            QPushButton#PrimaryBtn:disabled {
                background: #1e2430;
                border-color: #2a3140;
                color: #555f75;
            }
            QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #161922;
                border: 1px solid #2a3042;
                border-radius: 4px;
                padding: 5px 8px;
                color: #f0f2f7;
            }
            QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {
                border-color: #0084ff;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #1e2330;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #0084ff;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #f0f3fa;
                border: 1px solid #0056b3;
                width: 14px;
                margin-top: -4px;
                margin-bottom: -4px;
                border-radius: 7px;
            }
            QTabWidget::pane {
                border: 1px solid #272c3d;
                background-color: #13161c;
                border-radius: 6px;
            }
            QTabBar::tab {
                background: #181d26;
                color: #8c96ab;
                padding: 8px 18px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                margin-right: 4px;
                font-weight: 600;
            }
            QTabBar::tab:selected {
                background: #252d3d;
                color: #ffffff;
                border-bottom: 2px solid #0084ff;
            }
            QProgressBar {
                background-color: #161922;
                border: 1px solid #2a3042;
                border-radius: 4px;
                text-align: center;
                color: #ffffff;
                font-weight: 600;
                height: 18px;
            }
            QProgressBar::chunk {
                background-color: #0084ff;
                border-radius: 3px;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            """
        )

    def _build_ui(self):
        """Constructs the sidebar controls and visual output viewport."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # ==========================================
        # LEFT PANEL: Controls & Settings (Scrollable)
        # ==========================================
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(360)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        left_widget = QWidget()
        sidebar_layout = QVBoxLayout(left_widget)
        sidebar_layout.setContentsMargins(6, 6, 8, 6)
        sidebar_layout.setSpacing(14)

        # Header Title
        title_label = QLabel("PS1 3D Studio")
        title_label.setFont(QFont("Segoe UI", 16, QFont.Bold))
        title_label.setStyleSheet("color: #ffffff; margin-bottom: -4px;")
        subtitle_label = QLabel("1998 32-Bit Low-Poly Hardware Reconstruction")
        subtitle_label.setStyleSheet("color: #6d7b98; font-size: 11px;")
        sidebar_layout.addWidget(title_label)
        sidebar_layout.addWidget(subtitle_label)

        # 1. Image Input Box
        img_group = QGroupBox("1. Input Image")
        img_box_layout = QVBoxLayout(img_group)

        btn_row = QHBoxLayout()
        self.btn_open = QPushButton("Browse Image...")
        self.btn_open.clicked.connect(self._on_browse_image)
        self.btn_sample = QPushButton("Load Sample")
        self.btn_sample.clicked.connect(self._on_load_sample)
        btn_row.addWidget(self.btn_open)
        btn_row.addWidget(self.btn_sample)
        img_box_layout.addLayout(btn_row)

        self.lbl_file_info = QLabel("Drag & drop an image or click Browse")
        self.lbl_file_info.setStyleSheet("color: #7b86a0; font-size: 11px;")
        self.lbl_file_info.setWordWrap(True)
        img_box_layout.addWidget(self.lbl_file_info)
        sidebar_layout.addWidget(img_group)

        # 2. Hardware Presets & Scene Mode
        preset_group = QGroupBox("2. Preset & Mode")
        preset_layout = QVBoxLayout(preset_group)

        preset_layout.addWidget(QLabel("Hardware Preset:"))
        self.combo_preset = QComboBox()
        self.combo_preset.addItems(["mgs1", "balanced", "ultra_lowpoly"])
        self.combo_preset.currentTextChanged.connect(self._on_preset_selected)
        preset_layout.addWidget(self.combo_preset)

        preset_layout.addWidget(QLabel("Scene Isolation Mode:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItem("Isolated Subject (3D Mesh + 2D Backdrop)", "isolated_subject")
        self.combo_mode.addItem("Full Scene (Landscapes/Outdoors)", "full_scene")
        preset_layout.addWidget(self.combo_mode)
        sidebar_layout.addWidget(preset_group)

        # 3. Geometry & Meshing Controls
        geom_group = QGroupBox("3. 3D Mesh & Geometry")
        geom_layout = QGridLayout(geom_group)

        # Poly count
        geom_layout.addWidget(QLabel("Triangle Polygons:"), 0, 0)
        self.spin_polys = QSpinBox()
        self.spin_polys.setRange(120, 900)
        self.spin_polys.setValue(420)
        self.slider_polys = QSlider(Qt.Horizontal)
        self.slider_polys.setRange(120, 900)
        self.slider_polys.setValue(420)
        self.spin_polys.valueChanged.connect(self.slider_polys.setValue)
        self.slider_polys.valueChanged.connect(self.spin_polys.setValue)
        geom_layout.addWidget(self.spin_polys, 0, 1)
        geom_layout.addWidget(self.slider_polys, 1, 0, 1, 2)

        # Texture Resolution
        geom_layout.addWidget(QLabel("VRAM Texture Res:"), 2, 0)
        self.spin_tex_res = QSpinBox()
        self.spin_tex_res.setRange(48, 160)
        self.spin_tex_res.setValue(92)
        geom_layout.addWidget(self.spin_tex_res, 2, 1)

        # 3D Camera Yaw
        geom_layout.addWidget(QLabel("Camera Yaw (°):"), 3, 0)
        self.spin_yaw = QDoubleSpinBox()
        self.spin_yaw.setRange(-8.0, 8.0)
        self.spin_yaw.setSingleStep(0.2)
        self.spin_yaw.setValue(1.2)
        geom_layout.addWidget(self.spin_yaw, 3, 1)

        sidebar_layout.addWidget(geom_group)

        # 4. Color & Shading
        color_group = QGroupBox("4. Color, Lighting & CLUT")
        color_layout = QGridLayout(color_group)

        # CLUT Colors
        color_layout.addWidget(QLabel("CLUT Colors:"), 0, 0)
        self.spin_colors = QSpinBox()
        self.spin_colors.setRange(8, 32)
        self.spin_colors.setValue(22)
        color_layout.addWidget(self.spin_colors, 0, 1)

        # Facet Flatness
        color_layout.addWidget(QLabel("Facet Flatness:"), 1, 0)
        self.spin_flatness = QDoubleSpinBox()
        self.spin_flatness.setRange(0.0, 1.0)
        self.spin_flatness.setSingleStep(0.05)
        self.spin_flatness.setValue(0.48)
        color_layout.addWidget(self.spin_flatness, 1, 1)

        # Face Preservation
        color_layout.addWidget(QLabel("Face Preservation:"), 2, 0)
        self.spin_face_pres = QDoubleSpinBox()
        self.spin_face_pres.setRange(0.0, 1.0)
        self.spin_face_pres.setSingleStep(0.02)
        self.spin_face_pres.setValue(0.96)
        color_layout.addWidget(self.spin_face_pres, 2, 1)

        # Crease Strength
        color_layout.addWidget(QLabel("Dihedral Creases:"), 3, 0)
        self.spin_crease = QDoubleSpinBox()
        self.spin_crease.setRange(0.0, 0.50)
        self.spin_crease.setSingleStep(0.02)
        self.spin_crease.setValue(0.18)
        color_layout.addWidget(self.spin_crease, 3, 1)

        # Scanlines
        self.chk_scanlines = QCheckBox("CRT 240p Scanline Pass")
        self.chk_scanlines.setChecked(True)
        color_layout.addWidget(self.chk_scanlines, 4, 0, 1, 2)

        sidebar_layout.addWidget(color_group)

        # 5. Render Button & Progress
        self.btn_render = QPushButton("RENDER PS1 3D")
        self.btn_render.setObjectName("PrimaryBtn")
        self.btn_render.clicked.connect(self._on_render_clicked)
        sidebar_layout.addWidget(self.btn_render)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        sidebar_layout.addWidget(self.progress_bar)

        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet("color: #7b88a3; font-size: 11px;")
        self.lbl_status.setWordWrap(True)
        sidebar_layout.addWidget(self.lbl_status)

        sidebar_layout.addStretch()
        left_scroll.setWidget(left_widget)
        splitter.addWidget(left_scroll)

        # ==========================================
        # RIGHT PANEL: Output Viewport (Tabbed)
        # ==========================================
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        # Tab widget for different visualization modes
        self.tab_viewport = QTabWidget()

        # Tab 1: Side by Side (Input vs Render)
        tab_split = QWidget()
        split_layout = QHBoxLayout(tab_split)
        split_layout.setContentsMargins(8, 8, 8, 8)
        split_layout.setSpacing(12)

        left_side = QVBoxLayout()
        left_side.addWidget(QLabel("Original Photo (Input):", alignment=Qt.AlignCenter))
        self.view_split_original = ScalableImageLabel("No Input Image")
        left_side.addWidget(self.view_split_original)
        split_layout.addLayout(left_side)

        right_side = QVBoxLayout()
        right_side.addWidget(QLabel("Reconstructed PS1 3D:", alignment=Qt.AlignCenter))
        self.view_split_render = ScalableImageLabel("Awaiting Render...")
        right_side.addWidget(self.view_split_render)
        split_layout.addLayout(right_side)

        self.tab_viewport.addTab(tab_split, "Side-by-Side Comparison")

        # Tab 2: Full Render
        tab_full = QWidget()
        full_layout = QVBoxLayout(tab_full)
        full_layout.setContentsMargins(8, 8, 8, 8)
        self.view_full_render = ScalableImageLabel("Awaiting Render...")
        full_layout.addWidget(self.view_full_render)
        self.tab_viewport.addTab(tab_full, "PS1 Rendered (Full)")

        # Tab 3: 3D Wireframe Mesh
        tab_wireframe = QWidget()
        wf_layout = QVBoxLayout(tab_wireframe)
        wf_layout.setContentsMargins(8, 8, 8, 8)
        self.view_wireframe = ScalableImageLabel("Awaiting Render...")
        wf_layout.addWidget(self.view_wireframe)
        self.tab_viewport.addTab(tab_wireframe, "3D Low-Poly Wireframe")

        right_layout.addWidget(self.tab_viewport)

        # Bottom Bar: Export buttons and statistics
        bottom_bar = QHBoxLayout()
        bottom_bar.setContentsMargins(4, 4, 4, 4)

        self.lbl_stats = QLabel("No render data yet")
        self.lbl_stats.setStyleSheet("color: #7b88a3; font-weight: 500;")
        bottom_bar.addWidget(self.lbl_stats)

        bottom_bar.addStretch()

        self.btn_save_png = QPushButton("Save PNG Image")
        self.btn_save_png.setEnabled(False)
        self.btn_save_png.clicked.connect(self._on_save_png)
        bottom_bar.addWidget(self.btn_save_png)

        self.btn_save_obj = QPushButton("Export 3D Mesh (.OBJ)")
        self.btn_save_obj.setEnabled(False)
        self.btn_save_obj.clicked.connect(self._on_save_obj)
        bottom_bar.addWidget(self.btn_save_obj)

        right_layout.addLayout(bottom_bar)
        splitter.addWidget(right_widget)

        # Balance splitter proportion
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    def _load_default_preset(self, preset_name: str):
        """Loads values from PRESETS dictionary into the sidebar widgets."""
        cfg = PRESETS.get(preset_name, PRESETS["mgs1"])
        self.spin_polys.setValue(cfg.poly_count)
        self.spin_tex_res.setValue(cfg.tex_resolution)
        self.spin_yaw.setValue(cfg.camera_yaw_deg)
        self.spin_colors.setValue(cfg.clut_colors)
        self.spin_flatness.setValue(cfg.facet_flatness)
        self.spin_face_pres.setValue(cfg.face_preservation)
        self.spin_crease.setValue(cfg.crease_strength)
        self.chk_scanlines.setChecked(cfg.scanlines)
        idx = self.combo_mode.findData(cfg.scene_mode)
        if idx >= 0:
            self.combo_mode.setCurrentIndex(idx)

    def _on_preset_selected(self, preset_name: str):
        self._load_default_preset(preset_name)

    def _on_browse_image(self):
        start_dir = getattr(self, "last_directory", None) or os.path.dirname(os.path.abspath(__file__))
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input Image",
            start_dir,
            "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff *.gif);;All Files (*)",
        )
        if file_path:
            self.load_image_from_path(file_path)

    def _on_load_sample(self):
        workspace = os.path.dirname(os.path.abspath(__file__))
        samples = [
            os.path.join(workspace, "foto2.jpg"),
            os.path.join(workspace, "examples", "sample_portrait.jpg"),
            os.path.join(workspace, "examples", "sample_action.jpg"),
            os.path.join(workspace, "examples", "sample_duo.jpg"),
        ]
        valid_samples = [s for s in samples if os.path.exists(s)]
        if valid_samples:
            self.load_image_from_path(valid_samples[0])
        else:
            QMessageBox.information(self, "No Sample Found", "No default sample images found in workspace.")

    def load_image_from_path(self, path: str):
        """Loads and prepares an input image for processing and display, with full Unicode/Turkish path support."""
        try:
            img_rgb = load_image_unicode_safe(path)

            self.current_input_rgb = img_rgb
            pixmap = np_to_qpixmap(img_rgb)
            self.view_split_original.set_image(pixmap)

            fname = os.path.basename(path)
            h, w = img_rgb.shape[:2]
            self.lbl_file_info.setText(f"Loaded: {fname} ({w}x{h} px)")
            self.lbl_status.setText(f"Ready to reconstruct '{fname}'")
            self.last_directory = os.path.dirname(os.path.abspath(path))
        except Exception as exc:
            QMessageBox.warning(self, "Image Load Error", f"Görsel yüklenemedi:\n{exc}")

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        if not event.mimeData().hasUrls():
            return
        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif")
        for url in event.mimeData().urls():
            local_path = url.toLocalFile()
            if os.path.isfile(local_path):
                ext = os.path.splitext(local_path)[1].lower()
                if ext in valid_exts:
                    self.load_image_from_path(local_path)
                    event.acceptProposedAction()
                    break

    def _on_render_clicked(self):
        if self.current_input_rgb is None:
            QMessageBox.warning(self, "No Input Image", "Please load an input image first.")
            return

        # Prepare configuration object
        base_preset = self.combo_preset.currentText()
        cfg = PS1Config(**vars(PRESETS.get(base_preset, PRESETS["mgs1"])))
        cfg.scene_mode = self.combo_mode.currentData()
        cfg.poly_count = int(self.spin_polys.value())
        cfg.tex_resolution = int(self.spin_tex_res.value())
        cfg.camera_yaw_deg = float(self.spin_yaw.value())
        cfg.clut_colors = int(self.spin_colors.value())
        cfg.head_clut_colors = max(20, int(self.spin_colors.value()))
        cfg.facet_flatness = float(self.spin_flatness.value())
        cfg.face_preservation = float(self.spin_face_pres.value())
        cfg.crease_strength = float(self.spin_crease.value())
        cfg.scanlines = bool(self.chk_scanlines.isChecked())

        # Create singleton engine if needed
        if self.engine is None:
            self.lbl_status.setText("Initializing PyTorch MiDaS neural depth model...")
            self.engine = PS13DEngine()

        # Disable render button during execution
        self.btn_render.setEnabled(False)
        self.btn_save_png.setEnabled(False)
        self.btn_save_obj.setEnabled(False)
        self.progress_bar.setValue(10)

        temp_dir = os.path.join(tempfile.gettempdir(), "ps1_desktop_exports")
        os.makedirs(temp_dir, exist_ok=True)
        temp_obj = os.path.join(temp_dir, "reconstructed_ps1.obj")

        self.worker = RenderWorker(
            self.current_input_rgb,
            cfg,
            self.engine,
            temp_obj_path=temp_obj,
            parent=self,
        )
        self.worker.progress_signal.connect(self._on_worker_progress)
        self.worker.finished_signal.connect(self._on_worker_finished)
        self.worker.error_signal.connect(self._on_worker_error)
        self.worker.start()

    def _on_worker_progress(self, message: str, percent: int):
        self.lbl_status.setText(message)
        self.progress_bar.setValue(percent)

    def _on_worker_finished(self, rendered_rgb: np.ndarray, wireframe_rgb: np.ndarray, stats: dict, obj_path: str):
        self.current_rendered_rgb = rendered_rgb
        self.current_wireframe_rgb = wireframe_rgb
        self.current_obj_path = obj_path

        # Update visual displays
        pix_render = np_to_qpixmap(rendered_rgb)
        pix_wireframe = np_to_qpixmap(wireframe_rgb)

        self.view_split_render.set_image(pix_render)
        self.view_full_render.set_image(pix_render)
        self.view_wireframe.set_image(pix_wireframe)

        # Update stats
        out_h, out_w = rendered_rgb.shape[:2]
        stats_text = (
            f"{stats['triangles']} triangles | {stats['vertices']} vertices | "
            f"Native: {stats['native_w']}x{stats['native_h']} PS1 res -> Upscaled: {out_w}x{out_h} px | "
            f"{stats['faces_detected']} face(s) preserved"
        )
        self.lbl_stats.setText(stats_text)
        self.lbl_status.setText("Reconstruction finished successfully.")
        self.progress_bar.setValue(100)

        self.btn_render.setEnabled(True)
        self.btn_save_png.setEnabled(True)
        self.btn_save_obj.setEnabled(bool(obj_path and os.path.exists(obj_path)))

    def _on_worker_error(self, err_msg: str):
        self.lbl_status.setText("Rendering failed.")
        self.progress_bar.setValue(0)
        self.btn_render.setEnabled(True)
        QMessageBox.critical(self, "Rendering Error", f"An error occurred during reconstruction:\n\n{err_msg}")

    def _on_save_png(self):
        if self.current_rendered_rgb is None:
            return
        start_dir = getattr(self, "last_directory", None) or os.path.dirname(os.path.abspath(__file__))
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save PS1 Rendered Image",
            os.path.join(start_dir, "ps1_render.png"),
            "PNG Image (*.png);;JPEG Image (*.jpg);;All Files (*)",
        )
        if save_path:
            if save_image_unicode_safe(save_path, self.current_rendered_rgb):
                self.lbl_status.setText(f"Saved image to: {os.path.basename(save_path)}")
            else:
                QMessageBox.warning(self, "Save Error", "Görsel kaydedilemedi.")

    def _on_save_obj(self):
        if not self.current_obj_path or not os.path.exists(self.current_obj_path):
            QMessageBox.warning(self, "No OBJ", "No 3D OBJ mesh file has been generated.")
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export 3D Mesh File",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "ps1_mesh.obj"),
            "Wavefront OBJ (*.obj);;All Files (*)",
        )
        if save_path:
            import shutil
            shutil.copy2(self.current_obj_path, save_path)
            self.lbl_status.setText(f"Exported 3D mesh to: {os.path.basename(save_path)}")


def main():
    # Enable high-DPI scaling on modern Windows displays
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    app = QApplication(sys.argv)
    app.setApplicationName("PS1 3D Reconstruction Studio")
    app.setStyle("Fusion")

    window = PS1DesktopApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

