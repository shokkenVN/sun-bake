import os
import sys
import json
import traceback
from typing import Optional, List
import numpy as np
import cv2

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QTabWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QCheckBox,
    QSpinBox,
    QComboBox,
    QGroupBox,
    QProgressBar,
    QPlainTextEdit,
    QFileDialog,
    QMessageBox,
    QSplitter,
)

from .styles import apply_theme
from .widgets import DropZoneWidget, InteractiveImageWidget
from ..core.cubism_core import CubismCore
from ..core.model import MocModel
from ..core.reconstructor import export_model_to_psd, ExportOptions
from ..core.baker import bake_psd_to_atlases, BakeOptions
from ..core.preview import render_model_preview


class ExportWorker(QThread):
    progress = Signal(int, int, str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, model: MocModel, output_psd: str, options: ExportOptions):
        super().__init__()
        self.model = model
        self.output_psd = output_psd
        self.options = options

    def run(self):
        try:
            res = export_model_to_psd(
                self.model,
                self.output_psd,
                self.options,
                progress_callback=lambda cur, tot, msg: self.progress.emit(cur, tot, msg),
            )
            self.finished.emit(res)
        except Exception as e:
            self.error.emit(f"{e}\n{traceback.format_exc()}")


class BakeWorker(QThread):
    progress = Signal(int, int, str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, psd_path: str, model: MocModel, options: BakeOptions):
        super().__init__()
        self.psd_path = psd_path
        self.model = model
        self.options = options

    def run(self):
        try:
            res = bake_psd_to_atlases(
                self.psd_path,
                self.model,
                self.options,
                progress_callback=lambda cur, tot, msg: self.progress.emit(cur, tot, msg),
            )
            self.finished.emit(res)
        except Exception as e:
            self.error.emit(f"{e}\n{traceback.format_exc()}")


class PreviewWorker(QThread):
    finished = Signal(object)
    error = Signal(str)

    def __init__(
        self,
        model: MocModel,
        scale: float = 0.2,
        custom_textures: Optional[List[np.ndarray]] = None,
    ):
        super().__init__()
        self.model = model
        self.scale = scale
        self.custom_textures = custom_textures

    def run(self):
        try:
            img = render_model_preview(
                self.model,
                scale=self.scale,
                custom_textures=self.custom_textures,
            )
            self.finished.emit(img)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("sun-bake")
        self.resize(1240, 800)
        self.setAcceptDrops(True)

        self.export_model: Optional[MocModel] = None
        self.bake_model: Optional[MocModel] = None
        self.bake_psd_path: Optional[str] = None
        self.last_baked_textures: Optional[List[np.ndarray]] = None
        self._cached_diff_map: Optional[np.ndarray] = None
        self._cached_baked_preview: Optional[np.ndarray] = None
        self._preview_worker: Optional[PreviewWorker] = None
        self._bake_preview_worker: Optional[PreviewWorker] = None

        self._init_ui()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(8)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title_lbl = QLabel("sun-bake")
        title_lbl.setStyleSheet("font-size: 16px; font-weight: 700;")
        subtitle_lbl = QLabel("I bet you're making something lewd, aren't you?")
        subtitle_lbl.setStyleSheet("font-size: 12px; color: #8b949e;")
        title_box.addWidget(title_lbl)
        title_box.addWidget(subtitle_lbl)

        header.addLayout(title_box)
        header.addStretch()

        try:
            core = CubismCore.get_instance()
            ver = core.get_version()
            sdk_lbl = QLabel(f"Core {ver >> 24}.{(ver >> 16) & 0xff}.{ver & 0xffff}")
            sdk_lbl.setStyleSheet("color: #3fb950; font-size: 12px;")
        except Exception:
            sdk_lbl = QLabel("Core missing")
            sdk_lbl.setStyleSheet("color: #f85149; font-size: 12px;")
        header.addWidget(sdk_lbl)
        root_layout.addLayout(header)

        self.tabs = QTabWidget()
        self.tab_export = QWidget()
        self.tab_bake = QWidget()

        self._setup_export_tab()
        self._setup_bake_tab()

        self.tabs.addTab(self.tab_export, "Export (Moc3 → PSD)")
        self.tabs.addTab(self.tab_bake, "Bake (PSD → Textures)")
        root_layout.addWidget(self.tabs)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        if not path:
            return
        ext = path.lower()
        if ext.endswith(".psd"):
            self.tabs.setCurrentWidget(self.tab_bake)
            self._load_bake_psd(path)
        elif ext.endswith((".moc3", ".json")):
            if self.tabs.currentWidget() == self.tab_bake:
                self.txt_bake_model.setText(path)
                self._load_bake_model_internal(path)
            else:
                self.tabs.setCurrentWidget(self.tab_export)
                self._load_export_model(path)
        event.acceptProposedAction()

    def _setup_export_tab(self):
        layout = QHBoxLayout(self.tab_export)
        splitter = QSplitter(Qt.Horizontal)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(4, 4, 8, 4)
        left_layout.setSpacing(10)

        self.exp_dropzone = DropZoneWidget(
            title="Model File",
            subtitle="Drop .model3.json or .moc3",
            file_filter="Model Files (*.json *.moc3);;All Files (*.*)",
        )
        self.exp_dropzone.file_selected.connect(self._load_export_model)
        left_layout.addWidget(self.exp_dropzone)

        self.exp_info_group = QGroupBox("Model Info")
        info_layout = QGridLayout(self.exp_info_group)
        self.lbl_exp_name = QLabel("Name: -")
        self.lbl_exp_canvas = QLabel("Canvas: -")
        self.lbl_exp_drawables = QLabel("Drawables: -")
        self.lbl_exp_parts = QLabel("Parts: -")
        self.lbl_exp_textures = QLabel("Textures: -")
        self.lbl_exp_textures.setWordWrap(True)

        info_layout.addWidget(self.lbl_exp_name, 0, 0)
        info_layout.addWidget(self.lbl_exp_canvas, 0, 1)
        info_layout.addWidget(self.lbl_exp_drawables, 1, 0)
        info_layout.addWidget(self.lbl_exp_parts, 1, 1)
        info_layout.addWidget(self.lbl_exp_textures, 2, 0, 1, 2)
        left_layout.addWidget(self.exp_info_group)

        opt_group = QGroupBox("Options")
        opt_layout = QVBoxLayout(opt_group)

        self.chk_group_parts = QCheckBox("Group layers by Part")
        self.chk_group_parts.setChecked(True)
        self.chk_group_parts.setToolTip("Put contiguous drawables from the same Part into folders.")

        self.chk_write_meta = QCheckBox("Write .psd.meta.json")
        self.chk_write_meta.setChecked(True)
        self.chk_write_meta.setToolTip("Saves model/layer metadata next to the PSD for baking.")

        opt_layout.addWidget(self.chk_group_parts)
        opt_layout.addWidget(self.chk_write_meta)

        exp_row = QHBoxLayout()
        self.cmb_exp_expression = QComboBox()
        self.cmb_exp_expression.addItem("Default pose", None)
        self.cmb_exp_expression.currentIndexChanged.connect(self._on_export_expression_changed)
        btn_exp_browse_exp = QPushButton("Browse...")
        btn_exp_browse_exp.clicked.connect(self._browse_export_expression)
        exp_row.addWidget(QLabel("Expression:"))
        exp_row.addWidget(self.cmb_exp_expression, 1)
        exp_row.addWidget(btn_exp_browse_exp)
        opt_layout.addLayout(exp_row)

        dest_box = QHBoxLayout()
        self.txt_exp_out = QLineEdit()
        self.txt_exp_out.setPlaceholderText("Output .psd path")
        btn_exp_browse_out = QPushButton("Browse...")
        btn_exp_browse_out.clicked.connect(self._browse_export_out)
        dest_box.addWidget(self.txt_exp_out)
        dest_box.addWidget(btn_exp_browse_out)
        opt_layout.addLayout(dest_box)

        left_layout.addWidget(opt_group)

        self.btn_export = QPushButton("Export PSD")
        self.btn_export.setObjectName("ActionBtn")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._start_export)
        left_layout.addWidget(self.btn_export)

        self.exp_progress = QProgressBar()
        self.exp_progress.setValue(0)
        left_layout.addWidget(self.exp_progress)

        self.exp_log = QPlainTextEdit()
        self.exp_log.setReadOnly(True)
        self.exp_log.setMaximumBlockCount(500)
        left_layout.addWidget(self.exp_log, 1)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(8, 4, 4, 4)

        prev_header = QHBoxLayout()
        prev_lbl = QLabel("Preview")
        prev_lbl.setStyleSheet("font-weight: 600;")
        hint_lbl = QLabel("Scroll: zoom | Drag: pan | Double-click: fit")
        hint_lbl.setStyleSheet("font-size: 11px; color: #8b949e;")
        prev_header.addWidget(prev_lbl)
        prev_header.addStretch()
        prev_header.addWidget(hint_lbl)
        right_layout.addLayout(prev_header)

        self.exp_preview_view = InteractiveImageWidget()
        right_layout.addWidget(self.exp_preview_view, 1)

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)
        layout.addWidget(splitter)

    def _setup_bake_tab(self):
        layout = QHBoxLayout(self.tab_bake)
        splitter = QSplitter(Qt.Horizontal)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(4, 4, 8, 4)
        left_layout.setSpacing(10)

        self.bake_dropzone = DropZoneWidget(
            title="PSD File",
            subtitle="Drop edited .psd",
            file_filter="Photoshop PSD (*.psd);;All Files (*.*)",
        )
        self.bake_dropzone.file_selected.connect(self._load_bake_psd)
        left_layout.addWidget(self.bake_dropzone)

        src_group = QGroupBox("Target Model")
        src_layout = QVBoxLayout(src_group)

        model_box = QHBoxLayout()
        self.txt_bake_model = QLineEdit()
        self.txt_bake_model.setPlaceholderText(".model3.json or .moc3 path")
        btn_bake_model_browse = QPushButton("Browse...")
        btn_bake_model_browse.clicked.connect(self._browse_bake_model)
        model_box.addWidget(self.txt_bake_model)
        model_box.addWidget(btn_bake_model_browse)
        src_layout.addLayout(model_box)

        bake_exp_box = QHBoxLayout()
        self.cmb_bake_expression = QComboBox()
        self.cmb_bake_expression.addItem("Default pose", None)
        self.cmb_bake_expression.currentIndexChanged.connect(self._on_bake_expression_changed)
        btn_bake_exp_browse = QPushButton("Browse...")
        btn_bake_exp_browse.clicked.connect(self._browse_bake_expression)
        bake_exp_box.addWidget(QLabel("Expression:"))
        bake_exp_box.addWidget(self.cmb_bake_expression, 1)
        bake_exp_box.addWidget(btn_bake_exp_browse)
        src_layout.addLayout(bake_exp_box)

        out_dir_box = QHBoxLayout()
        self.txt_bake_out_dir = QLineEdit()
        self.txt_bake_out_dir.setPlaceholderText("Output directory (blank = overwrite model textures)")
        btn_bake_out_browse = QPushButton("Browse...")
        btn_bake_out_browse.clicked.connect(self._browse_bake_out_dir)
        out_dir_box.addWidget(self.txt_bake_out_dir)
        out_dir_box.addWidget(btn_bake_out_browse)
        src_layout.addLayout(out_dir_box)

        left_layout.addWidget(src_group)

        bake_opt_group = QGroupBox("Options")
        bake_opt_layout = QVBoxLayout(bake_opt_group)

        bleed_row = QHBoxLayout()
        bleed_lbl = QLabel("Edge bleed (px):")
        bleed_lbl.setToolTip("Dilates RGB into transparent borders to avoid bilinear filtering seams.")
        self.spn_bleed = QSpinBox()
        self.spn_bleed.setRange(0, 10)
        self.spn_bleed.setValue(2)
        bleed_row.addWidget(bleed_lbl)
        bleed_row.addWidget(self.spn_bleed)
        bleed_row.addStretch()
        bake_opt_layout.addLayout(bleed_row)

        self.chk_backup = QCheckBox("Backup original textures (.bak)")
        self.chk_backup.setChecked(True)
        bake_opt_layout.addWidget(self.chk_backup)

        left_layout.addWidget(bake_opt_group)

        self.btn_bake = QPushButton("Bake Textures")
        self.btn_bake.setObjectName("ActionBtn")
        self.btn_bake.setEnabled(False)
        self.btn_bake.clicked.connect(self._start_bake)
        left_layout.addWidget(self.btn_bake)

        self.bake_progress = QProgressBar()
        self.bake_progress.setValue(0)
        left_layout.addWidget(self.bake_progress)

        self.bake_log = QPlainTextEdit()
        self.bake_log.setReadOnly(True)
        self.bake_log.setMaximumBlockCount(500)
        left_layout.addWidget(self.bake_log, 1)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(8, 4, 4, 4)

        prev_header = QHBoxLayout()
        prev_lbl = QLabel("Viewer")
        prev_lbl.setStyleSheet("font-weight: 600;")

        self.cmb_bake_view = QComboBox()
        self.cmb_bake_view.addItems([
            "Original Atlas",
            "Baked Atlas",
            "Diff Map",
            "Assembled Model",
        ])
        self.cmb_bake_view.currentIndexChanged.connect(self._on_bake_view_mode_changed)

        prev_header.addWidget(prev_lbl)
        prev_header.addStretch()
        prev_header.addWidget(self.cmb_bake_view)
        right_layout.addLayout(prev_header)

        self.bake_preview_view = InteractiveImageWidget()
        right_layout.addWidget(self.bake_preview_view, 1)

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)
        layout.addWidget(splitter)

    def _populate_expression_combo(self, combo: QComboBox, model: MocModel):
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("Default pose", None)
        seen = set()
        for name, path in sorted(model.expressions.items()):
            if path not in seen:
                seen.add(path)
                combo.addItem(f"{name} ({os.path.basename(path)})", path)
        combo.blockSignals(False)

    def _load_export_model(self, file_path: str):
        try:
            self.exp_log.appendPlainText(f"Loading {file_path}")
            if file_path.lower().endswith(".json"):
                self.export_model = MocModel.from_model3_json(file_path)
            else:
                self.export_model = MocModel(moc_path=file_path)

            m = self.export_model
            c = m.canvas
            name = os.path.basename(m.moc_path)
            self.exp_dropzone.set_loaded_file(file_path)

            self.lbl_exp_name.setText(f"Name: {name}")
            self.lbl_exp_canvas.setText(f"Canvas: {c.width:.0f}x{c.height:.0f}")
            self.lbl_exp_drawables.setText(f"Drawables: {len(m.drawables)}")
            self.lbl_exp_parts.setText(f"Parts: {len(m.parts)}")
            tex_names = ", ".join(os.path.basename(t) for t in m.texture_paths)
            self.lbl_exp_textures.setText(f"Textures ({len(m.textures)}): {tex_names}")

            self._populate_expression_combo(self.cmb_exp_expression, m)

            default_psd = os.path.splitext(m.moc_path)[0] + ".psd"
            self.txt_exp_out.setText(default_psd)
            self.btn_export.setEnabled(True)

            self._render_export_preview_async()
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Could not load model:\n{e}")
            self.exp_log.appendPlainText(f"Error: {e}")

    def _render_export_preview_async(self):
        if not self.export_model:
            return
        self._preview_worker = PreviewWorker(self.export_model, scale=0.2)
        self._preview_worker.finished.connect(lambda img: self.exp_preview_view.set_image(img, keep_view=True))
        self._preview_worker.error.connect(lambda err: self.exp_log.appendPlainText(f"Preview error: {err}"))
        self._preview_worker.start()

    def _on_export_expression_changed(self, _idx):
        if not self.export_model:
            return
        expr_path = self.cmb_exp_expression.currentData()
        if expr_path:
            try:
                res = self.export_model.apply_expression(expr_path)
                self.exp_log.appendPlainText(f"Expression: {res['expression']} ({res['applied_parameters']} params)")
            except Exception as e:
                self.exp_log.appendPlainText(f"Error applying expression: {e}")
        else:
            self.export_model.reset_parameters()
            self.exp_log.appendPlainText("Pose reset to default.")

        self._render_export_preview_async()

    def _browse_export_expression(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select Expression", "", "Expression Files (*.exp3.json *.json);;All Files (*.*)"
        )
        if f and self.export_model:
            base_name = os.path.basename(f)
            self.cmb_exp_expression.addItem(base_name, f)
            self.cmb_exp_expression.setCurrentIndex(self.cmb_exp_expression.count() - 1)

    def _browse_export_out(self):
        f, _ = QFileDialog.getSaveFileName(self, "Save PSD", self.txt_exp_out.text(), "Photoshop PSD (*.psd)")
        if f:
            self.txt_exp_out.setText(f)

    def _start_export(self):
        if not self.export_model:
            return
        out_path = self.txt_exp_out.text().strip()
        if not out_path:
            QMessageBox.warning(self, "Missing Output", "Set an output PSD file path first.")
            return

        opts = ExportOptions(
            group_by_part=self.chk_group_parts.isChecked(),
            write_metadata=self.chk_write_meta.isChecked(),
            expression=self.cmb_exp_expression.currentData(),
        )

        self.btn_export.setEnabled(False)
        self.exp_progress.setValue(0)
        self.exp_log.appendPlainText(f"Exporting to {out_path}...")

        self.worker = ExportWorker(self.export_model, out_path, opts)
        self.worker.progress.connect(self._on_export_progress)
        self.worker.finished.connect(self._on_export_finished)
        self.worker.error.connect(self._on_export_error)
        self.worker.start()

    def _on_export_progress(self, cur, tot, msg):
        pct = int((cur / tot) * 100) if tot > 0 else 0
        self.exp_progress.setValue(pct)
        if cur % 25 == 0 or cur == tot:
            self.exp_log.appendPlainText(f"[{pct}%] {msg}")

    def _on_export_finished(self, res):
        self.btn_export.setEnabled(True)
        self.exp_progress.setValue(100)
        msg = (
            f"Exported {res['exported_layers']} layers to {res['output_psd_path']} "
            f"({res['file_size_mb']:.1f} MB, {res['elapsed_seconds']:.2f}s)"
        )
        self.exp_log.appendPlainText(msg)

        # Pre-fill bake tab so user can immediately switch and bake after editing
        self._load_bake_psd(res["output_psd_path"])

    def _on_export_error(self, err_msg):
        self.btn_export.setEnabled(True)
        QMessageBox.critical(self, "Export Failed", err_msg)
        self.exp_log.appendPlainText(f"Error: {err_msg}")

    def _load_bake_psd(self, psd_path: str):
        self.bake_psd_path = psd_path
        self.bake_dropzone.set_loaded_file(psd_path)
        self.bake_log.appendPlainText(f"PSD: {psd_path}")

        meta_path = f"{psd_path}.meta.json"
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                rel_moc = meta.get("model_moc")
                if rel_moc:
                    abs_moc = os.path.normpath(os.path.join(os.path.dirname(psd_path), rel_moc))
                    if os.path.isfile(abs_moc):
                        self.txt_bake_model.setText(abs_moc)
                        if not self.bake_model or os.path.abspath(self.bake_model.moc_path) != os.path.abspath(abs_moc):
                            self._load_bake_model_internal(abs_moc)

                rel_exp = meta.get("expression")
                if rel_exp:
                    found_exp = False
                    for i in range(self.cmb_bake_expression.count()):
                        val = self.cmb_bake_expression.itemText(i)
                        data = self.cmb_bake_expression.itemData(i)
                        if rel_exp in val or (data and rel_exp in data):
                            self.cmb_bake_expression.setCurrentIndex(i)
                            found_exp = True
                            break
                    if not found_exp and os.path.isfile(rel_exp):
                        base_name = os.path.basename(rel_exp)
                        self.cmb_bake_expression.addItem(base_name, rel_exp)
                        self.cmb_bake_expression.setCurrentIndex(self.cmb_bake_expression.count() - 1)
            except Exception as e:
                self.bake_log.appendPlainText(f"Metadata error: {e}")

        self._check_bake_ready()

    def _browse_bake_model(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select Model", "", "Model Files (*.json *.moc3);;All Files (*.*)")
        if f:
            self.txt_bake_model.setText(f)
            self._load_bake_model_internal(f)

    def _browse_bake_out_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if d:
            self.txt_bake_out_dir.setText(d)

    def _load_bake_model_internal(self, model_path: str):
        try:
            if model_path.lower().endswith(".json"):
                self.bake_model = MocModel.from_model3_json(model_path)
            else:
                self.bake_model = MocModel(moc_path=model_path)

            self._cached_diff_map = None
            self._cached_baked_preview = None
            self._populate_expression_combo(self.cmb_bake_expression, self.bake_model)

            if self.bake_model.textures:
                self.bake_preview_view.set_image(self.bake_model.textures[0])

            self.bake_log.appendPlainText(f"Model: {os.path.basename(self.bake_model.moc_path)}")
            self._check_bake_ready()
        except Exception as e:
            self.bake_log.appendPlainText(f"Error loading model: {e}")

    def _on_bake_expression_changed(self, _idx):
        if not self.bake_model:
            return
        expr_path = self.cmb_bake_expression.currentData()
        if expr_path:
            try:
                res = self.bake_model.apply_expression(expr_path)
                self.bake_log.appendPlainText(f"Expression: {res['expression']}")
            except Exception as e:
                self.bake_log.appendPlainText(f"Error applying expression: {e}")
        else:
            self.bake_model.reset_parameters()
            self.bake_log.appendPlainText("Pose reset to default.")
        self._cached_baked_preview = None
        self._update_bake_preview()

    def _browse_bake_expression(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select Expression", "", "Expression Files (*.exp3.json *.json);;All Files (*.*)"
        )
        if f and self.bake_model:
            base_name = os.path.basename(f)
            self.cmb_bake_expression.addItem(base_name, f)
            self.cmb_bake_expression.setCurrentIndex(self.cmb_bake_expression.count() - 1)

    def _check_bake_ready(self):
        ready = (
            bool(self.bake_psd_path)
            and os.path.isfile(self.bake_psd_path)
            and self.bake_model is not None
        )
        self.btn_bake.setEnabled(ready)

    def _start_bake(self):
        if not self.bake_psd_path or not self.bake_model:
            return

        opts = BakeOptions(
            bleed_radius=self.spn_bleed.value(),
            backup_original=self.chk_backup.isChecked(),
            output_dir=self.txt_bake_out_dir.text().strip() or None,
            expression=self.cmb_bake_expression.currentData(),
        )

        self.btn_bake.setEnabled(False)
        self.bake_progress.setValue(0)
        self.bake_log.appendPlainText("Baking textures...")

        self.bake_worker = BakeWorker(self.bake_psd_path, self.bake_model, opts)
        self.bake_worker.progress.connect(self._on_bake_progress)
        self.bake_worker.finished.connect(self._on_bake_finished)
        self.bake_worker.error.connect(self._on_bake_error)
        self.bake_worker.start()

    def _on_bake_progress(self, cur, tot, msg):
        pct = int((cur / tot) * 100) if tot > 0 else 0
        self.bake_progress.setValue(pct)
        if cur % 25 == 0 or cur == tot:
            self.bake_log.appendPlainText(f"[{pct}%] {msg}")

    def _on_bake_finished(self, res):
        self.btn_bake.setEnabled(True)
        self.bake_progress.setValue(100)
        self.bake_log.appendPlainText(
            f"Baked {res['baked_layers']} layers ({res['unchanged_layers']} unchanged) in {res['elapsed_seconds']:.2f}s"
        )
        for p in res["written_textures"]:
            self.bake_log.appendPlainText(f"  -> {p}")

        self.last_baked_textures = [
            cv2.imread(p, cv2.IMREAD_UNCHANGED) for p in res["written_textures"]
        ]
        self._cached_diff_map = None
        self._cached_baked_preview = None

        if self.cmb_bake_view.currentIndex() == 1:
            self._update_bake_preview()
        else:
            self.cmb_bake_view.setCurrentIndex(1)

    def _on_bake_error(self, err_msg):
        self.btn_bake.setEnabled(True)
        QMessageBox.critical(self, "Bake Failed", err_msg)
        self.bake_log.appendPlainText(f"Error: {err_msg}")

    def _on_bake_view_mode_changed(self, _idx):
        self._update_bake_preview()

    def _update_bake_preview(self):
        if not self.bake_model or not self.bake_model.textures:
            return

        idx = self.cmb_bake_view.currentIndex()
        orig = self.bake_model.textures[0]
        baked = self.last_baked_textures[0] if self.last_baked_textures else orig

        if idx == 0:
            self.bake_preview_view.set_image(orig, keep_view=True)
        elif idx == 1:
            self.bake_preview_view.set_image(baked, keep_view=True)
        elif idx == 2:
            if self._cached_diff_map is None:
                diff = np.abs(baked.astype(np.int16) - orig.astype(np.int16))
                has_diff = np.any(diff > 2, axis=2)
                vis_diff = np.full_like(orig, [30, 30, 30, 255])
                vis_diff[has_diff] = [0, 255, 0, 255]
                self._cached_diff_map = vis_diff
            self.bake_preview_view.set_image(self._cached_diff_map, keep_view=True)
        elif idx == 3:
            if self._cached_baked_preview is not None:
                self.bake_preview_view.set_image(self._cached_baked_preview, keep_view=True)
            else:
                self._bake_preview_worker = PreviewWorker(
                    self.bake_model, scale=0.2, custom_textures=self.last_baked_textures
                )
                self._bake_preview_worker.finished.connect(self._on_baked_preview_ready)
                self._bake_preview_worker.start()

    def _on_baked_preview_ready(self, img):
        self._cached_baked_preview = img
        if self.cmb_bake_view.currentIndex() == 3:
            self.bake_preview_view.set_image(img, keep_view=True)


def run_gui():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    apply_theme("dark")

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    run_gui()
