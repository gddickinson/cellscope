"""Helper dialogs for the focused GUI (system info, shortcuts, about)."""
import os
import subprocess
from PyQt5.QtWidgets import (
    QMessageBox, QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QSpinBox, QCheckBox, QPushButton, QFormLayout,
)


def channel_chooser(parent, n_channels, default_dic=1, default_fluo=0):
    """Modal dialog asking user to map DIC + fluorescence channels.

    Used after loading a multichannel TIFF (n_channels > 1).
    Returns (dic_idx, fluo_idx) or (None, None) if user picks
    "single-channel mode" (collapse channels).
    """
    dlg = QDialog(parent)
    dlg.setWindowTitle("Multichannel recording detected")
    layout = QVBoxLayout(dlg)
    layout.addWidget(QLabel(
        f"This file has {n_channels} channels. Select which channel\n"
        f"is DIC (for detection) and which is the fluorescence label\n"
        f"(SiR-actin Cy5 or similar — used for false-positive filtering\n"
        f"and Cy5 recovery). Channels are 0-indexed."))
    form = QFormLayout()
    dic_sp = QSpinBox(); dic_sp.setRange(0, n_channels - 1)
    dic_sp.setValue(min(default_dic, n_channels - 1))
    form.addRow("DIC channel:", dic_sp)
    fluo_sp = QSpinBox(); fluo_sp.setRange(0, n_channels - 1)
    fluo_sp.setValue(min(default_fluo, n_channels - 1))
    form.addRow("Fluorescence channel:", fluo_sp)
    use_fluo = QCheckBox("Use fluorescence channel for Cy5 recovery + scoring")
    use_fluo.setChecked(True)
    form.addRow(use_fluo)
    layout.addLayout(form)
    btn_row = QHBoxLayout()
    btn_single = QPushButton("Single-channel only")
    btn_ok = QPushButton("OK")
    btn_ok.setDefault(True)
    btn_row.addWidget(btn_single)
    btn_row.addStretch()
    btn_row.addWidget(btn_ok)
    layout.addLayout(btn_row)

    result = {"dic": None, "fluo": None}

    def _ok():
        result["dic"] = dic_sp.value()
        result["fluo"] = fluo_sp.value() if use_fluo.isChecked() else None
        dlg.accept()

    def _single():
        result["dic"] = None
        result["fluo"] = None
        dlg.accept()

    btn_ok.clicked.connect(_ok)
    btn_single.clicked.connect(_single)
    dlg.exec_()
    return result["dic"], result["fluo"]


def detect_gpu():
    try:
        import torch
        return (torch.cuda.is_available()
                or torch.backends.mps.is_available())
    except Exception:
        return False


def show_system_info(parent):
    import torch
    lines = [
        f"Python: {os.sys.version.split()[0]}",
        f"PyTorch: {torch.__version__}",
        f"CUDA available: {torch.cuda.is_available()}",
    ]
    if torch.cuda.is_available():
        lines.append(f"CUDA device: {torch.cuda.get_device_name(0)}")
    lines.append(f"MPS available: {torch.backends.mps.is_available()}")
    try:
        import cellpose
        lines.append(f"Cellpose: {cellpose.version}")
    except Exception:
        lines.append("Cellpose: not found")
    gpu_on = getattr(parent, "act_gpu", None)
    if gpu_on:
        lines.append(f"GPU enabled: {gpu_on.isChecked()}")
    QMessageBox.information(parent, "System Info", "\n".join(lines))


def show_recording_info(parent, recording, mode, detect_result):
    if recording is None:
        QMessageBox.information(parent, "Recording Info",
                                "No recording loaded.")
        return
    r = recording
    n = len(r["frames"])
    H, W = r["frames"][0].shape
    lines = [
        f"Name: {r.get('name', '?')}",
        f"Path: {r.get('video_path', '?')}",
        f"Frames: {n}",
        f"Size: {W} x {H} px",
        f"Pixel size: {r.get('um_per_px', '?')} um/px",
        f"Time interval: {r.get('time_interval_min', '?')} min",
        f"Pipeline mode: {mode}",
    ]
    if detect_result:
        masks = detect_result["masks"]
        detected = int(masks.any(axis=(1, 2)).sum())
        lines.append(f"Detected: {detected}/{n} frames")
        if "tracks" in detect_result:
            lines.append(f"Tracks: {len(detect_result['tracks'])}")
    QMessageBox.information(parent, "Recording Info", "\n".join(lines))


def show_shortcuts(parent):
    text = (
        "Keyboard Shortcuts\n\n"
        "Ctrl+O    Open recording\n"
        "Ctrl+S    Export results\n"
        "Ctrl+E    Edit masks\n"
        "Ctrl+I    Recording info\n"
        "Ctrl+=    Zoom in\n"
        "Ctrl+-    Zoom out\n"
        "Ctrl+0    Zoom to fit\n"
        "Ctrl+Shift+C  Clear all results\n"
        "Ctrl+Q    Quit\n\n"
        "Image Viewer:\n"
        "Scroll wheel    Zoom at cursor\n"
        "Right-drag      Pan image\n"
        "Ctrl+left-drag  Pan image\n\n"
        "Mask Editor (when open):\n"
        "B / E / P / F   Brush / Eraser / Polygon / Fill\n"
        "1-9             Select cell ID\n"
        "Left/Right      Previous/next frame\n"
        "Ctrl+Z          Undo\n"
        "Ctrl+Shift+Z    Redo"
    )
    QMessageBox.information(parent, "Keyboard Shortcuts", text)


def show_about(parent):
    QMessageBox.about(
        parent, "About CellScope",
        "CellScope — Focused Pipeline\n\n"
        "Analyzes DIC/phase-contrast time-lapse microscopy\n"
        "of migrating keratinocytes.\n\n"
        "Detection: Cellpose-SAM (cpsam) + DeepSea union\n"
        "Tracking: Hungarian algorithm with gap fill\n"
        "Analysis: speed, morphology, edge dynamics\n\n"
        "Supports single-cell and multi-cell recordings\n"
        "with automatic cell division detection.\n\n"
        "Built for Holt et al. 2021 (eLife) data."
    )


def open_doc(filename):
    path = os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), filename)
    if not os.path.exists(path):
        return
    if os.sys.platform == "darwin":
        subprocess.Popen(["open", path])
    elif os.sys.platform == "win32":
        os.startfile(path)
    else:
        subprocess.Popen(["xdg-open", path])
