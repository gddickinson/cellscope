"""Helper dialogs for the focused GUI (system info, shortcuts, about)."""
import os
import subprocess
from PyQt5.QtWidgets import QMessageBox


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
