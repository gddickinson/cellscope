"""Setup wizard — check, install, and configure environments and models.

Uses only tkinter + stdlib. Platform-aware: works on macOS, Linux,
and Windows. Can install conda environments, pip packages, and
download required models.

Launch:
    python setup_wizard.py
"""
import os
import sys
import platform
import subprocess
import shutil
import tempfile
import threading
import tkinter as tk
from tkinter import scrolledtext, messagebox

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
IS_WINDOWS = sys.platform == "win32"
IS_MAC = sys.platform == "darwin"
PLATFORM_NAME = {"darwin": "macOS", "win32": "Windows",
                 "linux": "Linux"}.get(sys.platform, sys.platform)

ENVS = {
    "cellpose4": {
        "python": "3.10",
        "pip": [
            "cellpose==4.1.1", "PyQt5", "matplotlib",
            "scikit-image", "scikit-learn", "scipy",
            "tifffile", "opencv-python-headless",
            "transformers", "huggingface_hub", "peft",
        ],
        "desc": "Primary environment (cpsam ViT detection)",
    },
    "cellpose": {
        "python": "3.10",
        "pip": [
            "cellpose==3.1.1.1", "PyQt5", "matplotlib",
            "scikit-image", "scikit-learn", "scipy",
            "tifffile", "opencv-python-headless",
            "transformers", "huggingface_hub", "peft",
        ],
        "desc": "Fallback environment (CP3 models)",
    },
}

DEEPSEA_MODEL_DIR = os.path.join(PROJECT_DIR, "data", "models", "deepsea")


def _find_conda():
    conda = shutil.which("conda")
    if conda:
        return conda
    if IS_WINDOWS:
        for p in [
            os.path.expanduser("~/anaconda3/Scripts/conda.exe"),
            os.path.expanduser("~/miniconda3/Scripts/conda.exe"),
            r"C:\ProgramData\anaconda3\Scripts\conda.exe",
            r"C:\ProgramData\miniconda3\Scripts\conda.exe",
        ]:
            if os.path.exists(p):
                return p
    return "conda"


CONDA = _find_conda()


def _get_system_info():
    """Gather system information for display."""
    import multiprocessing
    lines = [
        f"OS:       {PLATFORM_NAME} {platform.release()}",
        f"Machine:  {platform.machine()}",
        f"Python:   {platform.python_version()}",
        f"CPU:      {multiprocessing.cpu_count()} cores",
    ]

    # RAM
    try:
        if IS_MAC or sys.platform == "linux":
            import resource
            # os.sysconf is more reliable
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            ram_gb = (pages * page_size) / (1024 ** 3)
            lines.append(f"RAM:      {ram_gb:.1f} GB")
        elif IS_WINDOWS:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            c_ulong = ctypes.c_ulong
            class MEMORYSTATUS(ctypes.Structure):
                _fields_ = [("dwLength", c_ulong),
                            ("dwMemoryLoad", c_ulong),
                            ("dwTotalPhys", c_ulong),
                            ("dwAvailPhys", c_ulong),
                            ("dwTotalPageFile", c_ulong),
                            ("dwAvailPageFile", c_ulong),
                            ("dwTotalVirtual", c_ulong),
                            ("dwAvailVirtual", c_ulong)]
            mem = MEMORYSTATUS()
            mem.dwLength = ctypes.sizeof(MEMORYSTATUS)
            kernel32.GlobalMemoryStatus(ctypes.byref(mem))
            lines.append(f"RAM:      {mem.dwTotalPhys / (1024**3):.1f} GB")
    except Exception:
        lines.append("RAM:      unknown")

    # Disk
    try:
        usage = shutil.disk_usage(PROJECT_DIR)
        lines.append(f"Disk:     {usage.free / (1024**3):.0f} GB free "
                     f"/ {usage.total / (1024**3):.0f} GB total")
    except Exception:
        pass

    # GPU
    gpu_lines = []
    try:
        import torch
        if torch.cuda.is_available():
            gpu_lines.append(
                f"GPU:      CUDA — {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            gpu_lines.append("GPU:      Apple MPS (Metal)")
        else:
            gpu_lines.append("GPU:      None detected (CPU only)")
    except ImportError:
        gpu_lines.append("GPU:      PyTorch not installed (unknown)")
    lines.extend(gpu_lines)

    # Conda
    try:
        r = subprocess.run([CONDA, "--version"], capture_output=True,
                           text=True, timeout=5)
        lines.append(f"Conda:    {r.stdout.strip()}")
    except Exception:
        lines.append("Conda:    not found on PATH")

    # Git
    try:
        r = subprocess.run(["git", "--version"], capture_output=True,
                           text=True, timeout=5)
        lines.append(f"Git:      {r.stdout.strip()}")
    except Exception:
        lines.append("Git:      not found (needed for DeepSea install)")

    return "\n".join(lines)

MODELS = {
    "DeepSea": {
        "path": os.path.join(DEEPSEA_MODEL_DIR, "segmentation.pth"),
        "desc": "DeepSea segmentation model (~8 MB)",
    },
    "MedSAM": {
        "path": None, "desc": "Auto-downloads on first use (~375 MB)",
        "auto": True,
    },
    "cpsam (ViT)": {
        "path": None, "desc": "Auto-downloads on first use (~2.4 GB)",
        "auto": True,
    },
}


def _run_cmd(cmd, log_fn):
    try:
        proc = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, cwd=PROJECT_DIR)
        for line in proc.stdout:
            log_fn(line)
        proc.wait()
        return proc.returncode == 0
    except Exception as e:
        log_fn(f"ERROR: {e}\n")
        return False


def _check_env(name):
    try:
        r = subprocess.run(
            [CONDA, "env", "list"], capture_output=True,
            text=True, timeout=10)
        return name in r.stdout
    except Exception:
        return False


def _check_model(info):
    if info.get("auto"):
        return True
    path = info.get("path")
    return path and os.path.exists(path)


def _install_deepsea(log_fn):
    """Clone DeepSea to temp dir and copy model files into project.
    Uses Python stdlib — works on all platforms."""
    tmp = tempfile.mkdtemp(prefix="deepsea_")
    log_fn(f"Cloning DeepSea to {tmp}...\n")
    ok = _run_cmd(
        f"git clone --depth 1 "
        f"https://github.com/abzargar/DeepSea.git \"{tmp}\"",
        log_fn)
    if not ok:
        log_fn("FAILED: git clone failed. Is git installed?\n")
        shutil.rmtree(tmp, ignore_errors=True)
        return False

    src = os.path.join(tmp, "deepsea")
    os.makedirs(DEEPSEA_MODEL_DIR, exist_ok=True)
    for fname in ["segmentation.pth", "tracker.pth"]:
        s = os.path.join(src, "trained_models", fname)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(DEEPSEA_MODEL_DIR, fname))
            log_fn(f"  Copied {fname}\n")
    model_py = os.path.join(src, "model.py")
    if os.path.exists(model_py):
        shutil.copy2(model_py, os.path.join(DEEPSEA_MODEL_DIR, "model.py"))
        log_fn("  Copied model.py\n")

    shutil.rmtree(tmp, ignore_errors=True)
    log_fn("DeepSea installed successfully!\n")
    return True


def _gpu_instructions():
    if IS_MAC:
        return (
            "GPU (Apple Silicon MPS):\n"
            "  Automatic — no extra steps needed.\n"
            "  MPS is detected when torch.backends.mps.is_available()\n"
            "  returns True. Requires macOS 12.3+ and PyTorch 1.12+.\n")
    elif IS_WINDOWS:
        return (
            "GPU (NVIDIA CUDA):\n"
            "  pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/cu118\n"
            "  Requires NVIDIA GPU with CUDA 11.8+ drivers.\n"
            "  Verify: python -c "
            "\"import torch; print(torch.cuda.is_available())\"\n")
    else:
        return (
            "GPU (NVIDIA CUDA):\n"
            "  pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/cu118\n"
            "  Requires NVIDIA GPU with CUDA 11.8+ drivers.\n"
            "  Verify: python -c "
            "\"import torch; print(torch.cuda.is_available())\"\n"
            "\n"
            "  If no GPU: the software falls back to CPU automatically.\n"
            "  Detection will be ~10x slower but fully functional.\n")


def _deepsea_manual_instructions():
    if IS_WINDOWS:
        return (
            "3. Install DeepSea model:\n"
            "   git clone --depth 1 "
            "https://github.com/abzargar/DeepSea.git "
            "%TEMP%\\DeepSea_tmp\n"
            "   mkdir data\\models\\deepsea\n"
            "   copy %TEMP%\\DeepSea_tmp\\deepsea\\trained_models\\* "
            "data\\models\\deepsea\\\n"
            "   copy %TEMP%\\DeepSea_tmp\\deepsea\\model.py "
            "data\\models\\deepsea\\\n"
            "   rmdir /s /q %TEMP%\\DeepSea_tmp\n\n")
    else:
        return (
            "3. Install DeepSea model:\n"
            "   git clone --depth 1 "
            "https://github.com/abzargar/DeepSea.git /tmp/DeepSea_tmp\n"
            "   mkdir -p data/models/deepsea\n"
            "   cp /tmp/DeepSea_tmp/deepsea/trained_models/* "
            "data/models/deepsea/\n"
            "   cp /tmp/DeepSea_tmp/deepsea/model.py "
            "data/models/deepsea/\n"
            "   rm -rf /tmp/DeepSea_tmp\n\n")


class SetupWizard:

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("CellScope — Setup Wizard")
        self.root.geometry("650x750")
        self.root.resizable(False, False)
        self._build_ui()
        self._refresh_status()

    def _build_ui(self):
        tk.Label(self.root, text="Setup Wizard",
                 font=("Helvetica", 16, "bold")).pack(pady=(10, 2))

        sys_frame = tk.LabelFrame(self.root, text="System", padx=8, pady=4)
        sys_frame.pack(fill="x", padx=15, pady=(0, 6))
        sys_info = _get_system_info()
        tk.Label(sys_frame, text=sys_info, font=("Courier", 9),
                 justify="left", anchor="w").pack(fill="x")

        status_frame = tk.LabelFrame(self.root, text="Status",
                                      padx=8, pady=4)
        status_frame.pack(fill="x", padx=15, pady=4)
        self._status_labels = {}

        tk.Label(status_frame, text="Environments:",
                 font=("Helvetica", 10, "bold"),
                 anchor="w").pack(fill="x")
        for name, info in ENVS.items():
            row = tk.Frame(status_frame)
            row.pack(fill="x", pady=1)
            lbl = tk.Label(row, text="  ...", font=("Helvetica", 9),
                           anchor="w", width=50)
            lbl.pack(side="left")
            btn = tk.Button(row, text="Install", width=8,
                            command=lambda n=name: self._install_env(n))
            btn.pack(side="right")
            self._status_labels[f"env_{name}"] = (lbl, btn)

        tk.Label(status_frame, text="\nModels:",
                 font=("Helvetica", 10, "bold"),
                 anchor="w").pack(fill="x")
        for name, info in MODELS.items():
            row = tk.Frame(status_frame)
            row.pack(fill="x", pady=1)
            lbl = tk.Label(row, text="  ...", font=("Helvetica", 9),
                           anchor="w", width=50)
            lbl.pack(side="left")
            if not info.get("auto"):
                btn = tk.Button(row, text="Install", width=8,
                                command=lambda n=name: self._install_model(n))
                btn.pack(side="right")
                self._status_labels[f"model_{name}"] = (lbl, btn)
            else:
                self._status_labels[f"model_{name}"] = (lbl, None)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill="x", padx=15, pady=6)
        tk.Button(btn_frame, text="Refresh",
                  command=self._refresh_status).pack(side="left", padx=3)
        tk.Button(btn_frame, text="Install All Missing",
                  command=self._install_all).pack(side="left", padx=3)
        tk.Button(btn_frame, text="Manual Instructions",
                  command=self._show_manual).pack(side="left", padx=3)
        tk.Button(btn_frame, text="Verify",
                  command=self._verify).pack(side="left", padx=3)

        tk.Label(self.root, text="Log:",
                 font=("Helvetica", 10, "bold"),
                 anchor="w").pack(fill="x", padx=15, pady=(4, 0))
        self.log = scrolledtext.ScrolledText(
            self.root, height=14, font=("Courier", 9), state="disabled")
        self.log.pack(fill="both", expand=True, padx=15, pady=(0, 10))

    def _log(self, text):
        self.log.config(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.config(state="disabled")
        self.root.update_idletasks()

    def _refresh_status(self):
        for name in ENVS:
            ok = _check_env(name)
            lbl, btn = self._status_labels[f"env_{name}"]
            sym = "\u2713" if ok else "\u2717"
            color = "#2a2" if ok else "#c22"
            lbl.config(text=f"  {sym}  {name} — {ENVS[name]['desc']}",
                       fg=color)
            if btn:
                btn.config(state="disabled" if ok else "normal")

        for name, info in MODELS.items():
            ok = _check_model(info)
            lbl, btn = self._status_labels[f"model_{name}"]
            sym = "\u2713" if ok else "\u2717"
            color = "#2a2" if ok else "#c22"
            tag = " (auto)" if info.get("auto") else ""
            lbl.config(
                text=f"  {sym}  {name}{tag} — {info['desc']}",
                fg=color)
            if btn:
                btn.config(state="disabled" if ok else "normal")

    def _install_env(self, name):
        info = ENVS[name]
        self._log(f"\n--- Installing {name} ---\n")

        def do_install():
            self._log(f"Creating conda env (python {info['python']})...\n")
            _run_cmd(
                f'"{CONDA}" create -n {name} python={info["python"]} -y',
                self._log)
            pkgs = " ".join(f'"{p}"' for p in info["pip"])
            self._log(f"\nInstalling packages...\n")
            _run_cmd(f'"{CONDA}" run -n {name} pip install {pkgs}',
                     self._log)
            self._log(f"\n{name} done.\n")
            self.root.after(0, self._refresh_status)

        threading.Thread(target=do_install, daemon=True).start()

    def _install_model(self, name):
        if name == "DeepSea":
            self._log(f"\n--- Installing DeepSea ---\n")
            threading.Thread(
                target=lambda: (
                    _install_deepsea(self._log),
                    self.root.after(0, self._refresh_status)),
                daemon=True).start()

    def _install_all(self):
        self._log(f"\n=== Installing all missing ({PLATFORM_NAME}) ===\n")

        def do_all():
            for name in ENVS:
                if not _check_env(name):
                    self._log(f"\n--- {name} ---\n")
                    info = ENVS[name]
                    _run_cmd(
                        f"conda create -n {name} "
                        f"python={info['python']} -y", self._log)
                    pkgs = " ".join(f'"{p}"' for p in info["pip"])
                    _run_cmd(
                        f"conda run -n {name} pip install {pkgs}",
                        self._log)
            if not _check_model(MODELS["DeepSea"]):
                self._log("\n--- DeepSea ---\n")
                _install_deepsea(self._log)
            self._log("\n=== Done ===\n")
            self.root.after(0, self._refresh_status)

        threading.Thread(target=do_all, daemon=True).start()

    def _verify(self):
        self._log(f"\n=== Verifying ({PLATFORM_NAME}) ===\n")

        def do_verify():
            script = (
                "import cellpose; print(f'Cellpose: {cellpose.version}');"
                "import torch; print(f'PyTorch: {torch.__version__}');"
                f"print(f'CUDA: {{torch.cuda.is_available()}}');"
                f"print(f'MPS: {{torch.backends.mps.is_available()}}');"
                "from PyQt5.QtWidgets import QApplication; "
                "print('PyQt5: OK');"
                "print('Verification OK!')")
            _run_cmd(
                f'"{CONDA}" run -n cellpose4 python -c "{script}"',
                self._log)
            self.root.after(0, self._refresh_status)

        threading.Thread(target=do_verify, daemon=True).start()

    def _show_manual(self):
        self._log(
            f"=== Manual Instructions ({PLATFORM_NAME}) ===\n\n"
            "1. Create cellpose4 environment:\n"
            "   conda create -n cellpose4 python=3.10 -y\n"
            "   conda activate cellpose4\n"
            "   pip install cellpose==4.1.1 PyQt5 matplotlib "
            "scikit-image scikit-learn scipy\n"
            "   pip install tifffile opencv-python-headless "
            "transformers huggingface_hub peft\n\n"
            "2. Create cellpose (fallback) environment:\n"
            "   conda create -n cellpose python=3.10 -y\n"
            "   conda activate cellpose\n"
            "   pip install cellpose==3.1.1.1 PyQt5 matplotlib "
            "scikit-image scikit-learn scipy\n"
            "   pip install tifffile opencv-python-headless "
            "transformers huggingface_hub peft\n\n"
            + _deepsea_manual_instructions()
            + "4. GPU setup (optional):\n"
            + _gpu_instructions() + "\n"
            "5. Verify:\n"
            "   conda activate cellpose4\n"
            "   python -c \"import cellpose; print(cellpose.version)\"\n\n"
            "See INSTALLATION.md for full details.\n")

    def run(self):
        self.root.mainloop()


def main():
    wizard = SetupWizard()
    wizard.run()


if __name__ == "__main__":
    main()
