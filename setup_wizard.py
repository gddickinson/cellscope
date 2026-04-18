"""Setup wizard — check, install, and configure environments and models.

Uses only tkinter + stdlib. Platform-aware: works on macOS, Linux,
and Windows. Can install conda environments, pip packages, and
download required models. Includes dependency checker.

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
from tkinter import scrolledtext, messagebox, ttk

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
IS_WINDOWS = sys.platform == "win32"
IS_MAC = sys.platform == "darwin"
PLATFORM_NAME = {"darwin": "macOS", "win32": "Windows",
                 "linux": "Linux"}.get(sys.platform, sys.platform)

# Complete package lists matching requirements.txt
PIP_PACKAGES_CP4 = [
    "cellpose==4.1.1", "torch>=2.0", "torchvision",
    "numpy", "scipy", "scikit-image", "scikit-learn",
    "opencv-python-headless", "matplotlib", "tifffile",
    "PyQt5", "transformers", "huggingface_hub", "peft",
    "vampire-analysis",
]

PIP_PACKAGES_CP3 = [
    "cellpose==3.1.1.1", "torch>=2.0", "torchvision",
    "numpy", "scipy", "scikit-image", "scikit-learn",
    "opencv-python-headless", "matplotlib", "tifffile",
    "PyQt5", "transformers", "huggingface_hub", "peft",
]

ENVS = {
    "cellpose4": {
        "python": "3.10",
        "pip": PIP_PACKAGES_CP4,
        "desc": "Primary environment (cpsam ViT detection)",
    },
    "cellpose": {
        "python": "3.10",
        "pip": PIP_PACKAGES_CP3,
        "desc": "Fallback environment (CP3 models)",
    },
}

# Import names differ from pip names for some packages
IMPORT_MAP = {
    "cellpose==4.1.1": "cellpose",
    "cellpose==3.1.1.1": "cellpose",
    "cellpose>=4.1.0": "cellpose",
    "torch>=2.0": "torch",
    "scikit-image": "skimage",
    "scikit-learn": "sklearn",
    "opencv-python-headless": "cv2",
    "huggingface_hub": "huggingface_hub",
    "vampire-analysis": "vampire",
    "PyQt5": "PyQt5",
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


def _check_package_in_env(env_name, import_name):
    """Check if a package is importable in a conda env."""
    try:
        r = subprocess.run(
            [CONDA, "run", "-n", env_name, "python", "-c",
             f"import {import_name}"],
            capture_output=True, text=True, timeout=15)
        return r.returncode == 0
    except Exception:
        return False


def _install_deepsea(log_fn):
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


def _get_system_info():
    import multiprocessing
    lines = [
        f"OS:       {PLATFORM_NAME} {platform.release()}",
        f"Machine:  {platform.machine()}",
        f"Python:   {platform.python_version()}",
        f"CPU:      {multiprocessing.cpu_count()} cores",
    ]
    try:
        if not IS_WINDOWS:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            ram_gb = (pages * page_size) / (1024 ** 3)
            lines.append(f"RAM:      {ram_gb:.1f} GB")
    except Exception:
        lines.append("RAM:      unknown")
    try:
        usage = shutil.disk_usage(PROJECT_DIR)
        lines.append(f"Disk:     {usage.free / (1024**3):.0f} GB free "
                     f"/ {usage.total / (1024**3):.0f} GB total")
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            lines.append(
                f"GPU:      CUDA — {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and \
                torch.backends.mps.is_available():
            lines.append("GPU:      Apple MPS (Metal)")
        else:
            lines.append("GPU:      None detected (CPU only)")
    except ImportError:
        lines.append("GPU:      PyTorch not installed (unknown)")
    try:
        r = subprocess.run([CONDA, "--version"], capture_output=True,
                           text=True, timeout=5)
        lines.append(f"Conda:    {r.stdout.strip()}")
    except Exception:
        lines.append("Conda:    not found on PATH")
    try:
        r = subprocess.run(["git", "--version"], capture_output=True,
                           text=True, timeout=5)
        lines.append(f"Git:      {r.stdout.strip()}")
    except Exception:
        lines.append("Git:      not found")
    return "\n".join(lines)


def _gpu_instructions():
    if IS_MAC:
        return (
            "GPU (Apple Silicon MPS):\n"
            "  Automatic — no extra steps needed.\n"
            "  Requires macOS 12.3+ and PyTorch 1.12+.\n")
    else:
        return (
            "GPU (NVIDIA CUDA):\n"
            "  pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/cu118\n"
            "  Requires NVIDIA GPU with CUDA 11.8+ drivers.\n")


def _deepsea_manual_instructions():
    if IS_WINDOWS:
        return (
            "   git clone --depth 1 "
            "https://github.com/abzargar/DeepSea.git "
            "%TEMP%\\DeepSea_tmp\n"
            "   mkdir data\\models\\deepsea\n"
            "   copy %TEMP%\\DeepSea_tmp\\deepsea\\trained_models\\* "
            "data\\models\\deepsea\\\n"
            "   copy %TEMP%\\DeepSea_tmp\\deepsea\\model.py "
            "data\\models\\deepsea\\\n"
            "   rmdir /s /q %TEMP%\\DeepSea_tmp\n")
    else:
        return (
            "   git clone --depth 1 "
            "https://github.com/abzargar/DeepSea.git /tmp/DeepSea_tmp\n"
            "   mkdir -p data/models/deepsea\n"
            "   cp /tmp/DeepSea_tmp/deepsea/trained_models/* "
            "data/models/deepsea/\n"
            "   cp /tmp/DeepSea_tmp/deepsea/model.py "
            "data/models/deepsea/\n"
            "   rm -rf /tmp/DeepSea_tmp\n")


class SetupWizard:

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("CellScope — Setup Wizard")
        self.root.geometry("700x780")
        self.root.resizable(False, False)
        self._build_ui()
        self._refresh_status()

    def _build_ui(self):
        tk.Label(self.root, text="CellScope Setup Wizard",
                 font=("Helvetica", 16, "bold")).pack(pady=(10, 2))

        # System info
        sys_frame = tk.LabelFrame(self.root, text="System", padx=8,
                                   pady=4)
        sys_frame.pack(fill="x", padx=15, pady=(0, 4))
        sys_info = _get_system_info()
        tk.Label(sys_frame, text=sys_info, font=("Courier", 9),
                 justify="left", anchor="w").pack(fill="x")

        # Notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=15, pady=4)

        # Tab 1: Environments & Models
        self._build_envmodels_tab()
        # Tab 2: Dependencies
        self._build_deps_tab()

        # Log
        tk.Label(self.root, text="Log:",
                 font=("Helvetica", 10, "bold"),
                 anchor="w").pack(fill="x", padx=15, pady=(4, 0))
        self.log = scrolledtext.ScrolledText(
            self.root, height=8, font=("Courier", 9), state="disabled")
        self.log.pack(fill="both", expand=True, padx=15, pady=(0, 10))

    def _build_envmodels_tab(self):
        tab = tk.Frame(self.notebook)
        self.notebook.add(tab, text="Environments & Models")
        self._status_labels = {}

        tk.Label(tab, text="Environments:",
                 font=("Helvetica", 10, "bold"),
                 anchor="w").pack(fill="x", padx=8, pady=(8, 0))
        for name, info in ENVS.items():
            row = tk.Frame(tab)
            row.pack(fill="x", padx=8, pady=1)
            lbl = tk.Label(row, text="  ...", font=("Helvetica", 9),
                           anchor="w", width=55)
            lbl.pack(side="left")
            btn = tk.Button(row, text="Install", width=8,
                            command=lambda n=name: self._install_env(n))
            btn.pack(side="right")
            self._status_labels[f"env_{name}"] = (lbl, btn)

        tk.Label(tab, text="\nModels:",
                 font=("Helvetica", 10, "bold"),
                 anchor="w").pack(fill="x", padx=8)
        for name, info in MODELS.items():
            row = tk.Frame(tab)
            row.pack(fill="x", padx=8, pady=1)
            lbl = tk.Label(row, text="  ...", font=("Helvetica", 9),
                           anchor="w", width=55)
            lbl.pack(side="left")
            if not info.get("auto"):
                btn = tk.Button(row, text="Install", width=8,
                                command=lambda n=name:
                                    self._install_model(n))
                btn.pack(side="right")
                self._status_labels[f"model_{name}"] = (lbl, btn)
            else:
                self._status_labels[f"model_{name}"] = (lbl, None)

        btn_frame = tk.Frame(tab)
        btn_frame.pack(fill="x", padx=8, pady=6)
        tk.Button(btn_frame, text="Refresh",
                  command=self._refresh_status).pack(side="left", padx=3)
        tk.Button(btn_frame, text="Install All Missing",
                  command=self._install_all).pack(side="left", padx=3)
        tk.Button(btn_frame, text="Manual Instructions",
                  command=self._show_manual).pack(side="left", padx=3)
        tk.Button(btn_frame, text="Verify",
                  command=self._verify).pack(side="left", padx=3)

    def _build_deps_tab(self):
        tab = tk.Frame(self.notebook)
        self.notebook.add(tab, text="Dependencies")

        tk.Label(tab, text="Check individual package availability:",
                 font=("Helvetica", 10),
                 anchor="w").pack(fill="x", padx=8, pady=(8, 4))

        env_row = tk.Frame(tab)
        env_row.pack(fill="x", padx=8)
        tk.Label(env_row, text="Environment:").pack(side="left")
        self.dep_env = ttk.Combobox(env_row, values=list(ENVS.keys()),
                                     state="readonly", width=15)
        self.dep_env.set("cellpose4")
        self.dep_env.pack(side="left", padx=4)
        tk.Button(env_row, text="Check All Packages",
                  command=self._check_deps).pack(side="left", padx=4)
        tk.Button(env_row, text="Install Missing",
                  command=self._install_missing_deps).pack(
                      side="left", padx=4)

        # Deps tree
        cols = ("Package", "Import Name", "Status")
        self.dep_tree = ttk.Treeview(tab, columns=cols,
                                      show="headings", height=12)
        for c in cols:
            self.dep_tree.heading(c, text=c)
            self.dep_tree.column(c, width=180 if c == "Package" else 120)
        self.dep_tree.pack(fill="both", expand=True, padx=8, pady=4)

        self.dep_status_label = tk.Label(
            tab, text="Click 'Check All Packages' to scan",
            font=("Helvetica", 9), fg="#666")
        self.dep_status_label.pack(fill="x", padx=8, pady=2)

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

    def _check_deps(self):
        env = self.dep_env.get()
        if not _check_env(env):
            self.dep_status_label.config(
                text=f"Environment '{env}' not found", fg="#c22")
            return
        self.dep_status_label.config(
            text=f"Checking packages in {env}...", fg="#888")
        self.root.update_idletasks()

        def do_check():
            self.dep_tree.delete(*self.dep_tree.get_children())
            pkgs = ENVS[env]["pip"]
            found, missing = 0, 0
            for pkg in pkgs:
                base = pkg.split(">=")[0].split("==")[0]
                imp = IMPORT_MAP.get(pkg, base)
                ok = _check_package_in_env(env, imp)
                status = "\u2713 Installed" if ok else "\u2717 Missing"
                tag = "ok" if ok else "missing"
                self.dep_tree.insert("", "end",
                                     values=(pkg, imp, status),
                                     tags=(tag,))
                if ok:
                    found += 1
                else:
                    missing += 1
            self.dep_tree.tag_configure("ok", foreground="#2a2")
            self.dep_tree.tag_configure("missing", foreground="#c22")
            self.dep_status_label.config(
                text=f"{found} installed, {missing} missing in {env}",
                fg="#2a2" if missing == 0 else "#c22")

        threading.Thread(target=do_check, daemon=True).start()

    def _install_missing_deps(self):
        env = self.dep_env.get()
        if not _check_env(env):
            return
        missing = []
        for item in self.dep_tree.get_children():
            vals = self.dep_tree.item(item, "values")
            if "\u2717" in vals[2]:
                missing.append(vals[0])
        if not missing:
            self._log("No missing packages.\n")
            return
        self._log(f"\n--- Installing {len(missing)} missing packages "
                  f"in {env} ---\n")

        def do_install():
            pkgs = " ".join(f'"{p}"' for p in missing)
            _run_cmd(f'"{CONDA}" run -n {env} pip install {pkgs}',
                     self._log)
            self._log("Done. Click 'Check All Packages' to verify.\n")

        threading.Thread(target=do_install, daemon=True).start()

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
                        f'"{CONDA}" create -n {name} '
                        f'python={info["python"]} -y', self._log)
                    pkgs = " ".join(f'"{p}"' for p in info["pip"])
                    _run_cmd(
                        f'"{CONDA}" run -n {name} pip install {pkgs}',
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
                "import vampire; print('vampire-analysis: OK');"
                "print('Verification OK!')")
            _run_cmd(
                f'"{CONDA}" run -n cellpose4 python -c "{script}"',
                self._log)
            self.root.after(0, self._refresh_status)

        threading.Thread(target=do_verify, daemon=True).start()

    def _show_manual(self):
        pip4 = " ".join(PIP_PACKAGES_CP4)
        pip3 = " ".join(PIP_PACKAGES_CP3)
        self._log(
            f"=== Manual Instructions ({PLATFORM_NAME}) ===\n\n"
            "1. Create cellpose4 environment:\n"
            "   conda create -n cellpose4 python=3.10 -y\n"
            "   conda activate cellpose4\n"
            f"   pip install {pip4}\n\n"
            "2. Create cellpose (fallback) environment:\n"
            "   conda create -n cellpose python=3.10 -y\n"
            "   conda activate cellpose\n"
            f"   pip install {pip3}\n\n"
            "3. Install DeepSea model:\n"
            + _deepsea_manual_instructions() + "\n"
            "4. GPU setup (optional):\n"
            + _gpu_instructions() + "\n"
            "5. Verify:\n"
            "   conda activate cellpose4\n"
            "   python -c \"import cellpose; print(cellpose.version)\"\n"
            "   python -c \"import vampire; print('VAMPIRE OK')\"\n\n"
            "See INSTALLATION.md for full details.\n")

    def run(self):
        self.root.mainloop()


def main():
    wizard = SetupWizard()
    wizard.run()


if __name__ == "__main__":
    main()
