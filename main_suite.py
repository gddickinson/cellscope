"""CellScope Suite — unified launcher.

Uses only tkinter (Python stdlib) so it runs from ANY environment,
including base conda. Each GUI is launched as a subprocess in the
correct conda environment. Max 2 instances of the same GUI at once.

Launch from any env:
    python main_suite.py
"""
import os
import sys
import subprocess
import shutil
import tkinter as tk
from tkinter import messagebox

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

APPS = [
    {
        "title": "Detection & Analysis",
        "desc": "Single-recording cpsam pipeline: detect, refine, "
                "analyze, export",
        "script": "main_focused.py",
        "env": "cellpose4",
    },
    {
        "title": "Batch Processing",
        "desc": "Process multiple recordings grouped by folder, "
                "generate summary CSVs",
        "script": "main_batch.py",
        "env": "cellpose4",
    },
    {
        "title": "Tracking & Comparison",
        "desc": "Per-cell tracking, multi-cell analysis, "
                "inter-group statistics (ANOVA)",
        "script": "main_tracking.py",
        "env": "cellpose4",
    },
    {
        "title": "Mask Editor",
        "desc": "View, edit, and create cell masks with "
                "multi-cell label support",
        "script": "main_editor.py",
        "env": "cellpose4",
    },
    {
        "title": "Model Training",
        "desc": "Fine-tune cellpose on your own ground-truth "
                "masks with live loss curve",
        "script": "main_training.py",
        "env": "cellpose4",
    },
]

MAX_INSTANCES = 1


def _find_conda(target_env="cellpose4"):
    """Find a conda executable whose envs include ``target_env``.

    Some machines have multiple conda installs (e.g. anaconda3 first
    on PATH, miniconda3 elsewhere with the cellpose4 env). Using
    shutil.which alone can pick the wrong one and produce
    ``EnvironmentLocationNotFound: Not a conda environment:
    .../envs/cellpose4`` when launching the sub-GUIs. Walk a list of
    common install roots and prefer one that actually has
    ``envs/<target_env>``.
    """
    candidates = []
    which = shutil.which("conda")
    if which:
        candidates.append(which)
    home = os.path.expanduser("~")
    for root in ["~/miniconda3", "~/anaconda3",
                  "~/opt/anaconda3", "~/opt/miniconda3",
                  "/opt/miniconda3", "/opt/anaconda3"]:
        for sub in ["bin/conda", "Scripts/conda.exe"]:
            p = os.path.join(os.path.expanduser(root), sub)
            if os.path.exists(p) and p not in candidates:
                candidates.append(p)
    # Prefer a conda whose envs/<target_env> directory exists.
    for c in candidates:
        prefix = os.path.dirname(os.path.dirname(c))
        env_dir = os.path.join(prefix, "envs", target_env)
        if os.path.isdir(env_dir):
            return c
    # Nothing matched — fall back to the first found, or "conda".
    return candidates[0] if candidates else "conda"


CONDA = _find_conda()


def _check_env(env_name):
    try:
        result = subprocess.run(
            [CONDA, "env", "list"], capture_output=True, text=True,
            timeout=10)
        return env_name in result.stdout
    except Exception:
        return False


def _detect_envs():
    envs = {}
    for app in APPS:
        env = app["env"]
        if env not in envs:
            envs[env] = _check_env(env)
    return envs


class SuiteLauncher:
    """Tkinter-based launcher with process tracking."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("CellScope Suite")
        self.root.geometry("500x500")
        self.root.resizable(False, False)
        self.env_status = _detect_envs()
        # Track running processes per app: script -> list of Popen
        self._processes = {app["script"]: [] for app in APPS}
        self._buttons = {}
        self._count_labels = {}
        self._build_ui()
        self._poll_processes()
        self._maybe_start_remote()

    def _maybe_start_remote(self):
        """Start a small HTTP server (stdlib) so this launcher can be
        driven by curl. Tkinter has no signals — but the surface we
        expose (launch_app, list_apps, status) is stateless or only
        touches subprocess.Popen, so it runs safely in the HTTP
        thread without needing tk.after() round-trips."""
        port_env = os.environ.get("CELLSCOPE_REMOTE", "").strip().lower()
        if not port_env or port_env in ("0", "false", "off", "no"):
            return
        if port_env in ("1", "true", "on", "yes"):
            port = 8766
        else:
            try:
                port = int(port_env)
            except ValueError:
                return
        import threading, json, queue
        from http.server import (BaseHTTPRequestHandler, HTTPServer)
        from urllib.parse import urlparse

        launcher = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *_): pass

            def _json(self, status, body):
                payload = json.dumps(body, default=str).encode()
                self.send_response(status)
                self.send_header("Content-Type",
                                  "application/json")
                self.send_header("Content-Length",
                                  str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def _read_body(self):
                n = int(self.headers.get("Content-Length", 0))
                if not n: return {}
                raw = self.rfile.read(n)
                try: return json.loads(raw)
                except Exception: return {}

            def do_GET(self):
                p = urlparse(self.path).path
                if p == "/status":
                    counts = {s: len(ps) for s, ps in
                              launcher._processes.items()}
                    self._json(200, {"gui": "launcher",
                                      "port": port,
                                      "apps": [a["title"]
                                                for a in APPS],
                                      "running": counts})
                    return
                if p == "/list_apps":
                    self._json(200, {
                        "apps": [{"title": a["title"],
                                   "script": a["script"],
                                   "env": a["env"]}
                                  for a in APPS]})
                    return
                self._json(404, {"error": f"unknown {p}"})

            def do_POST(self):
                p = urlparse(self.path).path
                data = self._read_body()
                if p == "/launch_app":
                    script = data.get("script")
                    app = next((a for a in APPS
                                 if a["script"] == script), None)
                    if app is None:
                        self._json(404,
                            {"error": f"unknown script {script!r}"})
                        return
                    script_path = os.path.join(PROJECT_DIR, script)
                    proc = subprocess.Popen(
                        [CONDA, "run", "--no-capture-output",
                         "-n", app["env"], "python", script_path],
                        cwd=PROJECT_DIR,
                        env={**os.environ,
                              "CELLSCOPE_REMOTE": str(
                                  data.get("remote_port", "1"))})
                    launcher._processes[script].append(proc)
                    self._json(200, {"ok": True, "pid": proc.pid,
                                      "script": script})
                    return
                self._json(404, {"error": f"unknown {p}"})

        try:
            httpd = HTTPServer(("127.0.0.1", port), _Handler)
        except OSError as e:
            print(f"launcher remote control: bind failed: {e}")
            return
        threading.Thread(
            target=httpd.serve_forever,
            daemon=True,
            name=f"LauncherRemote-{port}").start()
        print(f"launcher remote control: http://127.0.0.1:{port}")

    def _build_ui(self):
        tk.Label(self.root, text="CellScope Suite",
                 font=("Helvetica", 18, "bold")).pack(pady=(15, 2))
        tk.Label(self.root,
                 text="DIC / phase-contrast time-lapse cell analysis",
                 font=("Helvetica", 10), fg="#666").pack()
        tk.Label(self.root,
                 text="Detection \u2022 Tracking \u2022 Morphology "
                      "\u2022 Edge Dynamics",
                 font=("Helvetica", 9), fg="#888").pack(pady=(0, 8))

        env_frame = tk.Frame(self.root)
        env_frame.pack(fill="x", padx=20)
        for env_name, available in self.env_status.items():
            color = "#2a2" if available else "#c22"
            symbol = "\u2713" if available else "\u2717"
            tk.Label(env_frame, text=f"  {symbol} {env_name}",
                     font=("Helvetica", 9), fg=color).pack(
                         side="left", padx=5)

        tk.Frame(self.root, height=1, bg="#ccc").pack(
            fill="x", padx=20, pady=8)

        for app in APPS:
            self._add_app_row(app)

        tk.Frame(self.root, height=1, bg="#ccc").pack(
            fill="x", padx=20, pady=4)

        setup_frame = tk.Frame(self.root)
        setup_frame.pack(fill="x", padx=20, pady=2)
        tk.Button(setup_frame, text="Setup Wizard",
                  command=self._open_setup,
                  width=14).pack(side="left")
        tk.Label(setup_frame,
                 text="  Install/check environments, models, "
                      "and dependencies",
                 font=("Helvetica", 9), fg="#666").pack(
                     side="left")

        tk.Label(self.root,
                 text="See INSTALLATION.md for setup instructions",
                 font=("Helvetica", 8), fg="#999").pack(
                     side="bottom", pady=5)

    def _add_app_row(self, app):
        env = app["env"]
        available = self.env_status.get(env, False)
        script = app["script"]

        btn_frame = tk.Frame(self.root, relief="groove", bd=1,
                             padx=10, pady=6)
        btn_frame.pack(fill="x", padx=20, pady=3)

        title_text = app["title"]
        if not available:
            title_text += f"  (needs {env})"

        tk.Label(btn_frame, text=title_text,
                 font=("Helvetica", 12, "bold"),
                 anchor="w").pack(fill="x")
        tk.Label(btn_frame, text=app["desc"],
                 font=("Helvetica", 9), fg="#555",
                 anchor="w", wraplength=380).pack(fill="x")

        right_frame = tk.Frame(btn_frame)
        right_frame.place(relx=1.0, rely=0.5, anchor="e")

        count_lbl = tk.Label(right_frame, text="",
                             font=("Helvetica", 8), fg="#888")
        count_lbl.pack(side="left", padx=(0, 4))
        self._count_labels[script] = count_lbl

        btn = tk.Button(right_frame, text="Launch",
                        command=lambda a=app: self._launch(a),
                        state="normal" if available else "disabled",
                        width=8)
        btn.pack(side="left")
        self._buttons[script] = (btn, available)

    def _launch(self, app):
        script = app["script"]
        env = app["env"]

        self._clean_dead(script)
        if len(self._processes[script]) >= MAX_INSTANCES:
            messagebox.showinfo(
                "Already Running",
                f"'{app['title']}' is already open.\n\n"
                f"Close it before launching another instance.")
            return

        script_path = os.path.join(PROJECT_DIR, script)
        if not os.path.exists(script_path):
            messagebox.showerror("File Not Found",
                                 f"Script not found: {script_path}")
            return

        try:
            proc = subprocess.Popen(
                [CONDA, "run", "--no-capture-output", "-n", env,
                 "python", script_path],
                cwd=PROJECT_DIR,
            )
            self._processes[script].append(proc)
            self._update_button(script)
        except Exception as e:
            messagebox.showerror("Launch Error", str(e))

    def _clean_dead(self, script):
        """Remove finished processes from the list."""
        self._processes[script] = [
            p for p in self._processes[script]
            if p.poll() is None]

    def _update_button(self, script):
        """Update button state and count label."""
        self._clean_dead(script)
        n = len(self._processes[script])
        btn, env_available = self._buttons[script]

        if not env_available:
            btn.config(state="disabled")
            self._count_labels[script].config(text="")
        elif n >= MAX_INSTANCES:
            btn.config(state="disabled")
            self._count_labels[script].config(
                text=f"{n}/{MAX_INSTANCES}", fg="#c22")
        else:
            btn.config(state="normal")
            if n > 0:
                self._count_labels[script].config(
                    text=f"{n}/{MAX_INSTANCES}", fg="#888")
            else:
                self._count_labels[script].config(text="")

    def _open_setup(self):
        """Launch the setup wizard as a subprocess."""
        script = os.path.join(PROJECT_DIR, "setup_wizard.py")
        subprocess.Popen([sys.executable, script], cwd=PROJECT_DIR)

    def _poll_processes(self):
        """Periodically check if child processes have exited."""
        for script in self._processes:
            self._update_button(script)
        self.root.after(1000, self._poll_processes)

    def run(self):
        self.root.mainloop()


def main():
    launcher = SuiteLauncher()
    launcher.run()


if __name__ == "__main__":
    main()
