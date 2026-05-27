"""Localhost-bound HTTP control server for the focused GUI.

Lets another process (curl, Python requests, an LLM assistant)
drive the GUI for automated testing and inspection:

  curl -sS -X POST http://127.0.0.1:8765/load_recording \\
      -H 'Content-Type: application/json' \\
      -d '{"path": "/path/to/file.ome.tif", "dic_channel": 1, "fluo_channel": 0}'

  curl -sS -X POST http://127.0.0.1:8765/set_frame -d '{"index": 16}'
  curl -sS -X POST http://127.0.0.1:8765/save_screenshot \\
      -d '{"path": "/tmp/snap.png", "viewer_only": true}'
  curl -sS http://127.0.0.1:8765/status

Started by FocusedMainWindow when CELLSCOPE_REMOTE env var is set
to '1', 'true', or a port number (default 8765). Tries successive
ports if the chosen one is taken so multiple GUIs can coexist.

Endpoints (POST unless noted, JSON body, JSON response):

  POST /load_recording        path[, dic_channel, fluo_channel]
  POST /load_pipeline_results path
  POST /load_project          path
  POST /clear_all
  POST /set_param             name, value
  POST /set_frame             index
  POST /set_view              control, checked
      control ∈ {mask, contour, ids, tracks, source, dropped,
                  channel}; channel value: "dic" or "fluo"
  POST /detect
  POST /test_frame
  POST /save_screenshot       path[, viewer_only=false]
  POST /save_project          path
  GET  /status
  GET  /params
  GET  /log                   ?limit=50

All command handlers run on the Qt main thread via a queued signal —
HTTP threads never touch widgets directly.
"""
import json
import logging
import os
import queue
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

from PyQt5.QtCore import QObject, pyqtSignal

log = logging.getLogger(__name__)

DEFAULT_PORT = 8765
MAX_PORT_TRIES = 10


class RemoteControlServer(QObject):
    """HTTP server bound to 127.0.0.1, dispatches JSON commands to the
    GUI's main thread via a Qt signal."""

    command_received = pyqtSignal(dict, object)

    def __init__(self, parent=None):
        # Some "window" classes (e.g. gui_editor.EditorWindow) are plain
        # Python wrappers around a QMainWindow rather than QObjects
        # themselves. Only pass parent to super if it's actually a QObject,
        # otherwise hand back to QObject's default (None).
        super().__init__(parent if isinstance(parent, QObject)
                          else None)
        self._httpd = None
        self._thread = None
        self._port = None

    def start(self, port=DEFAULT_PORT):
        """Bind to 127.0.0.1:port. If taken, try port+1, port+2, …
        Returns the bound port, or None on failure."""
        for attempt in range(MAX_PORT_TRIES):
            try_port = port + attempt
            try:
                handler_cls = _make_handler(self)
                self._httpd = HTTPServer(("127.0.0.1", try_port),
                                            handler_cls)
                self._port = try_port
                break
            except OSError as e:
                if e.errno not in (48, 98):   # EADDRINUSE
                    log.warning("Remote control bind failed: %s", e)
                    return None
        if self._httpd is None:
            log.warning("Remote control: no free port in range "
                        "%d-%d", port, port + MAX_PORT_TRIES - 1)
            return None
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            daemon=True,
            name=f"RemoteControl-{self._port}")
        self._thread.start()
        log.info("Remote control listening on http://127.0.0.1:%d",
                 self._port)
        return self._port

    def stop(self):
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        self._port = None

    @property
    def port(self):
        return self._port


def _make_handler(server):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            # Silence the default per-request stderr noise; we log on
            # the dispatch side instead.
            pass

        def do_GET(self):
            self._dispatch(method="GET")

        def do_POST(self):
            self._dispatch(method="POST")

        def _dispatch(self, method):
            parsed = urlparse(self.path)
            data = {}
            if method == "POST":
                n = int(self.headers.get("Content-Length", 0))
                if n:
                    body = self.rfile.read(n)
                    try:
                        data = json.loads(body)
                    except json.JSONDecodeError:
                        self._send_json(
                            400, {"error": "invalid JSON body"})
                        return
            else:
                # GET query string → data
                for k, v in parse_qs(parsed.query).items():
                    data[k] = v[0] if len(v) == 1 else v
            command = {
                "method": method,
                "path": parsed.path,
                "data": data,
            }
            reply_q = queue.Queue(maxsize=1)
            server.command_received.emit(command, reply_q)
            try:
                response = reply_q.get(timeout=180)
            except queue.Empty:
                self._send_json(
                    504, {"error": "GUI handler timed out (180s)"})
                return
            status = response.get("status", 200)
            body = response.get("body", {})
            self._send_json(status, body)

        def _send_json(self, status, body):
            payload = json.dumps(
                body, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return Handler


def parse_remote_env(default_port=DEFAULT_PORT):
    """Read CELLSCOPE_REMOTE env var → port number, or None to skip.

    Accepted values:
      unset / '0' / 'false' / ''   → None (disabled)
      '1' / 'true' / 'on'          → default_port
      <integer string>             → that port
    """
    val = os.environ.get("CELLSCOPE_REMOTE", "").strip().lower()
    if not val or val in ("0", "false", "off", "no"):
        return None
    if val in ("1", "true", "on", "yes"):
        return default_port
    try:
        return int(val)
    except ValueError:
        log.warning("CELLSCOPE_REMOTE=%r not understood; ignoring",
                    val)
        return None


def generic_save_screenshot(window):
    """Returns a handler that saves a PNG of the window."""
    def handler(data):
        path = data.get("path")
        if not path:
            raise ValueError("path required")
        pix = window.grab()
        if not pix.save(path, "PNG"):
            raise RuntimeError("QPixmap.save returned False")
        return {"ok": True, "path": path}
    return handler


def generic_close(window):
    """Returns a handler that closes the window."""
    def handler(data):
        window.close()
        return {"ok": True}
    return handler


def attach_minimal(window, gui_type, default_port=DEFAULT_PORT,
                    extra_handlers=None, status_extra=None):
    """Lightweight wrapper: any QMainWindow gets /status and
    /save_screenshot and /close for free. ``extra_handlers`` dict is
    merged on top. ``status_extra`` (callable returning dict) is
    merged into the status response.
    """
    handlers = {
        "/status":          lambda d: (status_extra(d) if status_extra
                                        else {}),
        "/save_screenshot": generic_save_screenshot(window),
        "/close":           generic_close(window),
    }
    if extra_handlers:
        handlers.update(extra_handlers)
    return attach(window, gui_type=gui_type, handlers=handlers,
                   default_port=default_port)


def attach(window, gui_type, handlers, default_port=DEFAULT_PORT,
            status_logger=None):
    """Start a remote control server attached to ``window`` (any
    QObject) and wire its command_received signal to ``handlers``.

    Returns the bound port or None if disabled / failed.

    ``handlers`` is a dict {path: callable(data) → body | None}.
    Two paths are auto-added if not provided:
      GET /status    → {"gui": <gui_type>, ...handler('/status')}
      GET /log       → {"lines": []} (override if your GUI has a log)

    ``status_logger``, if given, is called with (kind, message) for
    server lifecycle events (useful when the GUI has a built-in log
    widget). Falls back to the module logger otherwise.

    Each handler should accept a single dict argument (the request
    body, may be empty) and return a JSON-serialisable response body,
    or None for {"ok": True}.
    """
    port = parse_remote_env(default_port=default_port)
    if port is None:
        return None
    server = RemoteControlServer(window)

    def _logmsg(kind, msg):
        if status_logger is not None:
            try:
                status_logger(kind, msg)
                return
            except Exception:
                pass
        log.info("[%s] %s", kind, msg)

    def _dispatch(command, reply_q):
        path = command.get("path", "")
        data = command.get("data", {}) or {}
        try:
            handler = handlers.get(path)
            if handler is None:
                if path == "/status":
                    reply_q.put({"status": 200, "body": {
                        "gui": gui_type, "port": server.port}})
                    return
                reply_q.put({
                    "status": 404,
                    "body": {"error": f"unknown {path}",
                              "known": sorted(handlers.keys())}})
                return
            body = handler(data)
            if body is None:
                body = {"ok": True}
            # Auto-tag status response with gui_type for discovery
            if path == "/status" and isinstance(body, dict):
                body.setdefault("gui", gui_type)
                body.setdefault("port", server.port)
            reply_q.put({"status": 200, "body": body})
        except Exception as e:
            import traceback
            reply_q.put({"status": 500, "body": {
                "error": str(e),
                "traceback": traceback.format_exc()}})

    server.command_received.connect(_dispatch)
    bound = server.start(port=port)
    if bound is not None:
        _logmsg("info",
                f"Remote control ({gui_type}): http://127.0.0.1:{bound}")
    # Store so it's not GC'd
    window._remote = server
    return bound
