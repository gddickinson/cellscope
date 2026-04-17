"""Run logger: captures every pipeline step with params, timings, and stats.

Two parts:
  - RunLogger: in-memory record of events. Events are dicts with timestamp,
    kind, message, and optional params/stats. Saves to run_log.md and
    run_log.json next to the results.
  - RunLogWidget: scrolling Qt widget that displays events live.

The logger instance is shared between the main view and the background
workers (workers emit log entries through Qt signals).
"""
import os
import json
import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, List

from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtGui import QFont, QTextCursor
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPlainTextEdit, QPushButton,
    QFileDialog, QLabel, QCheckBox,
)


# --- Event data ---

@dataclass
class LogEvent:
    """One logged event."""
    timestamp: str  # ISO 8601
    kind: str  # "info" | "start" | "done" | "warn" | "error" | "params"
    message: str
    details: dict = field(default_factory=dict)  # arbitrary JSON-serializable

    def to_dict(self) -> dict:
        return asdict(self)


# --- Logger (non-UI, thread-safe enough for Qt signals) ---

class RunLogger(QObject):
    """In-memory event log that also emits Qt signals for the widget."""
    event_added = pyqtSignal(object)  # payload = LogEvent

    def __init__(self):
        super().__init__()
        self.events: List[LogEvent] = []

    # --- Low-level ---
    def log(self, kind: str, message: str, details: Optional[dict] = None):
        ev = LogEvent(
            timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
            kind=kind,
            message=message,
            details=details or {},
        )
        self.events.append(ev)
        self.event_added.emit(ev)
        return ev

    # --- Convenience ---
    def info(self, msg, **details):
        return self.log("info", msg, details or None)

    def start(self, msg, **details):
        return self.log("start", msg, details or None)

    def done(self, msg, **details):
        return self.log("done", msg, details or None)

    def warn(self, msg, **details):
        return self.log("warn", msg, details or None)

    def error(self, msg, **details):
        return self.log("error", msg, details or None)

    def params(self, label: str, params_dict: dict):
        """Log a full parameter dump under one event."""
        return self.log("params", f"Parameters: {label}",
                        {"params": params_dict})

    # --- Serialization ---
    def clear(self):
        self.events.clear()

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(
            [e.to_dict() for e in self.events], indent=indent
        )

    def to_markdown(self) -> str:
        """Render as human-readable markdown."""
        lines = ["# Run Log", ""]
        icons = {
            "info": "·", "start": "▶", "done": "✓",
            "warn": "⚠", "error": "✗", "params": "⚙",
        }
        for ev in self.events:
            icon = icons.get(ev.kind, "·")
            lines.append(f"- `{ev.timestamp}` {icon} **{ev.kind}** — "
                         f"{ev.message}")
            if ev.details:
                # Render params as fenced json; other details as key: value
                if "params" in ev.details and len(ev.details) == 1:
                    lines.append("  ```json")
                    for line in json.dumps(
                            ev.details["params"], indent=2
                    ).splitlines():
                        lines.append(f"  {line}")
                    lines.append("  ```")
                else:
                    for k, v in ev.details.items():
                        lines.append(f"  - {k}: `{v}`")
        lines.append("")
        return "\n".join(lines)

    def save(self, out_dir: str) -> dict:
        """Write run_log.md and run_log.json into `out_dir`.

        Returns dict with paths written.
        """
        os.makedirs(out_dir, exist_ok=True)
        md_path = os.path.join(out_dir, "run_log.md")
        js_path = os.path.join(out_dir, "run_log.json")
        with open(md_path, "w") as f:
            f.write(self.to_markdown())
        with open(js_path, "w") as f:
            f.write(self.to_json())
        return {"markdown": md_path, "json": js_path}


# --- Widget ---

class RunLogWidget(QWidget):
    """Live-scrolling display of the run log."""

    def __init__(self, logger: RunLogger, parent=None):
        super().__init__(parent)
        self.logger = logger
        self._build_ui()
        logger.event_added.connect(self._on_event)
        # Populate with any pre-existing events
        for ev in logger.events:
            self._append(ev)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        bar = QHBoxLayout()
        self.chk_autoscroll = QCheckBox("Auto-scroll")
        self.chk_autoscroll.setChecked(True)
        bar.addWidget(self.chk_autoscroll)

        self.chk_details = QCheckBox("Show details")
        self.chk_details.setChecked(True)
        self.chk_details.toggled.connect(self._reload)
        bar.addWidget(self.chk_details)

        bar.addStretch()

        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self._on_clear)
        bar.addWidget(btn_clear)

        btn_save = QPushButton("Save log…")
        btn_save.clicked.connect(self._on_save)
        bar.addWidget(btn_save)

        bar.addWidget(QLabel("   "))
        self.count_label = QLabel("0 events")
        bar.addWidget(self.count_label)

        layout.addLayout(bar)

        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setFont(QFont("Menlo", 10))
        self.text.setLineWrapMode(QPlainTextEdit.NoWrap)
        layout.addWidget(self.text, 1)

    # --- Rendering ---
    _KIND_ICONS = {
        "info": "·", "start": "▶", "done": "✓",
        "warn": "⚠", "error": "✗", "params": "⚙",
    }

    def _format(self, ev: LogEvent) -> str:
        icon = self._KIND_ICONS.get(ev.kind, "·")
        ts = ev.timestamp.split("T")[-1] if "T" in ev.timestamp else ev.timestamp
        head = f"[{ts}] {icon} {ev.kind:6s} {ev.message}"
        if not self.chk_details.isChecked() or not ev.details:
            return head
        lines = [head]
        if "params" in ev.details and len(ev.details) == 1:
            for line in json.dumps(
                    ev.details["params"], indent=2
            ).splitlines():
                lines.append(f"        {line}")
        else:
            for k, v in ev.details.items():
                # Compact multi-line values
                sv = str(v)
                if len(sv) > 200:
                    sv = sv[:200] + "…"
                lines.append(f"        {k}: {sv}")
        return "\n".join(lines)

    def _append(self, ev: LogEvent):
        self.text.appendPlainText(self._format(ev))
        if self.chk_autoscroll.isChecked():
            self.text.moveCursor(QTextCursor.End)
        self._update_count()

    def _reload(self):
        self.text.clear()
        for ev in self.logger.events:
            self.text.appendPlainText(self._format(ev))
        self._update_count()
        if self.chk_autoscroll.isChecked():
            self.text.moveCursor(QTextCursor.End)

    def _update_count(self):
        self.count_label.setText(f"{len(self.logger.events)} events")

    def _on_event(self, ev: LogEvent):
        self._append(ev)

    def _on_clear(self):
        self.logger.clear()
        self.text.clear()
        self._update_count()

    def _on_save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save run log", "run_log.md",
            "Markdown (*.md);;JSON (*.json);;All files (*.*)"
        )
        if not path:
            return
        if path.endswith(".json"):
            with open(path, "w") as f:
                f.write(self.logger.to_json())
        else:
            with open(path, "w") as f:
                f.write(self.logger.to_markdown())
