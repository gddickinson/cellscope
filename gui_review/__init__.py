"""Rapid single-cell review GUI.

Repurposes the annotation GUI's zoomable-crop + LRU-cache + CSV pattern
for fast QC of curated single-cell recordings: step through a cell's
frames with the arrow keys, scroll to zoom, flag problematic cells with a
keypress. Auto-discovers all recordings; flags persist to a CSV.

Contents:
  review_data.py   — discover_recordings, ReviewRenderer (single-channel,
                     LRU cache), trajectory_bbox, FlagStore (CSV).
  review_window.py — ReviewWindow (QMainWindow): recording list + zoomable
                     DIC viewer + frame/recording navigation + flagging.
See main_review.py for the launcher.
"""
