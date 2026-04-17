"""Core analysis modules — pure functions, no GUI dependencies.

Submodules:
    io           — load video files and metadata
    contour      — contour extraction and conversion
    detection    — cellpose, optical flow, fused detection
    refinement   — edge snap, Fourier smoothing
    auto_params  — automatic parameter selection
    tracking     — centroid tracking, speed, MSD, persistence
    morphology   — area, perimeter, circularity, etc.
    edge_dynamics — polar boundary, edge velocity, kymograph
    evaluation   — quality metrics (boundary confidence, IoU, area stability)
    pipeline     — full analysis orchestration
"""
