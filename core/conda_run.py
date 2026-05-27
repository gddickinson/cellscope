"""Helpers for invoking sibling conda envs via subprocess.

Single source of truth for "how do we spawn cellpose4 from cellpose
(and vice versa)" so a Windows-vs-POSIX quirk only has to be fixed
once.
"""
import os


def conda_exe() -> str:
    """Return the conda executable to pass to subprocess.

    Uses CONDA_EXE (which `conda activate` exports — always the full
    path to the actual conda.exe / conda binary) if set, otherwise
    falls back to the bare name "conda".

    Why this matters: on Windows, ``subprocess.run(["conda", ...])``
    without ``shell=True`` searches PATH only for ``conda.exe``.
    Modern Miniconda installs often ship ``conda.bat`` shims rather
    than ``conda.exe`` on PATH, so the bare-name lookup fails with
    ``WinError 2: The system cannot find the file specified``.
    ``CONDA_EXE`` always points at the real executable so the call
    works on Windows, macOS, and Linux uniformly.
    """
    return os.environ.get("CONDA_EXE") or "conda"
