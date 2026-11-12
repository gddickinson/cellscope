"""README's cell-state section describes the rule the code applies (core/cell_state.py).

It described a three-state balled / attached / transitional split with an "attached" gate
(circ <= 0.55 or solidity <= 0.85) that no longer exists: the code classifies rounded /
spread / unknown, by area_um2 <= rounded_area_um2 and eccentricity <= rounded_eccentricity
when the pixel size is known, else by the circularity / solidity fallback.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.cell_state import DEFAULT_THRESHOLDS  # noqa: E402


def _section():
    text = (ROOT / "README.md").read_text()
    start = text.index("## Cell-state classification")
    return text[start:text.index("\n## ", start + 5)]


def test_readme_state_section_matches_code():
    s = _section()
    assert "transitional" not in s and "circ≤0.55" not in s
    assert "rounded" in s and "spread" in s
    th = DEFAULT_THRESHOLDS
    assert f"{th['rounded_area_um2']:g} µm²" in s
    assert f"{th['rounded_eccentricity']:g}" in s
    assert f"{th['rounded_circ']:g}" in s and f"{th['rounded_solid']:g}" in s


def test_readme_key_module_line():
    assert "balled/attached classification" not in (ROOT / "README.md").read_text()
