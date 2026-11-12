"""The auto-downsample and mirror-pad cut-offs quoted in the docs match the code.

README's pipeline diagram said "<900 px: no change, 900-1500: 2x" and several docs said
mirror padding switches on at a detection min-dim of 1024 px; the code uses
DOWNSAMPLE_SMALL_PX = 1100 and MIRROR_PAD_MIN_DIM_PX = 1000.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.pipeline_defaults import (  # noqa: E402
    DOWNSAMPLE_LARGE_PX, DOWNSAMPLE_SMALL_PX, MIRROR_PAD_MIN_DIM_PX, resolve_downsample,
    should_use_mirror_pad)

DOCS = ["README.md", "docs/user_manual.md", "docs/pipeline_description.md",
        "docs/recording_recommendations.md"]


def test_readme_downsample_cutoffs():
    text = (ROOT / "README.md").read_text()
    assert f"<{DOWNSAMPLE_SMALL_PX} px" in text
    assert f"{DOWNSAMPLE_SMALL_PX}-{DOWNSAMPLE_LARGE_PX}" in text
    assert "<900 px" not in text


def test_docs_mirror_pad_cutoff():
    for doc in DOCS:
        text = (ROOT / doc).read_text()
        assert "min-dim ≥ 1024" not in text, doc
    assert f"min-dim ≥ {MIRROR_PAD_MIN_DIM_PX}" in (ROOT / "README.md").read_text()


def test_code_cutoffs_behave_as_documented():
    assert resolve_downsample("auto", (DOWNSAMPLE_SMALL_PX - 1, 500))[0] == 1
    assert resolve_downsample("auto", (DOWNSAMPLE_SMALL_PX, 500))[0] == 2
    assert should_use_mirror_pad("auto", (MIRROR_PAD_MIN_DIM_PX, 2000))
    assert not should_use_mirror_pad("auto", (MIRROR_PAD_MIN_DIM_PX - 1, 2000))
