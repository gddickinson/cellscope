"""SAM sub-panel widgets for the RefinementPanel.

Factored out of refinement_panel.py to keep that file under 500 lines.
"""
from PyQt5.QtWidgets import (
    QGroupBox, QFormLayout, QComboBox, QCheckBox, QLabel,
)

SAM_LABELS = {
    "vit_b": "vit_b (~360 MB, ships with project)",
    "vit_l": "vit_l (~1.2 GB, download required)",
    "vit_h": "vit_h (~2.5 GB, download required)",
}


def build_sam_group():
    """Create the SAM options QGroupBox and return (box, widgets_dict).

    widgets_dict keys: 'sam_model_type', 'sam_use_mask_prompt'.
    """
    from core.sam_refine import list_available_sam_models
    avail = list_available_sam_models() or ["vit_b"]

    box = QGroupBox("SAM (Segment Anything Model, mode=sam)")
    form = QFormLayout()

    model_type = QComboBox()
    for mt in avail:
        model_type.addItem(SAM_LABELS.get(mt, mt), userData=mt)
    model_type.setToolTip(
        "SAM model size. Larger = more accurate, slower, and needs a "
        "downloaded checkpoint. vit_b is bundled at "
        "data/models/sam/sam_vit_b_01ec64.pth."
    )
    form.addRow("Model type:", model_type)

    use_mask_prompt = QCheckBox(
        "Use seed mask as SAM mask prompt (recommended)"
    )
    use_mask_prompt.setToolTip(
        "Pass the cellpose mask (downsampled to 256x256 logit map) "
        "as an additional prompt alongside the centroid point and bbox."
    )
    form.addRow(use_mask_prompt)

    note = QLabel(
        "<i>SAM typically produces smooth blob-like contours on DIC "
        "keratinocytes (misses filopodia). Useful as a comparison "
        "baseline or on non-keratinocyte data.</i>"
    )
    note.setWordWrap(True)
    form.addRow(note)

    box.setLayout(form)
    return box, {
        "sam_model_type": model_type,
        "sam_use_mask_prompt": use_mask_prompt,
    }
