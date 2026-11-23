"""Target-part label vocabulary (used only by the use_part_label ablation).

The 15 "part" values of the HOT3D-Grasp clips. Index 0 means "unknown" and is the slot
used by condition dropout. Dependency-free so the model and dataset can both import it.
"""

TEXTHOM_PART_VOCAB = (
    "body", "bottom", "bridge", "cap", "center", "edge", "frame", "handle",
    "lid", "long_edge", "rim", "roof", "short_edge", "tail", "top",
)
TEXTHOM_PART_UNKNOWN = 0
_PART_TO_ID = {name: i + 1 for i, name in enumerate(TEXTHOM_PART_VOCAB)}


def texthom_part_name_to_id(name):
    """Part name -> 1..15; unknown names map to 0."""
    return _PART_TO_ID.get(str(name).strip().lower(), TEXTHOM_PART_UNKNOWN)
