"""Register Pascal VOC 2012 with heterogeneous weak-label assignment.

Similar to COCO registration, wraps standard VOC datasets and adds per-image
`weak_label_type` in {"tag", "point", "box"} drawn from cfg ratios + cfg seed.
Weak labels are derived on-the-fly in the mapper, not stored on disk.
"""
from __future__ import annotations

import copy
import hashlib
from typing import Sequence

from detectron2.data import DatasetCatalog, MetadataCatalog


def _assign_label_type(image_id: int, seed: int, ratios: Sequence[float]) -> str:
    """Deterministic per-image: hash(image_id, seed) -> one of {tag, point, box}."""
    types = ("tag", "point", "box")
    h = hashlib.md5(f"{seed}:{image_id}".encode()).digest()
    u = int.from_bytes(h[:4], "big") / 2 ** 32
    cum = 0.0
    for t, r in zip(types, ratios):
        cum += r
        if u < cum:
            return t
    return types[-1]


def register_voc_hetero(
    name: str = "voc_2012_train_hetero",
    base_name: str = "voc_2012_train",
    seed: int = 0,
    ratios: Sequence[float] = (0.34, 0.33, 0.33),
) -> None:
    """Register a thin wrapper around an existing VOC dataset.

    Each record from the base dataset is annotated with `weak_label_type` via a
    deterministic hash of `(seed, image_id)`. Metadata (thing_classes etc.) is
    copied from the base dataset so eval / vis still work.
    """
    if name in DatasetCatalog.list():
        return

    ratios_tuple = tuple(float(r) for r in ratios)

    def _loader():
        records = DatasetCatalog.get(base_name)
        out = []
        for r in records:
            r2 = copy.copy(r)
            iid = r2.get("image_id", None)
            if iid is None:
                iid = hash(r2.get("file_name", ""))
            r2["weak_label_type"] = _assign_label_type(int(iid), seed, ratios_tuple)
            out.append(r2)
        return out

    DatasetCatalog.register(name, _loader)

    base_meta = MetadataCatalog.get(base_name)
    new_meta = MetadataCatalog.get(name)
    for key, value in base_meta.as_dict().items():
        if key in ("name",):
            continue
        try:
            setattr(new_meta, key, value)
        except AttributeError:
            pass
    new_meta.weak_label_seed = seed
    new_meta.weak_label_ratios = list(ratios_tuple)
