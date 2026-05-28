from hetero.models.mask2former_r50 import (
    build_mask2former_r50,
    build_mask2former_r50_cfg,
    build_mask2former_r50_train_loader,
    build_mask2former_r50_training_bundle,
)

# Importing WISH modules registers META_ARCH and TRANSFORMER_DECODER entries
# with detectron2 so that cfg-driven build_model finds them.
from hetero.models import wish_transformer_decoder  # noqa: F401  (registers decoder)
from hetero.models import wish_meta_arch  # noqa: F401              (registers meta-arch)

__all__ = [
    "build_mask2former_r50",
    "build_mask2former_r50_cfg",
    "build_mask2former_r50_train_loader",
    "build_mask2former_r50_training_bundle",
]
