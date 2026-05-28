"""Quiet logging for WISH entrypoints.

Call `silence_third_party()` at the *very top* of an entrypoint (before any
torch / detectron2 imports if possible) to suppress Python warnings; call
`quiet_loggers()` after `default_setup` to mute noisy third-party loggers
while keeping the detectron2 training logger at INFO.
"""
from __future__ import annotations

import logging
import os
import warnings


_NOISY_LOGGERS = (
    "fvcore",
    "iopath",
    "PIL",
    "matplotlib",
    "urllib3",
    "h5py",
    "timm",
    "torch.distributed.distributed_c10d",
    "torch.nn.parallel.distributed",
)


def silence_third_party() -> None:
    """Suppress Python `warnings` and noisy backend env chatter.

    Safe to call multiple times. Idempotent.
    """
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    # Detectron2 / fvcore sometimes look at this:
    os.environ.setdefault("TQDM_DISABLE", "0")
    warnings.filterwarnings("ignore")


def quiet_loggers(level: int = logging.ERROR) -> None:
    """Raise log level on third-party libs known to spam INFO/WARN lines."""
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(level)
