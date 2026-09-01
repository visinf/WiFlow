"""WiFlow: Estimating Optical Flow using WiFi Channel State Information.

Public API
----------
Models:
    WiFlowRNN       — WiFlowSimple architecture
    WiFlowMask      — mask/motion-detection block (used in WiFlowRoI)
    WiFlowCombined  — WiFlowCombo architecture (recommended)
    RNNArgs         — shared model configuration (pydantic BaseModel)

Preprocessors:
    CSIPreprocessor             — base class / Raw preprocessor
    CSIQuotientPreprocessor     — Quotient preprocessor (recommended)

Dataset:
    MMFI_DatasetPairwise        — pairwise CSI + optical-flow dataset loader
"""

from .csi_preprocessor import (
    CSIPreprocessor as CSIPreprocessor,
)
from .csi_preprocessor import (
    CSIQuotientPreprocessor as CSIQuotientPreprocessor,
)
from .dataset import MMFI_DatasetPairwise as MMFI_DatasetPairwise
from .WIFlow_combined import WiFlowCombined as WiFlowCombined
from .WIFlow_mask import WiFlowMask as WiFlowMask
from .WIFlow_rnn import RNNArgs as RNNArgs
from .WIFlow_rnn import WIFlow as WIFlow
from .WIFlow_rnn import WiFlowRNN as WiFlowRNN

__all__ = [
    "WiFlowRNN",
    "WIFlow",
    "WiFlowMask",
    "WiFlowCombined",
    "RNNArgs",
    "CSIPreprocessor",
    "CSIQuotientPreprocessor",
    "MMFI_DatasetPairwise",
]
