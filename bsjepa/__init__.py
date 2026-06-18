"""Minimal BS-JEPA pretraining package."""

from .data import Atlas, BrainGraphDataset, SyntheticBrainDataset, load_atlas
from .masking import SubnetworkMaskCollator
from .model import BSJEPA, build_bsjepa
from .training import pretrain

__all__ = [
    "Atlas",
    "BSJEPA",
    "BrainGraphDataset",
    "SubnetworkMaskCollator",
    "SyntheticBrainDataset",
    "build_bsjepa",
    "load_atlas",
    "pretrain",
]
