"""Minimal BS-JEPA pretraining package."""

from .data import Atlas, BrainGraphDataset, SyntheticBrainDataset, load_atlas
from .evaluation import LabeledGraphDataset, evaluate_pmat, split_pmat_holdout
from .masking import SubnetworkMaskCollator
from .model import BSJEPA, build_bsjepa
from .training import pretrain

__all__ = [
    "Atlas",
    "BSJEPA",
    "BrainGraphDataset",
    "LabeledGraphDataset",
    "SubnetworkMaskCollator",
    "SyntheticBrainDataset",
    "build_bsjepa",
    "load_atlas",
    "pretrain",
    "evaluate_pmat",
    "split_pmat_holdout",
]
