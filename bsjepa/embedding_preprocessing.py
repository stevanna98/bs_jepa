"""Leakage-safe embedding preprocessing for downstream probes."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import torch

EmbeddingPreprocessingVariant = Literal[
    "raw",
    "centered",
    "standardized",
    "centered_pc_removed",
    "standardized_pc_removed",
]


@dataclass(frozen=True)
class EmbeddingPreprocessingSpec:
    """A named preprocessing configuration for downstream embedding probes."""

    variant: EmbeddingPreprocessingVariant
    pc_components: int = 0

    @property
    def metric_suffix(self) -> str:
        if self.variant == "centered_pc_removed":
            return f"centered_pc{self.pc_components}"
        if self.variant == "standardized_pc_removed":
            return f"standardized_pc{self.pc_components}"
        return self.variant


class EmbeddingPreprocessor:
    """Fit downstream embedding preprocessing on a training split, then transform any split.

    Raw embeddings contain both shared cohort structure and subject-specific deviations.
    Centering removes the cohort-average direction. PC removal removes dominant shared
    axes, which can help an ablation but can also remove biologically meaningful signal.
    Fit this object only on probe-training embeddings to avoid validation/test leakage.
    """

    def __init__(
        self,
        variant: EmbeddingPreprocessingVariant,
        *,
        pc_components: int = 0,
        standardize_epsilon: float = 1e-6,
    ) -> None:
        if variant not in {
            "raw",
            "centered",
            "standardized",
            "centered_pc_removed",
            "standardized_pc_removed",
        }:
            raise ValueError(f"Unknown embedding preprocessing variant: {variant}")
        if pc_components < 0:
            raise ValueError("pc_components must be non-negative")
        if standardize_epsilon <= 0:
            raise ValueError("standardize_epsilon must be positive")
        self.variant = variant
        self.pc_components = pc_components
        self.standardize_epsilon = standardize_epsilon
        self.mean_: torch.Tensor | None = None
        self.std_: torch.Tensor | None = None
        self.components_: torch.Tensor | None = None
        self.fitted_pc_components_: int = 0

    def fit(self, train_embeddings: torch.Tensor) -> "EmbeddingPreprocessor":
        train = self._as_2d_float(train_embeddings)
        if self.variant == "raw":
            self.mean_ = torch.zeros(train.shape[1], dtype=train.dtype)
            self.std_ = torch.ones(train.shape[1], dtype=train.dtype)
            self.components_ = train.new_empty((0, train.shape[1]))
            self.fitted_pc_components_ = 0
            return self

        self.mean_ = train.mean(dim=0)
        centered = train - self.mean_
        if self.variant.startswith("standardized"):
            self.std_ = train.std(dim=0, unbiased=False).clamp_min(
                self.standardize_epsilon
            )
            base = centered / self.std_
        else:
            self.std_ = torch.ones(train.shape[1], dtype=train.dtype)
            base = centered

        if self.variant.endswith("_pc_removed") and self.pc_components > 0:
            max_components = min(base.shape[0], base.shape[1])
            if self.pc_components > max_components:
                warnings.warn(
                    f"Requested {self.pc_components} PCs but only {max_components} "
                    "can be estimated; clamping",
                    stacklevel=2,
                )
            component_count = min(self.pc_components, max_components)
            if component_count > 0:
                _, _, vh = torch.linalg.svd(base, full_matrices=False)
                self.components_ = vh[:component_count].detach().clone()
                self.fitted_pc_components_ = component_count
            else:
                self.components_ = base.new_empty((0, base.shape[1]))
        else:
            self.components_ = base.new_empty((0, base.shape[1]))
        return self

    def transform(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.mean_ is None or self.std_ is None or self.components_ is None:
            raise RuntimeError("EmbeddingPreprocessor must be fit before transform")
        values = self._as_2d_float(embeddings)
        transformed = values if self.variant == "raw" else (values - self.mean_) / self.std_
        if self.components_.numel() > 0:
            transformed = transformed - transformed @ self.components_.T @ self.components_
        return transformed

    def fit_transform(self, train_embeddings: torch.Tensor) -> torch.Tensor:
        return self.fit(train_embeddings).transform(train_embeddings)

    @staticmethod
    def _as_2d_float(embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a two-dimensional tensor")
        return embeddings.detach().cpu().float()


def build_preprocessing_specs(
    config: dict,
) -> list[EmbeddingPreprocessingSpec]:
    """Expand preprocessing config into concrete metric variants.

    The default is raw only, preserving historical probe behavior.
    """
    variants = list(config.get("variants", ["raw"]))
    pc_components = [int(value) for value in config.get("pc_remove_components", [1])]
    specs: list[EmbeddingPreprocessingSpec] = []
    for variant in variants:
        if variant in {"centered_pc_removed", "standardized_pc_removed"}:
            for count in pc_components:
                specs.append(
                    EmbeddingPreprocessingSpec(
                        variant=variant,
                        pc_components=count,
                    )
                )
        else:
            specs.append(EmbeddingPreprocessingSpec(variant=variant))
    return specs
