"""Atlas loading, connectivity construction, and pretraining datasets."""

from __future__ import annotations

import csv
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

GraphStrategy = Literal["dense", "top_k", "absolute_threshold"]
NodeFeatures = Literal["bold", "fc_row", "ones"]


@dataclass(frozen=True)
class Atlas:
    """Zero-indexed region-to-RSN assignment."""

    rsn_ids: torch.Tensor
    rsn_names: list[str]

    @property
    def num_regions(self) -> int:
        return self.rsn_ids.numel()

    @property
    def num_rsns(self) -> int:
        return len(self.rsn_names)


def load_atlas(path: str | Path) -> Atlas:
    """Load columns ``rsn_id`` and ``rsn_name`` from an atlas CSV."""
    raw_ids: list[int] = []
    names: dict[int, str] = {}
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            rsn_id = int(row["rsn_id"])
            raw_ids.append(rsn_id)
            names[rsn_id] = row["rsn_name"]
    if not raw_ids:
        raise ValueError(f"Atlas is empty: {path}")
    ordered_ids = sorted(names)
    to_zero = {raw_id: index for index, raw_id in enumerate(ordered_ids)}
    return Atlas(
        torch.tensor([to_zero[raw_id] for raw_id in raw_ids], dtype=torch.long),
        [names[raw_id] for raw_id in ordered_ids],
    )


def synthetic_atlas(num_regions: int, num_rsns: int) -> Atlas:
    if num_rsns < 2 or num_regions < num_rsns:
        raise ValueError("Synthetic data requires num_regions >= num_rsns >= 2")
    return Atlas(
        torch.arange(num_regions) % num_rsns,
        [f"rsn_{index}" for index in range(num_rsns)],
    )


def pearson_correlation(time_series: torch.Tensor) -> torch.Tensor:
    """Compute an FC matrix from region-by-time BOLD data."""
    centered = time_series - time_series.mean(dim=1, keepdim=True)
    scaled = centered / centered.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-8)
    return (scaled @ scaled.T) / time_series.shape[1]


def build_graph(
    fc: torch.Tensor,
    *,
    strategy: GraphStrategy = "top_k",
    top_k: int = 10,
    threshold: float = 0.2,
) -> Data:
    """Convert a square FC matrix to a weighted PyG graph."""
    if fc.ndim != 2 or fc.shape[0] != fc.shape[1]:
        raise ValueError("FC matrix must be square")
    adjacency = fc.float().clone()
    adjacency.fill_diagonal_(0)
    if strategy == "top_k":
        node_count = adjacency.shape[0]
        indices = adjacency.abs().topk(min(top_k, node_count - 1), dim=1).indices
        selected = torch.zeros_like(adjacency)
        selected_mask = torch.zeros_like(adjacency, dtype=torch.bool)
        selected.scatter_(1, indices, adjacency.gather(1, indices))
        selected_mask.scatter_(1, indices, True)
        union_mask = selected_mask | selected_mask.T
        selected_count = selected_mask.float() + selected_mask.T.float()
        adjacency = (selected + selected.T) / selected_count.clamp_min(1)
        adjacency *= union_mask
    elif strategy == "absolute_threshold":
        adjacency *= adjacency.abs() >= threshold
    elif strategy != "dense":
        raise ValueError(f"Unknown graph strategy: {strategy}")
    source, target = adjacency.nonzero(as_tuple=True)
    return Data(
        edge_index=torch.stack([source, target]),
        edge_attr=adjacency[source, target].unsqueeze(-1),
        num_nodes=adjacency.shape[0],
    )


def _zscore(time_series: torch.Tensor) -> torch.Tensor:
    return (time_series - time_series.mean(1, keepdim=True)) / time_series.std(
        1, keepdim=True, unbiased=False
    ).clamp_min(1e-8)


def _validate_bold_shape(bold: torch.Tensor, expected_regions: int, subject: Any) -> None:
    if bold.ndim != 2:
        raise ValueError(f"Subject {subject!s} BOLD data must be region-by-time")
    if bold.shape[0] != expected_regions:
        raise ValueError(
            f"Subject {subject!s} BOLD data has {bold.shape[0]} regions; "
            f"atlas has {expected_regions}"
        )


def _load_file(path: Path) -> Any:
    if path.suffix == ".pkl":
        with path.open("rb") as handle:
            return pickle.load(handle)
    if path.suffix == ".pt":
        return torch.load(path, map_location="cpu", weights_only=False)
    if path.suffix == ".npz":
        archive = np.load(path, allow_pickle=True)
        return {key: archive[key] for key in archive.files}
    raise ValueError(f"Unsupported data file: {path}")


class BrainGraphDataset(Dataset[Data]):
    """Load subject files or one subject dictionary into pretraining graphs."""

    def __init__(
        self,
        source: str | Path,
        atlas: Atlas,
        *,
        node_features: NodeFeatures = "bold",
        bold_key: str = "BOLD",
        fc_key: str = "FC",
        transpose_bold: bool = False,
        graph_strategy: GraphStrategy = "top_k",
        top_k: int = 10,
        threshold: float = 0.2,
    ) -> None:
        self.atlas = atlas
        self.node_features = node_features
        self.bold_key = bold_key
        self.fc_key = fc_key
        self.transpose_bold = transpose_bold
        self.graph_strategy = graph_strategy
        self.top_k = top_k
        self.threshold = threshold
        source_path = Path(source)
        if source_path.is_dir():
            self._subjects = [
                path for path in sorted(source_path.iterdir()) if path.suffix in (".pt", ".npz")
            ]
            self._records: dict[Any, Any] | None = None
        else:
            loaded = _load_file(source_path)
            if not isinstance(loaded, dict):
                raise TypeError("A single data file must contain a subject dictionary")
            self._records = loaded
            self._subjects = list(loaded)
        if not self._subjects:
            raise ValueError(f"No subjects found in {source_path}")

    def __len__(self) -> int:
        return len(self._subjects)

    @property
    def subject_ids(self) -> list[Any]:
        """Return subject identifiers in dataset-index order."""
        return list(self._subjects)

    def __getitem__(self, index: int) -> Data:
        key = self._subjects[index]
        record = _load_file(key) if self._records is None else self._records[key]
        if not isinstance(record, dict):
            raise TypeError(f"Subject {key!s} is not a dictionary")
        bold_value = record.get(self.bold_key, record.get("time_series", record.get("X")))
        fc_value = record.get(self.fc_key, record.get("fc_matrix"))
        bold = None if bold_value is None else torch.as_tensor(bold_value, dtype=torch.float32)
        if bold is not None and self.transpose_bold:
            bold = bold.T
        if bold is not None:
            _validate_bold_shape(bold, self.atlas.num_regions, key)
        if fc_value is None:
            if bold is None:
                raise KeyError(f"Subject {key!s} contains neither BOLD nor FC data")
            fc = pearson_correlation(_zscore(bold))
        else:
            fc = torch.as_tensor(fc_value, dtype=torch.float32)
        if fc.shape[0] != self.atlas.num_regions:
            raise ValueError(
                f"Subject {key!s} has {fc.shape[0]} regions; atlas has {self.atlas.num_regions}"
            )

        graph = build_graph(
            fc, strategy=self.graph_strategy, top_k=self.top_k, threshold=self.threshold
        )
        if self.node_features == "bold":
            if bold is None:
                raise KeyError(f"Subject {key!s} has no BOLD data")
            graph.x = _zscore(bold)
        elif self.node_features == "fc_row":
            graph.x = fc
        elif self.node_features == "ones":
            graph.x = torch.ones(fc.shape[0], 1)
        else:
            raise ValueError(f"Unknown node feature type: {self.node_features}")
        graph.rsn_ids = self.atlas.rsn_ids.clone()
        graph.region_ids = torch.arange(self.atlas.num_regions)
        return graph


class SyntheticBrainDataset(Dataset[Data]):
    """Deterministic synthetic graphs for smoke tests and development."""

    def __init__(
        self, atlas: Atlas, num_subjects: int, feature_dim: int, *, top_k: int, seed: int
    ) -> None:
        self.atlas = atlas
        self.num_subjects = num_subjects
        self.feature_dim = feature_dim
        self.top_k = top_k
        self.seed = seed

    def __len__(self) -> int:
        return self.num_subjects

    @property
    def subject_ids(self) -> list[str]:
        return [f"synthetic_{index}" for index in range(self.num_subjects)]

    def __getitem__(self, index: int) -> Data:
        generator = torch.Generator().manual_seed(self.seed + index)
        regions = self.atlas.num_regions
        graph = build_graph(
            pearson_correlation(torch.randn(regions, 100, generator=generator)),
            strategy="top_k",
            top_k=self.top_k,
        )
        graph.x = torch.randn(regions, self.feature_dim, generator=generator)
        graph.rsn_ids = self.atlas.rsn_ids.clone()
        graph.region_ids = torch.arange(regions)
        return graph
