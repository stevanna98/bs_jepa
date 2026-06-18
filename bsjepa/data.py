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
SubnetworkStrategy = Literal["atlas_rsn", "fixed_random", "subject_communities"]


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


def _order_gradient_rows(
    gradients: torch.Tensor,
    region_values: list[Any] | np.ndarray | torch.Tensor | None,
    num_regions: int,
) -> torch.Tensor:
    if gradients.ndim != 2 or gradients.shape[0] != num_regions:
        raise ValueError(
            "Gradient features must have shape [num_regions, gradient_dim]"
        )
    if region_values is not None:
        raw_ids = [int(value) for value in region_values]
        if set(raw_ids) == set(range(num_regions)):
            offset = 0
        elif set(raw_ids) == set(range(1, num_regions + 1)):
            offset = 1
        else:
            raise ValueError(
                "Gradient region IDs must be unique zero-based or one-based integers"
            )
        order = torch.empty(num_regions, dtype=torch.long)
        for row, region_id in enumerate(raw_ids):
            order[region_id - offset] = row
        gradients = gradients[order]
    gradients = gradients.float()
    if gradients.shape[1] < 1 or not torch.isfinite(gradients).all():
        raise ValueError("Gradient features must be finite and have at least one column")
    return gradients


def load_gradient_features(
    path: str | Path,
    num_regions: int,
    *,
    gradient_columns: list[str] | None = None,
    region_column: str | None = None,
) -> torch.Tensor:
    """Load fixed atlas gradients from CSV, PT, or NPZ in atlas-region order."""
    gradient_path = Path(path)
    region_values: list[Any] | np.ndarray | torch.Tensor | None = None
    if gradient_path.suffix == ".csv":
        with gradient_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"Gradient file is empty: {gradient_path}")
        columns = gradient_columns or [
            column for column in rows[0] if column != region_column
        ]
        if not columns or any(column not in rows[0] for column in columns):
            raise KeyError("Configured gradient columns are missing from the CSV")
        if region_column:
            if region_column not in rows[0]:
                raise KeyError(f"Region column {region_column!r} is missing from the CSV")
            region_values = [row[region_column] for row in rows]
        gradients = torch.tensor(
            [[float(row[column]) for column in columns] for row in rows],
            dtype=torch.float32,
        )
    elif gradient_path.suffix == ".pt":
        loaded = torch.load(gradient_path, map_location="cpu", weights_only=False)
        if isinstance(loaded, torch.Tensor):
            if region_column:
                raise KeyError(
                    "A tensor-only PT file cannot provide the configured region column"
                )
            gradients = loaded
        elif isinstance(loaded, dict):
            if region_column:
                if region_column not in loaded:
                    raise KeyError(
                        f"Region key {region_column!r} is missing from the PT file"
                    )
                region_values = loaded[region_column]
            if gradient_columns and all(column in loaded for column in gradient_columns):
                gradients = torch.stack(
                    [torch.as_tensor(loaded[column]) for column in gradient_columns], dim=1
                )
            else:
                value = loaded.get("gradients", loaded.get("positional_features"))
                if value is None:
                    raise KeyError("PT gradient dictionary needs a 'gradients' tensor")
                gradients = torch.as_tensor(value)
        else:
            raise TypeError("PT gradient file must contain a tensor or dictionary")
    elif gradient_path.suffix == ".npz":
        with np.load(gradient_path, allow_pickle=False) as archive:
            if region_column:
                if region_column not in archive:
                    raise KeyError(
                        f"Region array {region_column!r} is missing from the NPZ file"
                    )
                region_values = archive[region_column].copy()
            if gradient_columns and all(column in archive for column in gradient_columns):
                gradients = torch.tensor(
                    np.stack([archive[column] for column in gradient_columns], axis=1)
                )
            elif "gradients" in archive:
                gradients = torch.tensor(archive["gradients"])
            elif "positional_features" in archive:
                gradients = torch.tensor(archive["positional_features"])
            else:
                raise KeyError("NPZ gradient file needs a 'gradients' array")
    else:
        raise ValueError("Gradient file must use .csv, .pt, or .npz")
    return _order_gradient_rows(gradients, region_values, num_regions)


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


def fixed_random_subnetworks(
    num_regions: int, num_subnetworks: int, seed: int
) -> torch.Tensor:
    """Create a balanced deterministic partition shared by all subjects."""
    if not 2 <= num_subnetworks <= num_regions:
        raise ValueError("num_subnetworks must be in [2, num_regions]")
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(num_regions, generator=generator)
    assignments = torch.empty(num_regions, dtype=torch.long)
    assignments[permutation] = torch.arange(num_regions) % num_subnetworks
    return assignments


def _repair_empty_clusters(
    assignments: torch.Tensor, distances: torch.Tensor, num_subnetworks: int
) -> torch.Tensor:
    """Move poorly represented points so every cluster has at least one region."""
    counts = torch.bincount(assignments, minlength=num_subnetworks)
    for empty_cluster in (counts == 0).nonzero(as_tuple=True)[0]:
        assigned_distances = distances[
            torch.arange(len(assignments)), assignments
        ].clone()
        assigned_distances[counts[assignments] <= 1] = -1
        donor = int(assigned_distances.argmax())
        old_cluster = int(assignments[donor])
        assignments[donor] = empty_cluster
        counts[old_cluster] -= 1
        counts[empty_cluster] += 1
    return assignments


def fc_community_subnetworks(
    fc: torch.Tensor,
    num_subnetworks: int,
    *,
    method: str = "fc_kmeans",
    seed: int = 42,
    max_iterations: int = 25,
) -> torch.Tensor:
    """Cluster normalized absolute FC profiles into exactly k subject communities."""
    if method not in {"fc_kmeans", "kmeans"}:
        raise ValueError(
            f"Unsupported community method {method!r}; use 'fc_kmeans'"
        )
    num_regions = fc.shape[0]
    if fc.ndim != 2 or fc.shape[1] != num_regions:
        raise ValueError("FC matrix must be square")
    if not 2 <= num_subnetworks <= num_regions:
        raise ValueError("num_subnetworks must be in [2, num_regions]")
    profiles = fc.float().abs().clone()
    profiles.fill_diagonal_(0)
    features = torch.nn.functional.normalize(profiles, dim=1)
    generator = torch.Generator().manual_seed(seed)
    first_center = int(torch.randint(num_regions, (1,), generator=generator))
    center_indices = [first_center]
    minimum_distances = (features - features[first_center]).square().sum(1)
    for _ in range(1, num_subnetworks):
        minimum_distances[center_indices] = -1
        next_center = int(minimum_distances.argmax())
        if minimum_distances[next_center] <= 0:
            next_center = next(
                index for index in range(num_regions) if index not in center_indices
            )
        center_indices.append(next_center)
        distances = (features - features[next_center]).square().sum(1)
        minimum_distances = torch.minimum(minimum_distances, distances)
    centers = features[center_indices].clone()
    assignments = torch.full((num_regions,), -1, dtype=torch.long)
    for _ in range(max_iterations):
        distances = torch.cdist(features, centers).square()
        updated = _repair_empty_clusters(
            distances.argmin(1), distances, num_subnetworks
        )
        if torch.equal(updated, assignments):
            break
        assignments = updated
        centers = torch.stack(
            [features[assignments == group].mean(0) for group in range(num_subnetworks)]
        )
    distances = torch.cdist(features, centers).square()
    return _repair_empty_clusters(
        distances.argmin(1), distances, num_subnetworks
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
        subnetwork_strategy: SubnetworkStrategy = "atlas_rsn",
        num_subnetworks: int | None = None,
        subnetwork_seed: int = 42,
        community_method: str = "fc_kmeans",
        gradient_key: str | None = None,
    ) -> None:
        self.atlas = atlas
        self.node_features = node_features
        self.bold_key = bold_key
        self.fc_key = fc_key
        self.transpose_bold = transpose_bold
        self.graph_strategy = graph_strategy
        self.top_k = top_k
        self.threshold = threshold
        self.subnetwork_strategy = subnetwork_strategy
        self.num_subnetworks = (
            atlas.num_rsns if num_subnetworks is None else num_subnetworks
        )
        self.subnetwork_seed = subnetwork_seed
        self.community_method = community_method
        self.gradient_key = gradient_key
        if subnetwork_strategy not in (
            "atlas_rsn",
            "fixed_random",
            "subject_communities",
        ):
            raise ValueError(f"Unknown subnetwork strategy: {subnetwork_strategy}")
        if subnetwork_strategy == "atlas_rsn" and self.num_subnetworks != atlas.num_rsns:
            raise ValueError(
                "masking.num_subnetworks must match the atlas RSN count for atlas_rsn"
            )
        if not 2 <= self.num_subnetworks <= atlas.num_regions:
            raise ValueError("num_subnetworks must be in [2, num_regions]")
        self._fixed_subnetwork_ids = (
            fixed_random_subnetworks(
                atlas.num_regions, self.num_subnetworks, subnetwork_seed
            )
            if subnetwork_strategy == "fixed_random"
            else None
        )
        self._community_cache: dict[int, torch.Tensor] = {}
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

    def get_subject_metadata(self, index: int) -> dict[str, Any]:
        """Return scalar subject metadata without imaging arrays."""
        record = self._get_record(index)
        imaging_keys = {
            self.bold_key,
            self.fc_key,
            "time_series",
            "X",
            "fc_matrix",
            self.gradient_key,
        }
        return {key: value for key, value in record.items() if key not in imaging_keys}

    def _get_record(self, index: int) -> dict[Any, Any]:
        key = self._subjects[index]
        record = _load_file(key) if self._records is None else self._records[key]
        if not isinstance(record, dict):
            raise TypeError(f"Subject {key!s} is not a dictionary")
        return record

    def __getitem__(self, index: int) -> Data:
        key = self._subjects[index]
        record = self._get_record(index)
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
        if self.subnetwork_strategy == "atlas_rsn":
            graph.subnetwork_ids = graph.rsn_ids.clone()
        elif self.subnetwork_strategy == "fixed_random":
            graph.subnetwork_ids = self._fixed_subnetwork_ids.clone()
        else:
            if index not in self._community_cache:
                self._community_cache[index] = fc_community_subnetworks(
                    fc,
                    self.num_subnetworks,
                    method=self.community_method,
                    seed=self.subnetwork_seed,
                )
            graph.subnetwork_ids = self._community_cache[index].clone()
        if self.gradient_key is not None:
            if self.gradient_key not in record:
                raise KeyError(
                    f"Subject {key!s} has no positional gradients at "
                    f"record key {self.gradient_key!r}"
                )
            positional_features = torch.as_tensor(
                record[self.gradient_key], dtype=torch.float32
            )
            if (
                positional_features.ndim != 2
                or positional_features.shape[0] != self.atlas.num_regions
                or positional_features.shape[1] < 1
            ):
                raise ValueError(
                    f"Subject {key!s} positional gradients must have shape "
                    "[num_regions, gradient_dim]"
                )
            if not torch.isfinite(positional_features).all():
                raise ValueError(f"Subject {key!s} positional gradients must be finite")
            graph.positional_features = positional_features
        graph.region_ids = torch.arange(self.atlas.num_regions)
        return graph


class SyntheticBrainDataset(Dataset[Data]):
    """Deterministic synthetic graphs for smoke tests and development."""

    def __init__(
        self,
        atlas: Atlas,
        num_subjects: int,
        feature_dim: int,
        *,
        top_k: int,
        seed: int,
        subnetwork_strategy: SubnetworkStrategy = "atlas_rsn",
        num_subnetworks: int | None = None,
        subnetwork_seed: int = 42,
        community_method: str = "fc_kmeans",
    ) -> None:
        self.atlas = atlas
        self.num_subjects = num_subjects
        self.feature_dim = feature_dim
        self.top_k = top_k
        self.seed = seed
        self.subnetwork_strategy = subnetwork_strategy
        self.num_subnetworks = (
            atlas.num_rsns if num_subnetworks is None else num_subnetworks
        )
        self.subnetwork_seed = subnetwork_seed
        self.community_method = community_method
        if subnetwork_strategy not in (
            "atlas_rsn",
            "fixed_random",
            "subject_communities",
        ):
            raise ValueError(f"Unknown subnetwork strategy: {subnetwork_strategy}")
        if subnetwork_strategy == "atlas_rsn" and self.num_subnetworks != atlas.num_rsns:
            raise ValueError(
                "masking.num_subnetworks must match data.num_rsns for synthetic atlas_rsn"
            )
        if not 2 <= self.num_subnetworks <= atlas.num_regions:
            raise ValueError("num_subnetworks must be in [2, num_regions]")
        self._fixed_subnetwork_ids = (
            fixed_random_subnetworks(
                atlas.num_regions, self.num_subnetworks, subnetwork_seed
            )
            if subnetwork_strategy == "fixed_random"
            else None
        )
        self._community_cache: dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return self.num_subjects

    @property
    def subject_ids(self) -> list[str]:
        return [f"synthetic_{index}" for index in range(self.num_subjects)]

    def get_subject_metadata(self, index: int) -> dict[str, str]:
        """Provide balanced synthetic labels for probe smoke tests."""
        return {"gender": "F" if index % 2 == 0 else "M"}

    def __getitem__(self, index: int) -> Data:
        generator = torch.Generator().manual_seed(self.seed + index)
        regions = self.atlas.num_regions
        fc = pearson_correlation(torch.randn(regions, 100, generator=generator))
        graph = build_graph(
            fc,
            strategy="top_k",
            top_k=self.top_k,
        )
        graph.x = torch.randn(regions, self.feature_dim, generator=generator)
        graph.rsn_ids = self.atlas.rsn_ids.clone()
        if self.subnetwork_strategy == "atlas_rsn":
            graph.subnetwork_ids = graph.rsn_ids.clone()
        elif self.subnetwork_strategy == "fixed_random":
            graph.subnetwork_ids = self._fixed_subnetwork_ids.clone()
        else:
            if index not in self._community_cache:
                self._community_cache[index] = fc_community_subnetworks(
                    fc,
                    self.num_subnetworks,
                    method=self.community_method,
                    seed=self.subnetwork_seed,
                )
            graph.subnetwork_ids = self._community_cache[index].clone()
        graph.region_ids = torch.arange(regions)
        return graph


class SubjectSubset(Dataset[Data]):
    """Index subset that preserves subject IDs and metadata access."""

    def __init__(self, dataset: Dataset[Data], indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Data:
        return self.dataset[self.indices[index]]

    @property
    def subject_ids(self) -> list[Any]:
        source_ids = getattr(self.dataset, "subject_ids", None)
        if source_ids is None:
            raise TypeError("Underlying dataset does not expose subject_ids")
        return [source_ids[index] for index in self.indices]

    def get_subject_metadata(self, index: int) -> dict[str, Any]:
        metadata_getter = getattr(self.dataset, "get_subject_metadata", None)
        if metadata_getter is None:
            raise TypeError("Underlying dataset does not expose subject metadata")
        return metadata_getter(self.indices[index])
