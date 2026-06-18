"""Subnetwork masking and induced-subgraph extraction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_geometric.data import Batch, Data


@dataclass
class MaskOutput:
    """Context and independently predicted target masks for a graph batch."""

    context_node_masks: list[torch.Tensor]
    target_subnetwork_masks: list[list[torch.Tensor]]
    target_subnetwork_ids: torch.Tensor

    @property
    def target_rsn_ids(self) -> torch.Tensor:
        """Backward-compatible alias for atlas-RSN callers."""
        return self.target_subnetwork_ids

    def to(self, device: torch.device) -> "MaskOutput":
        self.context_node_masks = [mask.to(device) for mask in self.context_node_masks]
        self.target_subnetwork_masks = [
            [mask.to(device) for mask in subject] for subject in self.target_subnetwork_masks
        ]
        self.target_subnetwork_ids = self.target_subnetwork_ids.to(device)
        return self


class SubnetworkMaskCollator:
    """Sample target subnetworks and collate subject graphs for BS-JEPA."""

    def __init__(self, num_subnetworks: int, num_targets: int) -> None:
        if not 1 <= num_targets < num_subnetworks:
            raise ValueError("num_targets must be in [1, num_subnetworks - 1]")
        self.num_subnetworks = num_subnetworks
        self.num_rsns = num_subnetworks  # Backward-compatible attribute alias.
        self.num_targets = num_targets

    def __call__(self, graphs: list[Data]) -> tuple[Batch, MaskOutput]:
        if not graphs:
            raise ValueError("Cannot collate an empty graph list")
        targets = torch.stack(
            [torch.randperm(self.num_subnetworks)[: self.num_targets] for _ in graphs]
        )
        contexts: list[torch.Tensor] = []
        target_groups: list[list[torch.Tensor]] = []
        for graph, subject_targets in zip(graphs, targets, strict=True):
            subnetwork_ids = getattr(graph, "subnetwork_ids", None)
            if subnetwork_ids is None:
                subnetwork_ids = graph.rsn_ids
            if subnetwork_ids.shape != (graph.num_nodes,):
                raise ValueError("Each graph needs one subnetwork ID per node")
            unique_ids = subnetwork_ids.unique()
            expected_ids = torch.arange(
                self.num_subnetworks, device=subnetwork_ids.device
            )
            if not torch.equal(unique_ids.sort().values, expected_ids):
                raise ValueError(
                    "Graph subnetwork IDs must contain every integer in "
                    "[0, num_subnetworks - 1]"
                )
            groups = [subnetwork_ids == group for group in subject_targets]
            target_union = torch.stack(groups).any(dim=0)
            contexts.append(~target_union)
            target_groups.append(groups)
        return Batch.from_data_list(graphs), MaskOutput(contexts, target_groups, targets)


def extract_subgraph(data: Data, node_mask: torch.Tensor) -> Data:
    """Return the node-induced subgraph while preserving atlas-region IDs."""
    region_ids = node_mask.nonzero(as_tuple=True)[0]
    old_to_new = torch.full(
        (data.num_nodes,), -1, dtype=torch.long, device=node_mask.device
    )
    old_to_new[region_ids] = torch.arange(region_ids.numel(), device=node_mask.device)
    src, dst = data.edge_index
    keep_edges = node_mask[src] & node_mask[dst]
    edge_attr = data.edge_attr[keep_edges] if data.edge_attr is not None else None
    subgraph = Data(
        x=data.x[region_ids],
        edge_index=old_to_new[data.edge_index[:, keep_edges]],
        edge_attr=edge_attr,
        region_ids=region_ids,
        num_nodes=region_ids.numel(),
    )
    if hasattr(data, "positional_features"):
        subgraph.positional_features = data.positional_features[region_ids]
    return subgraph
