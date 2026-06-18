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
    target_rsn_ids: torch.Tensor

    def to(self, device: torch.device) -> "MaskOutput":
        self.context_node_masks = [mask.to(device) for mask in self.context_node_masks]
        self.target_subnetwork_masks = [
            [mask.to(device) for mask in subject] for subject in self.target_subnetwork_masks
        ]
        self.target_rsn_ids = self.target_rsn_ids.to(device)
        return self


class SubnetworkMaskCollator:
    """Sample target RSNs and collate subject graphs for BS-JEPA."""

    def __init__(self, num_rsns: int, num_targets: int) -> None:
        if not 1 <= num_targets < num_rsns:
            raise ValueError("num_targets must be in [1, num_rsns - 1]")
        self.num_rsns = num_rsns
        self.num_targets = num_targets

    def __call__(self, graphs: list[Data]) -> tuple[Batch, MaskOutput]:
        if not graphs:
            raise ValueError("Cannot collate an empty graph list")
        targets = torch.stack(
            [torch.randperm(self.num_rsns)[: self.num_targets] for _ in graphs]
        )
        contexts: list[torch.Tensor] = []
        target_groups: list[list[torch.Tensor]] = []
        for graph, subject_targets in zip(graphs, targets, strict=True):
            groups = [graph.rsn_ids == rsn for rsn in subject_targets]
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
    return Data(
        x=data.x[region_ids],
        edge_index=old_to_new[data.edge_index[:, keep_edges]],
        edge_attr=edge_attr,
        region_ids=region_ids,
        num_nodes=region_ids.numel(),
    )
