"""BS-JEPA model, graph encoders, and predictor."""

from __future__ import annotations

import copy
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv, GCNConv
from torch_geometric.utils import unbatch

from .masking import MaskOutput, extract_subgraph

GraphLayer = Literal["gcn", "gat"]
FeatureMode = Literal["passthrough", "conv1d"]


class TemporalConv(nn.Module):
    """Trainable BOLD time-series feature extractor."""

    def __init__(self, out_channels: int, kernel_size: int = 7) -> None:
        super().__init__()
        hidden = max(out_channels // 2, 1)
        self.convs = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size, stride=2, padding=kernel_size // 2),
            nn.GELU(),
            nn.Conv1d(hidden, out_channels, kernel_size, stride=2, padding=kernel_size // 2),
            nn.GELU(),
        )
        self.projection = nn.Linear(2 * out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.convs(x.unsqueeze(1))
        pooled = torch.cat([hidden.mean(-1), hidden.amax(-1)], dim=-1)
        return self.projection(pooled)


class GraphNetwork(nn.Module):
    """GCN/GATv2 node network used as either encoder or predictor."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        *,
        kind: GraphLayer,
        num_layers: int,
        heads: int,
        dropout: float,
        num_regions: int | None,
        feature_mode: FeatureMode = "passthrough",
        feature_dim: int = 64,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        if kind not in ("gcn", "gat"):
            raise ValueError(f"Unsupported graph layer: {kind}")
        if feature_mode not in ("passthrough", "conv1d"):
            raise ValueError(f"Unsupported feature mode: {feature_mode}")
        self.kind = kind
        self.dropout = dropout
        self.feature_extractor = (
            TemporalConv(feature_dim) if feature_mode == "conv1d" else None
        )
        projection_input = feature_dim if self.feature_extractor else in_channels
        self.input_projection = nn.Linear(projection_input, hidden_channels)
        self.region_embedding = (
            nn.Embedding(num_regions, hidden_channels) if num_regions is not None else None
        )

        dimensions = [hidden_channels] * num_layers + [out_channels]
        if kind == "gcn":
            self.layers = nn.ModuleList(
                GCNConv(dimensions[i], dimensions[i + 1], add_self_loops=True)
                for i in range(num_layers)
            )
        else:
            self.layers = nn.ModuleList(
                GATv2Conv(
                    dimensions[i], dimensions[i + 1], heads=heads, concat=False,
                    dropout=dropout, edge_dim=1, add_self_loops=True,
                )
                for i in range(num_layers)
            )
        self.norms = nn.ModuleList(
            nn.LayerNorm(dimensions[i + 1]) for i in range(num_layers)
        )

    def forward(self, data: Data) -> torch.Tensor:
        x = self.feature_extractor(data.x) if self.feature_extractor else data.x
        x = self.input_projection(x)
        if self.region_embedding is not None:
            region_ids = getattr(data, "region_ids", None)
            if region_ids is None:
                region_ids = torch.arange(data.num_nodes, device=x.device)
            x = x + self.region_embedding(region_ids)

        edge_weight = data.edge_attr.squeeze(-1) if data.edge_attr is not None else None
        for index, (layer, norm) in enumerate(zip(self.layers, self.norms, strict=True)):
            x = (
                layer(x, data.edge_index, edge_weight)
                if self.kind == "gcn"
                else layer(x, data.edge_index, data.edge_attr)
            )
            x = norm(x)
            if index + 1 < len(self.layers):
                x = F.dropout(F.gelu(x), self.dropout, self.training)
        return x


class BSJEPA(nn.Module):
    """Brain Subnetwork Joint-Embedding Predictive Architecture."""

    def __init__(self, encoder: nn.Module, predictor: nn.Module, embed_dim: int) -> None:
        super().__init__()
        self.context_encoder = encoder
        self.target_encoder = copy.deepcopy(encoder)
        self.target_encoder.requires_grad_(False)
        self.predictor = predictor
        self.mask_token = nn.Parameter(torch.empty(embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def train(self, mode: bool = True) -> "BSJEPA":
        super().train(mode)
        self.target_encoder.eval()  # EMA targets must not receive dropout noise.
        return self

    def forward(
        self, batch: Batch, masks: MaskOutput, *, return_groups: bool = False
    ) -> tuple[torch.Tensor, ...]:
        graphs = batch.to_data_list()
        with torch.no_grad():
            target_all = unbatch(self.target_encoder(batch), batch.batch)

        context_graphs = [
            extract_subgraph(graph, mask)
            for graph, mask in zip(graphs, masks.context_node_masks, strict=True)
        ]
        context_batch = Batch.from_data_list(context_graphs)
        context_embeddings = self.context_encoder(context_batch)
        context_per_graph = unbatch(context_embeddings, context_batch.batch)

        predictor_graphs: list[Data] = []
        target_flags: list[torch.Tensor] = []
        target_embeddings: list[torch.Tensor] = []
        row_group_ids: list[torch.Tensor] = []
        group_rsn_ids: list[torch.Tensor] = []
        group_index = 0

        for subject, graph in enumerate(graphs):
            context_mask = masks.context_node_masks[subject]
            context_ids = context_mask.nonzero(as_tuple=True)[0]
            node_embeddings = self.mask_token.expand(graph.num_nodes, -1).clone()
            node_embeddings = node_embeddings.index_copy(
                0, context_ids, context_per_graph[subject]
            )
            for target_index, target_mask in enumerate(
                masks.target_subnetwork_masks[subject]
            ):
                if not bool(target_mask.any()):
                    continue
                reconnected = Data(
                    x=node_embeddings,
                    edge_index=graph.edge_index,
                    edge_attr=graph.edge_attr,
                    num_nodes=graph.num_nodes,
                )
                subgraph = extract_subgraph(reconnected, context_mask | target_mask)
                is_target = target_mask[subgraph.region_ids]
                predictor_graphs.append(subgraph)
                target_flags.append(is_target)
                target_embeddings.append(target_all[subject][subgraph.region_ids[is_target]])
                if return_groups:
                    count = int(is_target.sum())
                    row_group_ids.append(
                        torch.full((count,), group_index, device=batch.x.device, dtype=torch.long)
                    )
                    group_rsn_ids.append(masks.target_rsn_ids[subject, target_index])
                    group_index += 1

        if not predictor_graphs:
            raise RuntimeError("No target nodes were found; check the atlas RSN mapping")
        prediction_batch = Batch.from_data_list(predictor_graphs)
        predictions = self.predictor(prediction_batch)[torch.cat(target_flags)]
        outputs: tuple[torch.Tensor, ...] = (
            predictions,
            torch.cat(target_embeddings),
            context_embeddings,
        )
        if return_groups:
            outputs += (torch.cat(row_group_ids), torch.stack(group_rsn_ids))
        return outputs

    @torch.no_grad()
    def encode(self, batch: Batch) -> torch.Tensor:
        return self.target_encoder(batch)


def build_bsjepa(
    *,
    in_channels: int,
    num_regions: int,
    embed_dim: int,
    encoder_type: GraphLayer = "gcn",
    encoder_hidden: int = 256,
    encoder_layers: int = 4,
    encoder_heads: int = 4,
    encoder_dropout: float = 0.0,
    feature_mode: FeatureMode = "passthrough",
    feature_dim: int = 64,
    predictor_type: GraphLayer = "gcn",
    predictor_hidden: int = 256,
    predictor_layers: int = 2,
    predictor_heads: int = 4,
    predictor_dropout: float = 0.0,
    region_positional_encoding: bool = True,
) -> BSJEPA:
    """Build a BS-JEPA model from explicit hyperparameters."""
    positional_regions = num_regions if region_positional_encoding else None
    encoder = GraphNetwork(
        in_channels, encoder_hidden, embed_dim, kind=encoder_type,
        num_layers=encoder_layers, heads=encoder_heads, dropout=encoder_dropout,
        num_regions=positional_regions, feature_mode=feature_mode, feature_dim=feature_dim,
    )
    predictor = GraphNetwork(
        embed_dim, predictor_hidden, embed_dim, kind=predictor_type,
        num_layers=predictor_layers, heads=predictor_heads, dropout=predictor_dropout,
        num_regions=positional_regions,
    )
    return BSJEPA(encoder, predictor, embed_dim)
