from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Batch, Data

from bsjepa.masking import SubnetworkMaskCollator
from bsjepa.model import build_bsjepa


def _toy_graph() -> Data:
    num_nodes = 6
    source, target = torch.where(~torch.eye(num_nodes, dtype=torch.bool))
    edge_index = torch.stack([source, target])
    edge_attr = torch.linspace(0.1, 1.0, edge_index.shape[1]).unsqueeze(-1)
    return Data(
        x=torch.randn(num_nodes, 4),
        edge_index=edge_index,
        edge_attr=edge_attr,
        rsn_ids=torch.tensor([0, 0, 1, 1, 2, 2]),
        region_ids=torch.arange(num_nodes),
        num_nodes=num_nodes,
    )


@pytest.mark.parametrize("layer_type", ["gcn", "gat", "graphsage", "transformer", "gine"])
def test_graph_layer_supports_encoding_and_prediction(layer_type: str) -> None:
    torch.manual_seed(7)
    graphs = [_toy_graph(), _toy_graph()]
    model = build_bsjepa(
        in_channels=4,
        num_regions=6,
        embed_dim=8,
        encoder_type=layer_type,
        encoder_hidden=8,
        encoder_layers=2,
        encoder_heads=2,
        predictor_type=layer_type,
        predictor_hidden=8,
        predictor_layers=1,
        predictor_heads=2,
    )

    batch = Batch.from_data_list(graphs)
    encoded = model.encode(batch)

    assert encoded.shape == (12, 8)

    collator = SubnetworkMaskCollator(num_rsns=3, num_targets=1)
    batch, masks = collator(graphs)
    predictions, targets, context = model(batch, masks)

    assert predictions.shape == targets.shape
    assert predictions.shape[1] == 8
    assert context.shape == (8, 8)


@pytest.mark.parametrize(
    "mask_token_mode",
    ["shared", "random_per_target_node", "rsn_specific", "zero"],
)
def test_mask_token_modes_support_prediction(mask_token_mode: str) -> None:
    torch.manual_seed(11)
    graphs = [_toy_graph(), _toy_graph()]
    model = build_bsjepa(
        in_channels=4,
        num_regions=6,
        num_rsns=3,
        embed_dim=8,
        encoder_hidden=8,
        encoder_layers=1,
        predictor_hidden=8,
        predictor_layers=1,
        mask_token_mode=mask_token_mode,
    )
    batch, masks = SubnetworkMaskCollator(num_rsns=3, num_targets=1)(graphs)

    predictions, targets, context = model(batch, masks)

    assert predictions.shape == targets.shape
    assert predictions.shape[1] == 8
    assert context.shape[1] == 8


def test_zero_mask_token_mode_uses_zero_placeholders() -> None:
    model = build_bsjepa(
        in_channels=4,
        num_regions=6,
        embed_dim=8,
        encoder_hidden=8,
        encoder_layers=1,
        predictor_hidden=8,
        predictor_layers=1,
        mask_token_mode="zero",
    )

    placeholders = model._masked_node_embeddings(_toy_graph(), torch.ones(2, 8))

    assert torch.equal(placeholders, torch.zeros(6, 8))
    assert model.mask_token is None


def test_random_mask_token_mode_samples_fresh_node_placeholders() -> None:
    torch.manual_seed(13)
    model = build_bsjepa(
        in_channels=4,
        num_regions=6,
        embed_dim=8,
        encoder_hidden=8,
        encoder_layers=1,
        predictor_hidden=8,
        predictor_layers=1,
        mask_token_mode="random_per_target_node",
        random_mask_token_std=0.5,
    )

    first = model._masked_node_embeddings(_toy_graph(), torch.ones(2, 8))
    second = model._masked_node_embeddings(_toy_graph(), torch.ones(2, 8))

    assert model.mask_token is None
    assert not torch.allclose(first, second)
    assert not torch.allclose(first[0], first[1])


def test_rsn_specific_mask_token_mode_uses_one_token_per_rsn() -> None:
    torch.manual_seed(17)
    model = build_bsjepa(
        in_channels=4,
        num_regions=6,
        num_rsns=3,
        embed_dim=8,
        encoder_hidden=8,
        encoder_layers=1,
        predictor_hidden=8,
        predictor_layers=1,
        mask_token_mode="rsn_specific",
    )

    placeholders = model._masked_node_embeddings(_toy_graph(), torch.ones(2, 8))

    assert model.mask_token is None
    assert model.rsn_mask_tokens is not None
    assert torch.allclose(placeholders[0], placeholders[1])
    assert torch.allclose(placeholders[2], placeholders[3])
    assert torch.allclose(placeholders[4], placeholders[5])
    assert not torch.allclose(placeholders[0], placeholders[2])
