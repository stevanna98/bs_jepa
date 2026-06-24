from __future__ import annotations

import math

from bsjepa.training import _subnetwork_loss_rows


def test_subnetwork_loss_rows_rank_and_name_subnetworks() -> None:
    rows = _subnetwork_loss_rows(
        epoch=3,
        rsn_loss_sums={0: 4.0, 1: 9.0},
        rsn_row_counts={0: 4, 1: 3},
        num_rsns=3,
        rsn_names=["visual", "default", "limbic"],
    )

    assert rows[0]["rsn_name"] == "visual"
    assert rows[0]["prediction_loss"] == 1.0
    assert rows[0]["rank_by_loss"] == 1
    assert rows[0]["difficulty"] == "easiest"
    assert rows[1]["prediction_loss"] == 3.0
    assert rows[1]["rank_by_loss"] == 2
    assert rows[1]["difficulty"] == "hardest"
    assert rows[2]["target_node_count"] == 0
    assert rows[2]["rank_by_loss"] == ""
    assert rows[2]["difficulty"] == ""
    assert math.isnan(rows[2]["prediction_loss"])
