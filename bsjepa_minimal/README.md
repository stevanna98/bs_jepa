# Minimal BS-JEPA

This folder is a self-contained implementation of BS-JEPA pretraining. It contains only the graph model, RSN masking, FC-to-graph data loading, training losses, EMA/optimization loop, configuration, and command-line entry point. It has no dependency on the parent repository's `src` package.

## Run

From this directory:

```bash
python -m pip install -r requirements.txt
python pretrain.py
```

The default configuration runs a small synthetic pretraining job and writes checkpoints to `outputs/`. Override individual settings without editing the YAML:

```bash
python pretrain.py --set training.epochs=10 --set model.encoder_type=gat
```

For real data, set `data.source` to either a `.pkl`/`.pt` dictionary of subjects or a directory containing one `.pt`/`.npz` file per subject, and set `data.atlas_csv`. Subject records must provide BOLD and/or FC arrays using the configured keys. The atlas CSV must contain `rsn_id` and `rsn_name` columns, with one row per region.

When training from raw BOLD, set `model.feature_mode: conv1d` to learn temporal features in the encoder. `fc_row` and `ones` node features should use `passthrough`.
