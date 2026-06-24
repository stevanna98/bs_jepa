#!/usr/bin/env python
"""Run neurobiological hypothesis-test analyses for a trained BS-JEPA checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from bsjepa.hypothesis_tests import run_neurobiological_hypothesis_tests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate neurobiological hypotheses for a trained BS-JEPA model"
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/neurobiological_hypothesis_tests"),
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="Override a YAML config value before rebuilding data/model",
    )
    parser.add_argument("--device", default=None, help="Override device, e.g. cpu or cuda")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Only save PNG plots instead of PNG plus PDF",
    )
    parser.add_argument(
        "--skip-downstream-probes",
        action="store_true",
        help="Use logged downstream metrics only; do not rerun probe training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = run_neurobiological_hypothesis_tests(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        overrides=args.set,
        device=args.device,
        batch_size=args.batch_size,
        save_pdf=not args.no_pdf,
        run_downstream_probes=not args.skip_downstream_probes,
    )
    report = state.paths.reports / "neurobiological_hypothesis_report.md"
    print(f"results_dir={state.paths.root}")
    print(f"report={report}")


if __name__ == "__main__":
    main()
