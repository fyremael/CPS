from __future__ import annotations

import argparse
import json
from pathlib import Path

from .campaign import aggregate_campaign, evaluate_campaign
from .checkpoints import download_native_checkpoint
from .config import load_probe_config
from .continuation import ContinuationConfig, ContinuationControl, run_matched_continuation
from .native_state import discover_native_checkpoint
from .registry import list_run_specs
from .runner import run_longitudinal, run_probe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cps-pythia")
    sub = parser.add_subparsers(dest="command", required=True)

    probe = sub.add_parser("probe", help="run one projected optimizer-state CPS probe")
    probe.add_argument("config", type=Path)

    longitudinal = sub.add_parser("longitudinal", help="run a checkpoint sequence")
    longitudinal.add_argument("config", type=Path)
    longitudinal.add_argument("--revisions", nargs="+", required=True)

    inspect = sub.add_parser("inspect-native", help="inspect a native GPT-NeoX checkpoint")
    inspect.add_argument("checkpoint_dir", type=Path)

    download = sub.add_parser("download-native", help="download native optimizer-state files")
    download.add_argument("repo_id")
    download.add_argument("revision")
    download.add_argument("local_dir", type=Path)

    aggregate = sub.add_parser("aggregate", help="aggregate manifests into a CSV feature table")
    aggregate.add_argument("artifact_root", type=Path)
    aggregate.add_argument("output_csv", type=Path)

    evaluate = sub.add_parser("evaluate", help="evaluate incremental predictive value")
    evaluate.add_argument("csv_path", type=Path)
    evaluate.add_argument("--label-column", required=True)
    evaluate.add_argument("--group-column", required=True)
    evaluate.add_argument("--output-json", type=Path, required=True)

    continuation = sub.add_parser("continuation", help="run a matched continuation experiment")
    continuation.add_argument("--model-id", default="EleutherAI/pythia-70m")
    continuation.add_argument("--revision", default="step1000")
    continuation.add_argument("--steps", type=int, default=20)
    continuation.add_argument("--lr-scale", type=float, default=0.8)
    continuation.add_argument("--output-dir", default="artifacts/pythia/continuation")

    sub.add_parser("registry", help="print the built-in Pythia registry")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "probe":
        print(run_probe(load_probe_config(args.config)))
    elif args.command == "longitudinal":
        print(run_longitudinal(load_probe_config(args.config), list(args.revisions)))
    elif args.command == "inspect-native":
        print(json.dumps(discover_native_checkpoint(args.checkpoint_dir).to_dict(), indent=2))
    elif args.command == "download-native":
        print(
            json.dumps(
                download_native_checkpoint(args.repo_id, args.revision, args.local_dir).to_dict(),
                indent=2,
            )
        )
    elif args.command == "aggregate":
        print(aggregate_campaign(args.artifact_root, args.output_csv))
    elif args.command == "evaluate":
        print(
            evaluate_campaign(
                args.csv_path,
                label_column=args.label_column,
                group_column=args.group_column,
                output_json=args.output_json,
            )
        )
    elif args.command == "continuation":
        config = ContinuationConfig(
            model_id=args.model_id,
            revision=args.revision,
            steps=args.steps,
            intervention=ContinuationControl(name="cps", learning_rate_scale=args.lr_scale),
            output_dir=args.output_dir,
        )
        print(run_matched_continuation(config))
    elif args.command == "registry":
        print(json.dumps([item.to_dict() for item in list_run_specs()], indent=2))


if __name__ == "__main__":
    main()
