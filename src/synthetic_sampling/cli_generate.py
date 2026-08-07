"""CLI: generate prediction instances."""

from __future__ import annotations

import argparse
from pathlib import Path

from synthetic_sampling.surveys import (
    ALL_SURVEYS,
    ESS_SURVEYS,
    DataPaths,
    DatasetConfig,
    GeneratorConfig,
    list_surveys_detailed,
    load_config,
)
from synthetic_sampling.profiles import DatasetBuilder


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate survey prediction instances")
    ap.add_argument("--config", type=Path, default=None,
                    help="YAML config (paths/generator/dataset)")
    ap.add_argument("--survey", action="append", default=None,
                    help="Survey id (repeatable). Default: all")
    ap.add_argument("--ess-only", action="store_true")
    ap.add_argument("--list", action="store_true", help="List surveys and exit")
    ap.add_argument("--raw-data-dir", type=Path, default=None)
    ap.add_argument("--metadata-dir", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None,
                    help="Output JSONL (default: output_dir/instances.jsonl)")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)

    if args.list:
        print(list_surveys_detailed())
        return 0

    if args.config:
        cfg = load_config(args.config)
        paths, gen_cfg, ds_cfg = cfg["paths"], cfg["generator"], cfg["dataset"]
    else:
        if not args.raw_data_dir:
            ap.error("--raw-data-dir or --config is required")
        paths = DataPaths.default_bundled(
            args.raw_data_dir, args.output_dir or "./outputs")
        gen_cfg, ds_cfg = GeneratorConfig(), DatasetConfig()

    if args.metadata_dir:
        paths.metadata_dir = Path(args.metadata_dir)
    if args.output_dir:
        paths.output_dir = Path(args.output_dir)
    if args.seed is not None:
        ds_cfg.seed = args.seed

    if args.ess_only:
        surveys = list(ESS_SURVEYS)
    elif args.survey:
        surveys = args.survey
    elif ds_cfg.surveys:
        surveys = ds_cfg.surveys
    else:
        surveys = list(ALL_SURVEYS)

    builder = DatasetBuilder(paths, ds_cfg, gen_cfg)
    instances = builder.build_dataset(surveys)
    out = args.out or (paths.output_dir / "instances.jsonl")
    builder.save_jsonl(instances, out)
    print(f"wrote {len(instances)} instances -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
