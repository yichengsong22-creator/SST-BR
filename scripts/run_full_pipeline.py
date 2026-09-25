"""CLI for complete local/server RHB reproduction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sstbr.pipeline import run_full_pipeline
from sstbr.utils.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rhb_full.yaml")
    parser.add_argument("--reuse-cache", action="store_true")
    args = parser.parse_args()
    configure_logging()
    metrics = run_full_pipeline(Path(args.config), reuse_cache=args.reuse_cache)
    print(json.dumps(metrics["pooled"], indent=2))


if __name__ == "__main__":
    main()
