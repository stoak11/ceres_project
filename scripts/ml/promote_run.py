#!/usr/bin/env python3
"""Point production.json at an existing run_id."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ceres_package.ml_logic.registry import promote_to_production


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_id", help="Run folder name, e.g. random_forest_default_20260604-114526")
    p.add_argument("--alias", default=None, help="Production alias (default: manifest model_id)")
    args = p.parse_args()
    promote_to_production(args.run_id, alias=args.alias)


if __name__ == "__main__":
    main()
