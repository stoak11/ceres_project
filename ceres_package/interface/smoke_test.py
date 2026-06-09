#!/usr/bin/env python3
"""Offline smoke test — run before asking the team to pull."""
from __future__ import annotations

import json
import sys

from ceres_package.interface.predictor import Predictor


def main() -> int:
    print("Loading model + features…")
    result = Predictor().predict("51", 2023)

    if result["actual_qha"] != 85.0:
        print("FAIL: expected actual_qha=85.0 for dept 51 / 2023")
        print(json.dumps(result, indent=2))
        return 1

    if not isinstance(result["predict_qha"], float):
        print("FAIL: predict_qha is not a float")
        return 1

    print("PASS")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
