#!/usr/bin/env python3
"""CLI: regenerate ``data/NDVI/`` CSVs."""

from ceres_package.ndvi.export import export_ndvi_products


def main() -> None:
    paths = export_ndvi_products()
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
