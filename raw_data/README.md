# Local data cache layout

This tree is **not** the source of truth — files here are downloaded from **`gs://ceres-ai-bucket`** via `ceres_package.ml_logic.data.load_from_gcp()`.

Tracked in git:

- `.gitkeep` — preserves folders after clone
- `meteofrance/meteoREADME.md` — team meteo documentation

Ignored (see `.gitignore`): CSV, parquet, and other large files under these paths.

## Layout

| Path | Role |
|------|------|
| `agrestesaa/` | Agreste yield cache |
| `meteofrance/departements/` | Per-dept hourly meteo (`dept_XX.csv`) |
| `meteofrance/france/` | Consolidated national meteo |
| `soil_grid/` | SoilGrids extract |
| `ndvi/` | Legacy NDVI cache (products also under `data/NDVI/`) |

New ML registry, NDVI module, and GCP docs live under `ceres_package/`, `docs/`, and `infra/` — this folder is unchanged in purpose.
