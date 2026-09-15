# Changelog

## v0.1.11 - 2026-09-15

- Fixed `varistar.catalog` raising `ModuleNotFoundError` on import — it referenced a `varistar.catalog.gaia` module that was never created; the dead import/export has been removed (`gaiadr3.py` already covers Gaia data loading).
- Standardized all module and function docstrings to NumPy convention (matching numpy/scipy/astropy), with complete `Parameters`/`Returns` sections.
- Enabled `ruff`'s pydocstyle rules (NumPy convention) to keep documentation consistent going forward.

## v0.1.10 - 2026-09-15

- Republished with corrected `CITATION.cff` metadata so Zenodo can archive the release correctly.

## v0.1.9 - 2026-09-15

- Added `catalog/gaiadr3.py`, supporting Gaia DR3 epoch photometry CSV files (2026-09-03).
- Fixed license mismatch: `pyproject.toml` and `CITATION.cff` claimed GPLv3 while the `LICENSE` file and GitHub repository metadata said MIT; aligned everything to MIT and fixed the invalid SPDX identifier in `CITATION.cff`.
- Fixed a `ruff` lint failure (unused `load_gaia` import) that was blocking the publish workflow.
- Removed the redundant `publish-unsafe.yml` workflow, which duplicated `publish.yml`'s tag-triggered PyPI publish without running tests first.
