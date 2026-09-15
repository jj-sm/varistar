# varistar \~\*\~


[![PyPI - Version](https://img.shields.io/pypi/v/varistar?style=flat-round)](https://pypi.org/project/varistar/)
[![PyPI - License](https://img.shields.io/pypi/l/varistar?style=flat-round)](https://pypi.org/project/varistar/)
[![Python Versions](https://img.shields.io/pypi/pyversions/varistar.svg?style=flat-round)](https://pypi.org/project/varistar/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22772940.svg)](https://doi.org/10.5281/zenodo.22772940)


`varistar` is a Python package designed to simplify the management and interaction with timeseries and lightcurve data 
coming from multiple sources.

<p align="center">
  <img src="/markdown/EB.png" alt="example binary star image">
</p>


---

## Features

* **Data Retrieval**: Automated utilities to fetch data from varistar databases.
* **Format Conversion**: Seamlessly convert varistar-specific data into `astropy` tables or `pandas` DataFrames.
* **Analysis Tools**: Functions for processing gravitational lensing light-curves.
* **Modern Workflow**: Full support for `uv`, `pip`, and type-hinting for high-performance research.

## Installation

Install the stable version from [PyPI](https://pypi.org/project/varistar/):

```bash
pip install varistar
```

Or, if you prefer using [uv](https://github.com/astral-sh/uv):

```bash
uv add varistar
```

## Documentation and Usage

All the docs can be found at [docs.jjsm.science/varistar](https://docs.jjsm.science/varistar)

## Citation

```bibtex
@software{sanchez_medina_2026_22775284,
  author       = {Sánchez Medina, Juan José},
  title        = {varistar Python Package},
  month        = sep,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v0.1.12},
  doi          = {10.5281/zenodo.22775284},
  url          = {https://doi.org/10.5281/zenodo.22775284},
  swhid        = {swh:1:dir:841ac30ec6b30ec2ae1281c227b87b6c8b114ad3
                   ;origin=https://doi.org/10.5281/zenodo.22772939;vi
                   sit=swh:1:snp:6151bc2d06ff905533ed69544d9b9893b5e5
                   9179;anchor=swh:1:rel:de9138bf25a85f4c6f5ad344dc1d
                   f2aa95fff8e4;path=jj-sm-varistar-f1d4611
                  },
}
```

> [!TIP]
> Export latest BibTeX citation from [zenodo.org/records/22775284/export/bibtex](https://zenodo.org/records/22775284/export/bibtex)


## Development

This project is built using the latest Python standards. If you are using this as a template or contributing:

1.  **Clone the repo**:
    ```bash
    git clone https://github.com/jj-sm/varistar.git
    cd varistar
    ```
2.  **Sync the environment (using uv)**:
    ```bash
    uv sync
    ```
3.  **Run the test suite**:
    ```bash
    uv run pytest
    ```

### Releasing

Releases are cut with the `Makefile`, which runs lint + tests locally, bumps
the version, pushes, tags, waits for the PyPI publish workflow to succeed,
and only then creates the GitHub release:

```bash
make check              # ruff check + pytest (same gate as CI)
make lint                # ruff check only
make test                # pytest only
make format-check        # ruff format --check (advisory, not release-blocking)

make release VERSION=0.1.12
```

`make release` requires a clean working tree on `main`, up to date with
`origin/main`. It bumps the version in `pyproject.toml` and
`src/varistar/__init__.py`, commits (`chore: release vX.Y.Z`), pushes,
tags `vX.Y.Z`, pushes the tag (which triggers
[`.github/workflows/publish.yml`](.github/workflows/publish.yml) to test
and publish to PyPI), watches that workflow, and creates the GitHub
release only if it succeeds — a release is never created for a version
that failed to publish.

See [CHANGELOG.md](CHANGELOG.md) for release history.

## License

This project is licensed under the **MIT License**. See the [LICENSE.md](LICENSE.md) file for details.

## Contributing

1. Check out the [Contributing Guidelines](CONTRIBUTING.md).
2. Adhere to the [Code of Conduct](CODE_OF_CONDUCT.md).
3. Open a [Feature Request](https://github.com/jj-sm/varistar/issues) for new ideas.

## Citation

If you use **varistar** in your research or publications, please cite it using the metadata provided in the `CITATION.cff` file, or click the **"Cite this repository"** button in the GitHub sidebar.

---
*Maintained by [Juan José Sánchez Medina](mailto:pip@jjsm.science), BSc. Astronomy Student (Pontificia Universidad Católica de Chile)*