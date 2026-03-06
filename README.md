# varistar 🌟

![PyPI - Version](https://img.shields.io/pypi/v/varistar?style=flat-square)
![PyPI - License](https://img.shields.io/pypi/l/varistar?style=flat-square)
[![Python Versions](https://img.shields.io/pypi/pyversions/varistar.svg?style=flat-square)](https://pypi.org/project/varistar/)

**varistar** is a Python package designed to simplify the management and interaction with timeseries and lightcurve data 
coming from multiple sources.

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

## License

This project is licensed under the **GNU General Public License v3 (GPLv3)**. This ensures the software remains free and open for the scientific community. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions make the scientific community stronger! 
1. Check out the [Contributing Guidelines](CONTRIBUTING.md).
2. Adhere to the [Code of Conduct](CODE_OF_CONDUCT.md).
3. Open a [Feature Request](https://github.com/jj-sm/varistar/issues) for new ideas.

## Citation

If you use **varistar** in your research or publications, please cite it using the metadata provided in the `CITATION.cff` file, or click the **"Cite this repository"** button in the GitHub sidebar.

---
*Maintained by [Juan José Sánchez Medina](mailto:pip@jjsm.science), BSc. Astronomy Student (Pontificia Universidad Católica de Chile)*