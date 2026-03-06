# ogle 🔭

![PyPI - Version](https://img.shields.io/pypi/v/ogle?style=flat-square)
![PyPI - License](https://img.shields.io/pypi/l/ogle?style=flat-square)
[![Python Versions](https://img.shields.io/pypi/pyversions/ogle.svg?style=flat-square)](https://pypi.org/project/ogle/)

**ogle** is a Python package designed to simplify the management, retrieval, and interaction with **Optical Gravitational Lensing Experiment (OGLE)** data. It provides a streamlined interface for astronomers and researchers to handle light-curve data and lensing events.

---

## Features

* **Data Retrieval**: Automated utilities to fetch data from OGLE databases.
* **Format Conversion**: Seamlessly convert OGLE-specific data into `astropy` tables or `pandas` DataFrames.
* **Analysis Tools**: Functions for processing gravitational lensing light-curves.
* **Modern Workflow**: Full support for `uv`, `pip`, and type-hinting for high-performance research.

## Installation

Install the stable version from [PyPI](https://pypi.org/project/ogle/):

```bash
pip install ogle
```

Or, if you prefer using [uv](https://github.com/astral-sh/uv):

```bash
uv add ogle
```

## Documentation and Usage

All the docs can be found at [docs.jjsm.science/ogle](https://docs.jjsm.science/ogle)

## Development

This project is built using the latest Python standards. If you are using this as a template or contributing:

1.  **Clone the repo**:
    ```bash
    git clone https://github.com/jj-sm/ogle.git
    cd ogle
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
3. Open a [Feature Request](https://github.com/jj-sm/ogle/issues) for new ideas.

## Citation

If you use **ogle** in your research or publications, please cite it using the metadata provided in the `CITATION.cff` file, or click the **"Cite this repository"** button in the GitHub sidebar.

---
*Maintained by [Juan José Sánchez Medina](mailto:pip@jjsm.science), BSc. Astronomy Student (Pontificia Universidad Católica de Chile)*