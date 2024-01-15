# ElikoPy

[![Documentation Status](https://readthedocs.org/projects/elikopy/badge/?version=latest)](https://elikopy.readthedocs.io/en/latest/?badge=latest) [![PyPI](https://img.shields.io/pypi/v/elikopy?label=pypi%20package)](https://pypi.org/project/elikopy//) ![GitHub repo size](https://img.shields.io/github/repo-size/Hyedryn/elikopy)

ElikoPy is a Python library designed to simplify the processing of diffusion imaging for microstructural analysis.

## Installation

### Prerequistes

ElikoPy requires [Python](https://www.python.org/) v3.8+.

### Installation Steps

Clone the repository and install the dependencies:

```sh
git clone https://github.com/Hyedryn/elikopy.git
cd elikopy
python -m pip install .
```

If you wish to use movement correction or TBSS, ensure FSL is installed and available in your path.

**Note:** The DIAMOND microstructural model is not publicly available. If you have it, add the executable to your path. Otherwise, you won't be able to use this algorithm.

## Development

Interested in contributing? Wonderful!

Feel free to open issues or pull requests. Your contributions are welcome!

## Publications & Citations

If you use ElikoPy in your research, please cite it using the package DOI.
