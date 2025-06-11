# whoot
Tools for capturing, analyzing, and parsing audio data

# Installation Instructions

## Default Python Instructions
1) Install Python>=3.10
2) Run in project root `pip install -e .`

To install optional dependencies run `pip install -e .[extra1,extra2,...]`

Current support optional dependency collections include

- `cpu`: Installs torch and torchvision for CPU use only
- `cu128`: Installs torch and torchvision with Cuda 12.8 Binaries



## Developer Notes

When adding a new package, like `assess_birdnet` to the whoot toolkit, add your package name to the `[tool.setuptools]` section of `pyproject.toml` 