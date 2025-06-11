# whoot
Tools for capturing, analyzing, and parsing audio data

# Installation Instructions

## Default Python Instructions
1) Install Python>=3.10
2) Create a virtual enviroment via `python -m venv`
3) Activate the enviroment using an activate script:

- Windows: `.venv\Scripts\activate`
- macOS/Linux: `source .venv/bin/activate`

If this works, you should see in your command line `(whoot)`. If not check https://docs.python.org/3/library/venv.html#how-venvs-work

4) Run in project root `pip install -e .`

To install optional dependencies run `pip install -e .[extra1,extra2,...]`

Current support optional dependency collections include

- `cpu`: Installs torch and torchvision for CPU use only
- `cu128`: Installs torch and torchvision with Cuda 12.8 Binaries



## Developer Notes

When adding a new package, like `assess_birdnet` to the whoot toolkit, add your package name to the `[tool.setuptools]` section of `pyproject.toml` 