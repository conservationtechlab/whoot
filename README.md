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
- `model-training`: Required for running scripts in `whoot/model_training`, make sure to add either `cpu` or `cu128`
- `dev`: Installs linters pylint and flake8. MUST be used by developers of whoot
- `data_exporters` Installs LabelStudioSdk

## Usage

Once the enviroment is activated, you should be able to do `python path/to/script.py` to run any of the whoot scripts. If a script states a package is missing, you might not be using the virtual enviroment.

# Developer Notes

## Creating a new Project

When adding a new package, like `assess_birdnet` to the whoot toolkit, add your package name to the `[tool.setuptools]` section of `pyproject.toml` 

### Linting 

Style guidelines are listed in `.flake8` and `pylintrc`. To use these tools do the following

1) Follow the Installation Instructions, on pip install do `pip install -e .[dev,extra1,extra2,...]`.
2) Activate the environment

To run the linters run `python -m flake8` and `python -m pylint --recursive y PATH/TO/FILES.py`
In order to contribute to whoot, both of these must be cleared. 