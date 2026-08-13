# Development Commands

Run these commands from the repository root.

## Install all dependencies

This repo is managed through [UV](https://docs.astral.sh/uv/getting-started/installation/)
The following command installs the project's base dependencies, the `docs` optional extra, and all dependency groups (`dev`, `examples`, and `tests`):

```bash
uv sync --all-extras --all-groups
```

Python 3.12 or newer is required.

## Sync input data using the `justfile`

`just` calls series of commands called "recipes" similar to a `make` file. Install on linux with `apt get just` or follow linked readme for other platforms. After installing `just`, you can use the following commands to set up the data sources for `nhf-builds`. You can also use `just` to build hydrofabrics for each domain or specify a config.

Provide AWS credentials in the current shell:

```bash
export AWS_DEFAULT_REGION="us-east-1"
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_SESSION_TOKEN="..."  # Required for temporary credentials
```

Verify that AWS recognizes the credentials:

```bash
aws sts get-caller-identity
```

Then sync the input data for the desired domain:

```bash
just sync       # CONUS
just sync-ak    # Alaska
just sync-hi    # Hawaii
just sync-prvi  # Puerto Rico and the US Virgin Islands
```

To select a different OCONUS reference-fabric version, pass the `oconus-version` variable:

```bash
just oconus-version=0.1.8 sync-ak
```

> **Warning:** The sync recipes overwrite the corresponding input datasets under `data/`.

### AWS credential handling

Exporting AWS credentials in the shell is functionally sufficient because `just` and its child processes inherit those environment variables. The `justfile` also automatically loads variables from a repository-root `.env` file.

Avoid committing credentials or entering long-lived secrets directly into commands that may be saved in shell history. When available, prefer an AWS SSO or named-profile workflow for data synchronization:

```bash
aws sso login --profile ngwpc-test
AWS_PROFILE=ngwpc-test just sync
```

An AWS profile is sufficient for the `aws s3` commands used by the sync recipes. Some hydrofabric build paths access S3 credentials directly through `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_SESSION_TOKEN`, so a profile alone may not be sufficient for every build configuration.

## Run the hydrofabric build

Run the main build script directly with the CONUS example configuration:

```bash
uv run python scripts/hf_runner.py --config configs/example_config.yaml
```

Alternatively, use a domain-specific `just` recipe:

```bash
just build-conus
just build-ak
just build-hi
just build-prvi
```

Run the build with a custom configuration:

```bash
just build "configs/my_custom_config.yaml"
```

## Run tests

### Run the complete test suite

```bash
uv run pytest tests
```

### Run all tests in one module

```bash
uv run pytest tests/test_config.py
```

### Run one test in a module

```bash
uv run pytest tests/test_config.py::test_from_yaml_1
```

For a test method defined inside a class, include the class name in the pytest node ID. For example:

```bash
uv run pytest tests/test_graph.py::TestBuildGraphUnit::test_simple_linear_network
```

### Development
To ensure that hydrofabric-builds follows the specified structure, be sure to install the local dev dependencies and run `uv run pre-commit install`
