### Quickstart

Retrieve NHF archive from s3. Unpack to local `data` directory. The files are organized for the canonical configuration. If you are making changes, set the folders up accordingly and change in config.


To run the NHF build, you can use the example config, or make your own based on it. The full run commands are:
```sh
uv sync --all-extras
uv run python scripts/hf_runner.py --config configs/example_prvi_config.yaml
```

This will output an NHF GPKG.
