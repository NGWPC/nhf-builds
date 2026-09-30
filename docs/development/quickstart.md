### Quickstart

Retrieve NHF archive from s3. Unpack to local `data` directory. The files are organized for the canonical configuration. If you are making changes, set the folders up accordingly and change in config.


To run the NHF build, you can use the example config, or make your own based on it. The full run commands are:
```sh
uv sync --all-extras
uv run python scripts/hf_runner.py --config configs/example_prvi_config.yaml
```

This will output an NHF GPKG.

The build process will depend on your CPU and memory availability. The network process can take 20+ minutes. There are over 60 divide attributes which can take 5+ hours - by far the longest step in the process. OCONUS domains are much shorter than CONUS. For debugging, you can turn on and off build steps and build off a pre-existing NHF network. Turning off divide attribute calculation will expediate things significantly.

Note:
Always re-run hydrolocations and reservoir_da when running lakes and gages.
