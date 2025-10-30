# Divide Attributes Build

The divide attributes task calculates zonal statistics for divides from a number of rasters.

To run:

1. Download source rasters from AWS test account and maintain folder structure:
```
mkdir ./data/divide_attributes
aws s3 sync s3://edfs-data/attributes/5070/ ./data/divide_attributes
```

2. `divide_attributes_config.yaml` is set to run on a small test case. Change `divides_path` to your desired domain.

3. Multiprocessing:
The default setting is to use total CPU cores. To specify another core number:
- Set the process number in the the `HFConfig` (`configs/example_config.yaml`). Set `divide_attributes_processes` to an int.

Then, in `divide_attributes_config.yaml`, do one of the following:
- set `split_vpu` to `True`: This will automatically split your domain file by VPU and create temp GPKGs
- add a `divides_path_list` key where the value is a list of paths: `['divides_1.gpkg', divides_2.gpkg']`

Final output will always merge all divides.

!!! note
    A number of temporary files will be created during the process. If multiprocessing, these will be copies of rasters and parquets with zonal statistics outputs for each process. By default these are always deleted when the run ends, even if it errors. If you are debugging and want to see the temporary outputs, set `debug: True` in `divide_attributes_config.yaml`

!!! note
    For small test cases, using a single process is recommended to avoid overhead of copying rasters. Multiprocessing is recommended for full pipeline runs.

4. Run using the `hf_runner.py` script: `python scripts/hf_runner.py --config ./configs/example_config.yaml`
