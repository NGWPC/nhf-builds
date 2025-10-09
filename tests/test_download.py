import geopandas as gpd

from hydrofabric_builds import HFConfig
from hydrofabric_builds.pipeline.download import download_reference_data
from scripts.hf_runner import LocalRunner


class TestDownloadReferenceData:
    """Tests for download_reference_data function."""

    def test_loads_flowpaths_from_geopackage(self, runner: LocalRunner, sample_config: HFConfig) -> None:
        """Test that flowpaths are loaded from GeoPackage."""
        # Run the task
        runner.run_task(task_id="download", python_callable=download_reference_data, op_kwargs={})

        # Check XCom for flowpaths
        flowpaths = runner.ti.xcom_pull("download", key="reference_flowpaths")

        assert flowpaths is not None
        assert isinstance(flowpaths, gpd.GeoDataFrame)
        assert len(flowpaths) == 3
        assert "id" in flowpaths.columns
        assert "name" in flowpaths.columns

    def test_loads_divides_from_geopackage(self, runner: LocalRunner) -> None:
        """Test that divides are loaded from GeoPackage."""

        # Run the task
        runner.run_task(task_id="download", python_callable=download_reference_data, op_kwargs={})

        # Check XCom for divides
        divides = runner.ti.xcom_pull("download", key="reference_divides")

        assert divides is not None
        assert isinstance(divides, gpd.GeoDataFrame)
        assert len(divides) == 3
        assert "divide_id" in divides.columns
