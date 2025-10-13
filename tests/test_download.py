import geopandas as gpd

from hydrofabric_builds.pipeline.download import download_reference_data
from scripts.hf_runner import LocalRunner


class TestDownloadReferenceData:
    """Tests for download_reference_data function."""

    def test_loads_flowpaths(self, runner: LocalRunner) -> None:
        """Test that flowpaths are loaded from GeoPackage."""
        # Run the task
        runner.run_task(task_id="download", python_callable=download_reference_data, op_kwargs={})

        # Check XCom for flowpaths
        flowpaths = runner.ti.xcom_pull("download", key="reference_flowpaths")

        assert flowpaths is not None
        assert isinstance(flowpaths, gpd.GeoDataFrame)
        assert len(flowpaths) == 12
        assert "flowpath_id" in flowpaths.columns
        assert "VPUID" in flowpaths.columns

    def test_loads_divides(self, runner: LocalRunner) -> None:
        """Test that divides are loaded from GeoPackage."""

        # Run the task
        runner.run_task(task_id="download", python_callable=download_reference_data, op_kwargs={})

        # Check XCom for divides
        divides = runner.ti.xcom_pull("download", key="reference_divides")

        assert divides is not None
        assert isinstance(divides, gpd.GeoDataFrame)
        assert len(divides) == 12
        assert "divide_id" in divides.columns
