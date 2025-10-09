"""Conftests for the test suite"""

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point

from hydrofabric_builds import HFConfig
from scripts.hf_runner import LocalRunner, TaskInstance


@pytest.fixture
def task_instance() -> TaskInstance:
    """Fixture providing a TaskInstance."""
    return TaskInstance()


@pytest.fixture
def mock_geopackage(tmp_path: Path) -> str:
    """Create a temporary GeoPackage with test data."""
    gpkg_path = tmp_path / "test_reference.gpkg"

    flowpaths = gpd.GeoDataFrame(
        {
            "id": [1, 2, 3],
            "name": ["flowpath_1", "flowpath_2", "flowpath_3"],
            "length": [100.0, 200.0, 150.0],
        },
        geometry=[
            LineString([(0, 0), (1, 1)]),
            LineString([(1, 1), (2, 2)]),
            LineString([(2, 2), (3, 3)]),
        ],
        crs="EPSG:4326",
    )

    # Create mock divides layer
    divides = gpd.GeoDataFrame(
        {
            "id": [10, 20, 30],
            "divide_id": ["div_1", "div_2", "div_3"],
            "area": [50.0, 75.0, 60.0],
        },
        geometry=[
            Point(0.5, 0.5),
            Point(1.5, 1.5),
            Point(2.5, 2.5),
        ],
        crs="EPSG:4326",
    )

    flowpaths.to_file(gpkg_path, layer="reference_flowpaths", driver="GPKG")
    divides.to_file(gpkg_path, layer="reference_divides", driver="GPKG")

    return str(gpkg_path)


@pytest.fixture
def sample_config(mock_geopackage: str) -> HFConfig:
    """Fixture providing a sample HFConfig."""
    return HFConfig(dx=3000, reference_fabric_path=mock_geopackage)


@pytest.fixture
def runner(sample_config: HFConfig) -> LocalRunner:
    """Fixture providing a LocalRunner instance."""
    return LocalRunner(sample_config)
