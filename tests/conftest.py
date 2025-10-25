"""Conftests for the test suite"""

import geopandas as gpd
import pandas as pd
import pytest
from pyprojroot import here
from shapely.geometry import LineString, Point, Polygon

from hydrofabric_builds import HFConfig
from hydrofabric_builds.schemas.hydrofabric import Aggregations, Classifications
from scripts.hf_runner import LocalRunner, TaskInstance


@pytest.fixture
def task_instance() -> TaskInstance:
    """Fixture providing a TaskInstance."""
    return TaskInstance()


@pytest.fixture
def mock_geopackages() -> tuple[str, str]:
    """Create a temporary GeoPackage with test data."""
    return str(here() / "tests/data/sample_divides.parquet"), str(
        here() / "tests/data/sample_flowpaths.parquet"
    )


@pytest.fixture
def sample_config(mock_geopackages: str) -> HFConfig:
    """Fixture providing a sample HFConfig."""
    return HFConfig(reference_divides_path=mock_geopackages[0], reference_flowpaths_path=mock_geopackages[1])


@pytest.fixture
def runner(sample_config: HFConfig) -> LocalRunner:
    """Fixture providing a LocalRunner instance."""
    return LocalRunner(sample_config)


@pytest.fixture
def expected_graph() -> dict[str, list[str]]:
    return {
        "6720675": ["6722501"],
        "6720683": ["6720773", "6720689"],
        "6720797": ["6720703", "6720701"],
        "6720703": ["6720683", "6720651"],
        "6720689": ["6720681", "6720679"],
        "6720679": ["6720677", "6720675"],
    }


@pytest.fixture
def base_crs() -> str:
    return "EPSG:4326"


@pytest.fixture
def gages_empty(base_crs: str) -> gpd.GeoDataFrame:
    cols = ["geometry", "state", "site_no", "name_plain", "name_raw", "description"]
    gdf = gpd.GeoDataFrame({c: [] for c in cols}, geometry="geometry", crs=base_crs)
    return gdf


@pytest.fixture
def gages_seed(base_crs: str) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(
        {
            "site_no": ["00000001"],
            "state": ["-"],
            "name_plain": ["-"],
            "name_raw": ["seed-name"],
            "description": ["-"],
            "geometry": [Point(-100.0, 40.0)],
        },
        geometry="geometry",
        crs=base_crs,
    )
    return gdf


def sample_flowpath_data() -> pd.DataFrame:
    """Create sample flowpath data for unit testing individual rules."""
    data = {
        "flowpath_id": [1, 2, 3, 4, 5, 6, 7, 8],
        "totdasqkm": [100.0, 50.0, 25.0, 10.0, 5.0, 2.0, 1.0, 0.5],
        "areasqkm_left": [5.0, 2.5, 1.5, 0.8, 2.0, 1.0, 0.5, 0.3],
        "lengthkm": [10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.5, 1.0],
        "streamorder": [3, 2, 2, 1, 1, 1, 1, 1],
        "hydroseq": [1, 2, 3, 4, 5, 6, 7, 8],
        "dnhydroseq": [0, 1, 1, 2, 3, 4, 4, 5],
        "mainstemlp": [100, 100, 200, 100, 200, 100, 100, 200],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_aggregate_data() -> Aggregations:
    """Sample aggregation data for testing."""
    return Aggregations(
        aggregates=[
            {
                "dn_id": "6720797",
                "up_id": "6720703",
                "line_geometry": LineString([(0, 0), (1, 1)]),
                "polygon_geometry": Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            },
            {
                "dn_id": "6720703",
                "up_id": "6720683",
                "line_geometry": LineString([(1, 1), (2, 2)]),
                "polygon_geometry": Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]),
            },
        ],
        independents=[
            {
                "ref_ids": "6720651",
                "line_geometry": LineString([(3, 3), (4, 4)]),
                "polygon_geometry": Polygon([(3, 3), (4, 3), (4, 4), (3, 4)]),
            }
        ],
        connectors=[
            {
                "ref_ids": "6720681",
                "line_geometry": LineString([(5, 5), (6, 6)]),
                "polygon_geometry": Polygon([(5, 5), (6, 5), (6, 6), (5, 6)]),
            }
        ],
        minor_flowpaths=[
            {
                "ref_ids": "6720689",
                "line_geometry": LineString([(7, 7), (8, 8)]),
                "polygon_geometry": Polygon([(7, 7), (8, 7), (8, 8), (7, 8)]),
            }
        ],
        small_scale_connectors=[],
    )


@pytest.fixture
def sample_classifications() -> Classifications:
    """Sample classification data for testing."""
    return Classifications(
        aggregation_pairs=[("6720703", "6720797"), ("6720683", "6720703")],
        minor_flowpaths=["6720689", "6720679"],
        independent_flowpaths=["6720651"],
        connector_segments=["6720681"],
        subdivide_candidates=[],
        upstream_merge_points=[],
        processed_flowpaths={"6720797", "6720703", "6720683", "6720651", "6720681"},
        cumulative_merge_areas={},
    )


@pytest.fixture
def sample_reference_data() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Sample reference flowpaths and divides for testing."""
    flowpaths = pd.DataFrame(
        {
            "flowpath_id": [6720797, 6720703, 6720683, 6720651, 6720681, 6720689, 6720679],
            "hydroseq": [1, 2, 3, 4, 5, 6, 7],
            "dnhydroseq": [0, 1, 2, 2, 3, 3, 4],
            "lengthkm": [1.0, 1.5, 2.0, 1.2, 0.8, 0.5, 0.6],
            "areasqkm": [5.0, 4.0, 3.0, 2.5, 1.5, 1.0, 0.8],
            "totdasqkm": [50.0, 40.0, 30.0, 20.0, 15.0, 10.0, 8.0],
            "streamorder": [3, 2, 2, 1, 1, 1, 1],
            "VPUID": ["01", "01", "01", "01", "01", "01", "01"],
            "geometry": [
                LineString([(0, 0), (1, 1)]),
                LineString([(1, 1), (2, 2)]),
                LineString([(2, 2), (3, 3)]),
                LineString([(3, 3), (4, 4)]),
                LineString([(5, 5), (6, 6)]),
                LineString([(7, 7), (8, 8)]),
                LineString([(9, 9), (10, 10)]),
            ],
        }
    )

    divides = pd.DataFrame(
        {
            "divide_id": [6720797, 6720703, 6720683, 6720651, 6720681, 6720689, 6720679],
            "areasqkm": [5.0, 4.0, 3.0, 2.5, 1.5, 1.0, 0.8],
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]),
                Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
                Polygon([(3, 3), (4, 3), (4, 4), (3, 4)]),
                Polygon([(5, 5), (6, 5), (6, 6), (5, 6)]),
                Polygon([(7, 7), (8, 7), (8, 8), (7, 8)]),
                Polygon([(9, 9), (10, 9), (10, 10), (9, 10)]),
            ],
        }
    )

    return gpd.GeoDataFrame(flowpaths, crs="EPSG:5070"), gpd.GeoDataFrame(divides, crs="EPSG:5070")
