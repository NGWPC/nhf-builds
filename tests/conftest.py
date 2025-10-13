"""Conftests for the test suite"""

import pytest
from pyprojroot import here

from hydrofabric_builds import HFConfig
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
    return HFConfig(
        dx=3000, reference_divides_path=mock_geopackages[0], reference_flowlines_path=mock_geopackages[1]
    )


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
