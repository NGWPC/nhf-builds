"""Conftests for the test suite"""

import pytest

from hydrofabric_builds import HFConfig
from scripts.hf_runner import LocalRunner, TaskInstance


@pytest.fixture
def sample_config() -> HFConfig:
    """Fixture providing a sample HFConfig."""
    return HFConfig(dx=3000)


@pytest.fixture
def runner(sample_config: HFConfig) -> LocalRunner:
    """Fixture providing a LocalRunner instance."""
    return LocalRunner(sample_config)


@pytest.fixture
def task_instance() -> TaskInstance:
    """Fixture providing a TaskInstance."""
    return TaskInstance()
