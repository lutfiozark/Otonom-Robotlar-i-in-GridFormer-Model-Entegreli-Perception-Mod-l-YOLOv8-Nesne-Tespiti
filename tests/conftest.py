"""Pytest configuration and fixtures."""

import pytest
import os
import sys
import torch
import numpy as np

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture(scope="session")
def device():
    """Get available device (GPU if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_image():
    """Create a sample image tensor for testing."""
    return torch.randn(1, 3, 256, 256)


@pytest.fixture
def sample_batch():
    """Create a sample batch of images for testing."""
    return torch.randn(4, 3, 256, 256)


@pytest.fixture
def weather_conditions():
    """List of weather conditions for testing."""
    return ['clear', 'fog', 'rain', 'snow', 'storm']


@pytest.fixture(scope="session")
def test_data_dir():
    """Test data directory path."""
    return os.path.join(os.path.dirname(__file__), '..', 'data', 'test')


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (skip if no GPU available)"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip GPU tests if no GPU available."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="GPU not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
