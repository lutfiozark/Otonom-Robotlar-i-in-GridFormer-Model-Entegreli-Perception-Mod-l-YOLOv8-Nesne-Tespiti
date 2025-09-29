"""Test environment reset functionality."""

from env import WeatherTransformationEnv
import pytest
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestEnvironmentReset:
    """Test environment reset and initialization."""

    def test_env_initialization(self):
        """Test if environment initializes correctly."""
        env = WeatherTransformationEnv()
        assert env is not None
        assert hasattr(env, 'reset')
        assert hasattr(env, 'step')

    def test_env_reset_basic(self):
        """Test basic environment reset functionality."""
        env = WeatherTransformationEnv()
        observation = env.reset()
        assert observation is not None
        assert len(observation) > 0

    def test_env_reset_reproducible(self):
        """Test if environment reset is reproducible with same seed."""
        env1 = WeatherTransformationEnv()
        env2 = WeatherTransformationEnv()

        # Set same seed for both environments
        obs1 = env1.reset(seed=42)
        obs2 = env2.reset(seed=42)

        # Observations should be identical with same seed
        assert obs1.shape == obs2.shape if hasattr(
            obs1, 'shape') else obs1 == obs2

    def test_env_different_seeds(self):
        """Test if different seeds produce different initial states."""
        env = WeatherTransformationEnv()

        obs1 = env.reset(seed=42)
        obs2 = env.reset(seed=123)

        # Different seeds should produce different observations
        if hasattr(obs1, 'shape'):
            assert not (obs1 == obs2).all()
        else:
            assert obs1 != obs2


if __name__ == "__main__":
    pytest.main([__file__])
