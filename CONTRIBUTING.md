# Contributing to Weather-Adaptive Autonomous Navigation System

Thank you for your interest in contributing to this project! This document provides guidelines for contributing to the development of our weather-adaptive autonomous navigation system.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Contribution Process](#contribution-process)
5. [Coding Standards](#coding-standards)
6. [Testing Guidelines](#testing-guidelines)
7. [Documentation](#documentation)
8. [Hardware Requirements](#hardware-requirements)

## Code of Conduct

By participating in this project, you agree to abide by our code of conduct:

- **Be respectful**: Treat all contributors with respect and kindness
- **Be collaborative**: Work together to improve the project
- **Be constructive**: Provide helpful feedback and suggestions
- **Be patient**: Understand that reviews take time and effort

## Getting Started

### Prerequisites

- Python 3.10+
- ROS 2 Humble
- CUDA-compatible GPU (GTX 1650 or higher recommended)
- Docker (optional but recommended)

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd staj-2-
   ```

2. **Set up development environment**
   ```bash
   # Windows
   .\setup_dev_env.ps1
   
   # Linux/macOS
   ./setup_dev_env.sh
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run tests**
   ```bash
   pytest tests/
   ```

## Development Setup

### Virtual Environment

Always use a virtual environment for development:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows
```

### IDE Configuration

We recommend using VS Code with the following extensions:
- Python
- ROS 2
- Docker
- GitLens

Configuration files are included in `.vscode/` directory.

## Contribution Process

### 1. Issue Creation

Before starting work, create or find an existing issue:

- **Bug Reports**: Use the bug report template
- **Feature Requests**: Use the feature request template
- **Documentation**: Use the documentation template

### 2. Fork and Branch

1. Fork the repository
2. Create a feature branch from `develop`:
   ```bash
   git checkout -b feature/your-feature-name develop
   ```

### 3. Development

- Follow the coding standards below
- Write tests for new functionality
- Update documentation as needed
- Keep commits atomic and descriptive

### 4. Testing

Before submitting a pull request:

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/ -m "not slow"  # Skip slow tests
pytest tests/ -m "gpu"       # Run only GPU tests

# Check code quality
flake8 .
black --check .
isort --check-only .

# Test ROS 2 components
colcon build --packages-select navigation perception
colcon test --packages-select navigation perception
```

### 5. Pull Request

1. Push your branch to your fork
2. Create a pull request to `develop` branch
3. Fill out the PR template completely
4. Wait for code review and address feedback

## Coding Standards

### Python Style

We follow PEP 8 with some modifications:

- Line length: 127 characters (to match GitHub's width)
- Use `black` for formatting
- Use `isort` for import sorting
- Use `flake8` for linting

### Code Organization

```
project/
├── gridformer.py           # GridFormer model implementation
├── env.py                  # Environment and RL components
├── perception/             # ROS 2 perception nodes
├── navigation/             # ROS 2 navigation nodes
├── tests/                  # All test files
├── data/                   # Data generation and configuration
└── docs/                   # Documentation
```

### Naming Conventions

- **Classes**: PascalCase (`WeatherDataGenerator`)
- **Functions/Methods**: snake_case (`add_fog_effect`)
- **Variables**: snake_case (`weather_conditions`)
- **Constants**: UPPER_SNAKE_CASE (`MAX_INTENSITY`)
- **Files**: snake_case (`test_gridformer.py`)

### Documentation

All code must be properly documented:

```python
def add_fog_effect(self, image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    """Add fog effect to image.
    
    Args:
        image: Input image as numpy array
        intensity: Fog intensity (0.0 to 1.0)
        
    Returns:
        Image with fog effect applied
        
    Raises:
        ValueError: If intensity is outside valid range
    """
```

## Testing Guidelines

### Test Structure

- **Unit Tests**: Test individual functions/methods
- **Integration Tests**: Test component interactions
- **System Tests**: Test end-to-end functionality

### Test Organization

```
tests/
├── __init__.py
├── conftest.py                 # Pytest configuration
├── test_env_reset.py          # Environment tests
├── test_gridformer_infer.py   # GridFormer tests
├── integration/               # Integration tests
└── fixtures/                  # Test data and fixtures
```

### Writing Tests

```python
class TestWeatherEffects:
    """Test weather effect generation."""
    
    @pytest.fixture
    def sample_image(self):
        return np.random.uint8(0, 255, (480, 640, 3))
    
    def test_fog_effect_intensity(self, sample_image):
        """Test fog effect with different intensities."""
        generator = WeatherDataGenerator()
        
        # Test low intensity
        result_low = generator.add_fog_effect(sample_image, 0.2)
        assert result_low.shape == sample_image.shape
        
        # Test high intensity
        result_high = generator.add_fog_effect(sample_image, 0.8)
        assert result_high.shape == sample_image.shape
        
        # High intensity should be more different from original
        diff_low = np.mean(np.abs(sample_image - result_low))
        diff_high = np.mean(np.abs(sample_image - result_high))
        assert diff_high > diff_low
```

### Performance Testing

For GPU-constrained testing (GTX 1650):

```python
@pytest.mark.gpu
def test_memory_usage(device):
    """Test memory usage doesn't exceed GTX 1650 limits."""
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        # Run your test
        # ...
        
        memory_used = torch.cuda.memory_allocated() - initial_memory
        max_memory_mb = 1200 * 1024 * 1024  # 1.2GB limit
        assert memory_used < max_memory_mb
```

## Documentation

### Code Documentation

- All public functions/classes must have docstrings
- Use Google-style docstrings
- Include type hints where possible

### README Updates

Update relevant README sections when:
- Adding new features
- Changing installation process
- Modifying configuration

### CHANGELOG

Maintain `CHANGELOG.md` with:
- Added features
- Fixed bugs
- Changed behavior
- Deprecated functionality

## Hardware Requirements

### Minimum Requirements

- **GPU**: GTX 1650 (4GB VRAM) or equivalent
- **RAM**: 8GB
- **Storage**: 10GB free space
- **OS**: Ubuntu 20.04+ or Windows 10+

### Recommended Requirements

- **GPU**: RTX 3060 (8GB VRAM) or higher
- **RAM**: 16GB
- **Storage**: 50GB free space (for datasets)

### Testing on Different Hardware

If you have different hardware:

1. **CPU-only testing**: Use `--cpu-only` flag
2. **Limited VRAM**: Use reduced image sizes in `data.yaml`
3. **Different GPU**: Adjust batch sizes in configurations

## Questions and Support

- **Bugs**: Create an issue with the bug report template
- **Questions**: Use GitHub Discussions
- **Security**: Email security@project.domain (replace with actual email)

## Recognition

Contributors will be recognized in:
- `CONTRIBUTORS.md` file
- Release notes
- Project documentation

Thank you for contributing to making autonomous navigation safer and more reliable in adverse weather conditions! 