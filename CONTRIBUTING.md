# Contributing to VESLINT

We welcome contributions to the VESLINT project! This document provides guidelines for contributing to help ensure a smooth collaboration process.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Process](#contributing-process)
- [Code Standards](#code-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Issue Reporting](#issue-reporting)
- [Feature Requests](#feature-requests)

## 🤝 Code of Conduct

This project adheres to a code of conduct adapted from the [Contributor Covenant](https://www.contributor-covenant.org/). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of machine learning and maritime data
- Familiarity with AIS (Automatic Identification System) data is helpful

### First-Time Contributors

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/VESLINT/VESLINT.git
   cd VESLINT
   ```
3. **Set up the development environment** (see Development Setup below)
4. **Create a branch** for your contribution
5. **Make your changes** and test them
6. **Submit a pull request**

## 🛠️ Development Setup

### Local Development Environment

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install development dependencies**:
   ```bash
   make install-dev
   # or manually:
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

4. **Verify installation**:
   ```bash
   make smoke-test
   ```

### Using Docker for Development

```bash
# Build development environment
make docker-dev

# Run tests in Docker
docker-compose run maritime-detection pytest tests/
```

### Development Workflow

1. **Create a new branch**:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b bugfix/issue-number
   ```

2. **Make your changes**

3. **Run tests and checks**:
   ```bash
   make dev-test  # Runs format, lint, and test
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Descriptive commit message"
   ```

5. **Push and create pull request**

## 📝 Contributing Process

### Pull Request Process

1. **Update documentation** if you're changing functionality
2. **Add tests** for new features or bug fixes
3. **Ensure all tests pass** and code follows style guidelines
4. **Update the changelog** if applicable
5. **Request review** from maintainers

### Pull Request Guidelines

- **One feature per PR**: Keep changes focused and atomic
- **Clear description**: Explain what changes you made and why
- **Reference issues**: Link to related issues using `#issue-number`
- **Screenshots**: Include screenshots for UI changes
- **Breaking changes**: Clearly document any breaking changes

### Commit Message Guidelines

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(models): add new LSTM attention mechanism
fix(data): handle missing vessel type values
docs(readme): update installation instructions
test(features): add tests for Kalman filter
```

## 🎯 Code Standards

### Code Style

We use the following tools to maintain code quality:

- **Black** for code formatting (line length: 100)
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking
- **pre-commit** for automated checks

### Code Quality Checklist

- [ ] Code follows PEP 8 style guidelines
- [ ] Functions and classes have docstrings
- [ ] Type hints are used where appropriate
- [ ] Error handling is implemented properly
- [ ] Code is DRY (Don't Repeat Yourself)
- [ ] Magic numbers are avoided (use constants)
- [ ] Security best practices are followed

### Python Code Guidelines

```python
def extract_vessel_features(trajectory: pd.DataFrame) -> Dict[str, float]:
    """
    Extract features from vessel trajectory data.
    
    Args:
        trajectory: DataFrame with vessel trajectory points
        
    Returns:
        Dictionary containing extracted features
        
    Raises:
        ValueError: If trajectory is empty or invalid
    """
    if trajectory.empty:
        raise ValueError("Trajectory cannot be empty")
    
    # Implementation here
    return features
```

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── test_config.py          # Configuration tests
├── test_data.py            # Data loading/preprocessing tests
├── test_features.py        # Feature extraction tests
├── test_models.py          # Model tests
├── test_utils.py           # Utility function tests
└── test_pipeline.py        # Integration tests
```

### Writing Tests

1. **Use descriptive test names**:
   ```python
   def test_kalman_filter_smooths_noisy_trajectory():
       """Test that Kalman filter reduces noise in trajectory data."""
   ```

2. **Follow AAA pattern** (Arrange, Act, Assert):
   ```python
   def test_feature_extraction_returns_correct_shape():
       # Arrange
       sample_data = create_sample_trajectory()
       extractor = FeatureExtractor()
       
       # Act
       features = extractor.extract_features(sample_data)
       
       # Assert
       assert features.shape == (1, 25)  # Expected feature count
   ```

3. **Use fixtures for common test data**:
   ```python
   @pytest.fixture
   def sample_ais_data():
       return pd.DataFrame({
           'MMSI': [123456789, 123456789],
           'LAT': [34.05, 34.06],
           'LON': [-118.25, -118.24],
           # ... more columns
       })
   ```

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_data.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run performance tests
pytest tests/ -m "performance"
```

### Test Categories

Mark tests with appropriate categories:

```python
import pytest

@pytest.mark.unit
def test_data_loader_initialization():
    """Unit test for data loader."""
    pass

@pytest.mark.integration  
def test_complete_pipeline():
    """Integration test for full pipeline."""
    pass

@pytest.mark.performance
def test_model_training_time():
    """Performance test for model training."""
    pass
```

## 📚 Documentation

### Documentation Standards

- **Docstrings**: Use Google style docstrings for all public functions and classes
- **Type hints**: Include type hints for function parameters and return values
- **Comments**: Use comments sparingly to explain complex logic
- **README updates**: Update README.md when adding new features
- **Examples**: Include usage examples in docstrings

### Documentation Structure

```
docs/
├── api/                    # API documentation
├── tutorials/              # Step-by-step tutorials
├── examples/               # Code examples
├── deployment/             # Deployment guides
└── contributing.md         # This file
```

### Building Documentation

```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme myst-parser

# Build documentation
cd docs && make html
```

## 🐛 Issue Reporting

### Before Reporting an Issue

1. **Search existing issues** to avoid duplicates
2. **Check the documentation** for solutions
3. **Test with the latest version**
4. **Prepare a minimal reproduction case**

### Issue Template

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Load data with '...'
2. Run pipeline with '....'
3. See error

**Expected behavior**
What you expected to happen.

**Environment:**
- OS: [e.g. Ubuntu 20.04]
- Python version: [e.g. 3.9]
- Package version: [e.g. 1.0.0]

**Additional context**
Add any other context about the problem here.
```

### Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements or additions to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed
- `question`: Further information is requested

## 💡 Feature Requests

### Proposing New Features

1. **Check existing proposals** in issues and discussions
2. **Describe the use case** clearly
3. **Provide implementation ideas** if possible
4. **Consider alternatives** and explain why your approach is better

### Feature Request Template

```markdown
**Is your feature request related to a problem?**
A clear description of what the problem is.

**Describe the solution you'd like**
A clear description of what you want to happen.

**Describe alternatives you've considered**
Other solutions you've considered.

**Additional context**
Any other context about the feature request.
```

## 🏗️ Types of Contributions

### Areas Where We Need Help

- **Data preprocessing improvements**: Better handling of edge cases
- **Model performance**: New algorithms or hyperparameter optimization
- **Visualization enhancements**: Better plots and dashboards
- **Documentation**: Tutorials, examples, and API documentation
- **Testing**: More comprehensive test coverage
- **Performance optimization**: Speed and memory improvements
- **Deployment tools**: Docker, Kubernetes, cloud deployment guides

### Contribution Ideas

#### Beginner-Friendly
- Fix typos in documentation
- Add more test cases
- Improve error messages
- Add configuration validation

#### Intermediate
- Implement new feature extraction methods
- Add support for new data formats
- Improve model evaluation metrics
- Create visualization components

#### Advanced
- Implement new machine learning models
- Add distributed training support
- Create real-time processing capabilities
- Develop advanced optimization algorithms

## 🔍 Code Review Process

### What We Look For

- **Correctness**: Does the code work as intended?
- **Performance**: Is the code efficient?
- **Maintainability**: Is the code easy to understand and modify?
- **Security**: Are there any security vulnerabilities?
- **Style**: Does the code follow our style guidelines?

### Review Timeline

- **Initial response**: Within 48 hours
- **Full review**: Within 1 week
- **Follow-up**: As needed based on changes

### Addressing Feedback

- **Be responsive**: Address comments promptly
- **Ask questions**: If feedback is unclear, ask for clarification
- **Make requested changes**: Update your PR based on feedback
- **Test thoroughly**: Ensure changes don't break existing functionality

## 🏆 Recognition

### Contributors

We recognize contributors in several ways:

- **CONTRIBUTORS.md**: Listed in the contributors file
- **Release notes**: Mentioned in release notes
- **GitHub**: Contributor statistics and badges

### Becoming a Maintainer

Active contributors may be invited to become maintainers. Criteria include:

- **Consistent contributions** over time
- **Code quality** and adherence to guidelines
- **Community engagement** and helpfulness
- **Domain expertise** in relevant areas

## 📞 Getting Help

### Where to Get Help

- **GitHub Discussions**: For general questions and discussions
- **GitHub Issues**: For bug reports and feature requests
- **Documentation**: Check the docs first
- **Stack Overflow**: Tag with `VESLINT`

### Mentorship

New contributors can request mentorship for:

- Understanding the codebase
- Learning best practices
- Guidance on contribution ideas
- Code review feedback

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Thank you for contributing to VESLINT! Your efforts help make maritime traffic analysis more accessible and effective. 🚢⚓