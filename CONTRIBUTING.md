# Contributing to Room Relighting AI

Thank you for your interest in contributing to Room Relighting AI! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/room-relighting-ai.git
   cd room-relighting-ai
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🛠️ Development Setup

### Prerequisites
- Python 3.8+
- NVIDIA GPU with 8GB+ VRAM (for testing)
- CUDA toolkit installed

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8 pre-commit
```

## 📝 How to Contribute

### Reporting Bugs
- Use the GitHub issue tracker
- Include detailed reproduction steps
- Provide system information (GPU, Python version, etc.)
- Include error logs and screenshots

### Suggesting Features
- Open a GitHub issue with the "enhancement" label
- Describe the feature and its use case
- Discuss implementation approach

### Code Contributions

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**:
   ```bash
   python -m pytest tests/
   python app.py  # Test web interface
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add: your feature description"
   ```

5. **Push and create a pull request**:
   ```bash
   git push origin feature/your-feature-name
   ```

## 🎯 Areas for Contribution

### High Priority
- **Performance optimizations** for faster processing
- **New lighting conditions** and artistic styles
- **Better error handling** and user feedback
- **Documentation improvements** and tutorials

### Medium Priority
- **Additional output formats** (video, different image formats)
- **Batch processing UI** improvements
- **Cloud deployment** guides and scripts
- **Mobile/web optimization**

### Low Priority
- **Additional model support** (other diffusion models)
- **Plugin system** for custom lighting effects
- **Advanced configuration** options

## 📋 Code Style Guidelines

### Python Code Style
- Follow PEP 8 style guide
- Use Black for code formatting: `black .`
- Use meaningful variable and function names
- Add docstrings for all functions and classes

### Example Function Documentation
```python
def process_room_image(image_path: str, lighting_condition: str) -> Image:
    """
    Process a room image with specified lighting condition.
    
    Args:
        image_path: Path to the input room image
        lighting_condition: Lighting condition to apply
        
    Returns:
        PIL Image with applied lighting
        
    Raises:
        FileNotFoundError: If image_path doesn't exist
        ValueError: If lighting_condition is invalid
    """
    pass
```

### Commit Message Format
```
Type: Brief description

Detailed explanation if needed

- Use present tense ("Add feature" not "Added feature")
- Types: Add, Fix, Update, Remove, Refactor, Docs, Test
- Keep first line under 50 characters
- Reference issues: "Fix: memory leak in batch processing (#123)"
```

## 🧪 Testing

### Running Tests
```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_relighting.py

# Run with coverage
python -m pytest --cov=room_relighting
```

### Writing Tests
- Add tests for new features in the `tests/` directory
- Use descriptive test names
- Test both success and failure cases
- Mock external dependencies (model loading, file I/O)

## 📚 Documentation

### README Updates
- Keep the main README.md up to date
- Add new features to the feature list
- Update usage examples for new functionality

### Code Documentation
- Add docstrings to all public functions
- Include type hints where possible
- Comment complex algorithms and business logic

## 🔍 Review Process

### Pull Request Guidelines
- **Clear description** of changes and motivation
- **Link to related issues** if applicable
- **Screenshots/GIFs** for UI changes
- **Performance impact** notes for optimization changes

### Review Criteria
- Code quality and style compliance
- Test coverage for new features
- Documentation updates
- Performance considerations
- Backward compatibility

## 🌟 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes for significant contributions
- GitHub contributors page

## 📞 Getting Help

- **GitHub Discussions** for general questions
- **GitHub Issues** for bug reports and feature requests
- **Code review comments** for implementation questions

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for helping make Room Relighting AI better! 🏠✨