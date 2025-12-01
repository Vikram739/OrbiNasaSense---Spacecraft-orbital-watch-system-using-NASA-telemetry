# Contributing to OrbiNasaSense

Thank you for your interest in contributing to OrbiNasaSense! 

## How to Contribute

### Reporting Bugs
- Use the GitHub Issues tab
- Describe the bug clearly
- Include steps to reproduce
- Mention your Python version and OS

### Suggesting Features
- Open an issue with the "enhancement" label
- Explain why this feature would be useful
- Provide examples if possible

### Pull Requests
1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Test thoroughly
5. Commit with clear messages (`git commit -m 'Add amazing feature'`)
6. Push to your branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Style
- Follow PEP 8 guidelines
- Add docstrings to functions
- Comment complex logic
- Keep functions focused and small

### Testing
- Test your changes with sample data
- Verify the Streamlit UI works correctly
- Check both training and inference workflows

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/OrbiNasaSense---Spacecraft-orbital-watch-system-using-NASA-telemetry.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NASA data (see README for Kaggle link)

# Make your changes
# ...

# Test
python train.py --channel P-1 --epochs 10
streamlit run app.py
```

## Questions?

Feel free to open an issue or reach out to [@Vikram739](https://github.com/Vikram739).

Happy contributing! 🚀
