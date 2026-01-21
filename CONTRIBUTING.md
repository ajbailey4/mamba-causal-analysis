# Contributing to Mamba Causal Analysis

Thank you for your interest in contributing! This guide covers setup, development workflow, and contribution guidelines.

## 📋 Table of Contents

- [Development Setup](#development-setup)
- [Git Workflow](#git-workflow)
- [Development Guidelines](#development-guidelines)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

## 🚀 Development Setup

### Prerequisites

- **UV** (recommended) or conda
- **Git**
- **NVIDIA GPU** with CUDA support (recommended) or CPU
- **Python 3.10+**

### Quick Setup

```bash
# 1. Clone repository
git clone https://github.com/ajbailey4/mamba-causal-analysis.git
cd mamba-causal-analysis

# 2. Install UV (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Install dependencies
uv sync

# 4. Activate environment
source .venv/bin/activate

# 5. Verify setup
python test_phase1.py
```

### Alternative: Using Conda

If you prefer conda over UV:

```bash
# Create environment
conda env create -f environment.yml
conda activate mamba-causal

# Verify
python test_phase1.py
```

## 🔧 UV Commands Reference

### Package Management

```bash
# Install/sync all dependencies
uv sync

# Add new package
uv add package-name

# Add development dependency
uv add --dev package-name

# Remove package
uv remove package-name

# Update dependencies
uv lock --upgrade
uv sync
```

### Running Commands

```bash
# With activated environment
source .venv/bin/activate
python script.py
jupyter notebook

# Without activating (using uv run)
uv run python script.py
uv run jupyter notebook

# Check environment
which python  # Should show .venv/bin/python
uv pip list
```

### PyTorch/CUDA Configuration

Edit `pyproject.toml` to change PyTorch version:

```toml
[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu118"  # CUDA 11.8 (default)
# url = "https://download.pytorch.org/whl/cu121"  # CUDA 12.1
# url = "https://download.pytorch.org/whl/cpu"    # CPU-only
# url = "https://download.pytorch.org/whl/rocm5.7"  # AMD ROCm
explicit = true
```

Then reinstall PyTorch:

```bash
uv sync --reinstall-package torch
```

## 🌿 Git Workflow

### Initial Setup

```bash
# Fork the repository on GitHub (if contributing to main project)
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/mamba-causal-analysis.git
cd mamba-causal-analysis

# Add upstream remote
git remote add upstream https://github.com/ajbailey4/mamba-causal-analysis.git
```

### Daily Workflow

```bash
# 1. Pull latest changes
git pull upstream main

# 2. Create a branch for your feature
git checkout -b feature-name

# 3. Make changes and commit
git add .
git commit -m "Descriptive commit message"

# 4. Push to your fork
git push origin feature-name

# 5. Create Pull Request on GitHub
```

### Commit Message Guidelines

Good commit messages:
```bash
git commit -m "Add mamba_repr_tools.py for token representation extraction"
git commit -m "Fix layer naming bug in ssm_nethook.py"
git commit -m "Update documentation for Phase 2"
```

Bad commit messages:
```bash
git commit -m "updates"
git commit -m "fix"
git commit -m "wip"
```

### Working Across Multiple Computers

**On Computer A:**
```bash
# Make changes
git add .
git commit -m "Implement causal tracing"
git push
```

**On Computer B:**
```bash
# Get latest changes
git pull

# Install/sync dependencies
uv sync

# Continue working
```

## 👨‍💻 Development Guidelines

### Code Style

- Follow PEP 8 for Python code
- Use type hints where appropriate
- Add docstrings to functions and classes
- Keep functions focused and modular

Example:
```python
def get_mamba_states_at_tokens(
    model: MambaLMHeadModel,
    tok: AutoTokenizer,
    contexts: List[str],
    token_idxs: List[int],
    layer: int
) -> torch.Tensor:
    """
    Extract hidden states from Mamba at specific token positions.

    Args:
        model: Mamba model
        tok: Tokenizer
        contexts: Input text strings
        token_idxs: Token positions to extract
        layer: Layer number

    Returns:
        Tensor of shape (batch_size, hidden_dim)
    """
    # Implementation...
```

### File Organization

- **mamba_analysis/**: Core analysis algorithms
- **util_ssm/**: Utility functions and helpers
- **experiments_ssm/**: Experimental scripts
- **configs/**: Configuration files
- **notebooks/**: Jupyter notebooks for exploration

### Adding Dependencies

```bash
# Add to project dependencies
uv add package-name

# Add to dev dependencies (testing, notebooks, etc.)
uv add --dev package-name

# This updates pyproject.toml automatically
```

## 🧪 Testing

### Running Tests

```bash
# Run test suite
python test_phase1.py

# Or without activating
uv run python test_phase1.py

# Run specific test (once we have pytest)
# pytest tests/test_mamba_models.py
```

### Writing Tests

When adding new functionality:

1. Add tests to verify it works
2. Run existing tests to ensure nothing broke
3. Document new features

### Test Coverage

- Test core functionality
- Test edge cases
- Test error handling
- Verify compatibility across different Mamba models

## 🐛 Troubleshooting

### Common Issues

#### 1. `uv: command not found`

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Reload shell
source $HOME/.cargo/env
```

#### 2. `uv sync` fails with CUDA errors

Check your CUDA version:
```bash
nvcc --version
```

Update `pyproject.toml` with correct PyTorch index (see Configuration section above).

#### 3. Import errors after `uv sync`

```bash
# Make sure environment is activated
source .venv/bin/activate

# Or use uv run
uv run python test_phase1.py
```

#### 4. CUDA out of memory

```python
# Use smaller model
mt = load_mamba_model("state-spaces/mamba-130m", device="cpu")

# Or use half precision
mt = load_mamba_model("state-spaces/mamba-130m", torch_dtype=torch.float16)
```

#### 5. `mamba-ssm` installation fails

```bash
# Ensure CUDA toolkit is installed
conda install cudatoolkit=11.8 -c conda-forge

# Then retry
uv sync
```

### Getting Help

- **Documentation**: Check [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)
- **Issues**: Search existing issues on GitHub
- **UV Docs**: https://docs.astral.sh/uv/
- **ROME Docs**: https://github.com/kmeng01/rome

## 📦 Project Dependencies

### Core Dependencies

- `torch` - PyTorch deep learning framework
- `mamba-ssm` - Mamba State Space Models
- `transformers` - HuggingFace transformers
- `numpy`, `scipy` - Scientific computing
- `matplotlib`, `seaborn` - Visualization

### Development Dependencies

- `jupyter` - Interactive notebooks
- `ipykernel` - Jupyter kernel
- `pytest` - Testing framework (future)

See `pyproject.toml` for complete list.

## 🎯 Contribution Priorities

### Phase 2 (Current)

1. **mamba_repr_tools.py**
   - Extract token representations at specific positions
   - Handle Mamba's recurrent state dependencies
   - Pattern similar to ROME's `repr_tools.py`

2. **mamba_causal_trace.py**
   - Implement `trace_with_patch_mamba()`
   - Adapt ROME's corruption/restoration algorithm
   - Account for SSM recurrent dependencies

3. **run_causal_trace.py**
   - Command-line interface for experiments
   - Generate heatmaps showing critical (layer, position) pairs
   - Save results for analysis

### Future Contributions

- Phase 3: SSM-specific state tracing
- Phase 4: Visualization tools
- Phase 5: Scaling to larger models
- Documentation improvements
- Additional test coverage

## 📝 Pull Request Process

1. **Fork** the repository
2. **Create** a feature branch
3. **Make** your changes
4. **Test** your changes thoroughly
5. **Commit** with clear messages
6. **Push** to your fork
7. **Create** a Pull Request with description

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Tests added/updated
- [ ] All tests passing
- [ ] Tested on [model/configuration]

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

## 🔄 Keeping Your Fork Updated

```bash
# Add upstream remote (once)
git remote add upstream https://github.com/ajbailey4/mamba-causal-analysis.git

# Fetch latest changes
git fetch upstream

# Merge into your main branch
git checkout main
git merge upstream/main

# Push to your fork
git push origin main
```

## 💡 Tips

### Speed Up Development

```bash
# Use uv run to avoid activation
uv run python script.py

# Use IPython for interactive development
uv run ipython

# Use notebooks for exploration
uv run jupyter notebook
```

### Useful Aliases

Add to your `.bashrc` or `.zshrc`:

```bash
alias mamba-env='source .venv/bin/activate'
alias mamba-test='uv run python test_phase1.py'
alias mamba-sync='uv sync'
```

## 📚 Additional Resources

- **UV Documentation**: https://docs.astral.sh/uv/
- **ROME Paper**: https://arxiv.org/abs/2202.05262
- **Mamba Paper**: https://arxiv.org/abs/2312.00752
- **PyTorch Docs**: https://pytorch.org/docs/
- **HuggingFace Docs**: https://huggingface.co/docs

## 🙏 Acknowledgments

This project builds upon:
- **ROME** by Meng et al. (model editing methodology)
- **Mamba** by Gu & Dao (state space models)
- Open source community contributions

---

**Questions?** Open an issue or check [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)

**Ready to contribute?** Follow the setup guide above and start coding!
