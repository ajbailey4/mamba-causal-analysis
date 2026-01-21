# Setup Instructions for Mamba Causal Analysis

## Quick Start (Copy and Paste)

### Step 1: Install UV

UV is a fast Python package manager. Install it if you don't have it:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv

# Or with homebrew (macOS)
brew install uv
```

### Step 2: Navigate to Project and Install

```bash
# Navigate to project directory
cd /Users/ajbailey4@ad.wisc.edu/ssms

# Create virtual environment and install dependencies (one command!)
uv sync

# This will:
# - Create a .venv directory with Python 3.10
# - Install all dependencies including PyTorch with CUDA support
# - Be ready in seconds instead of minutes!
```

### Step 3: Activate Environment

```bash
# Activate the virtual environment
source .venv/bin/activate

# On Windows (if applicable)
# .venv\Scripts\activate
```

### Step 4: Verify Installation

```bash
python test_phase1.py
```

You should see output like:
```
==================================================================
PHASE 1 TEST: Mamba Infrastructure Setup
==================================================================

Test 1: Importing modules...
✓ All modules imported successfully

Test 2: Loading Mamba-130m...
✓ Model loaded successfully
  Device: cuda
  Layers: 24
  Model: state-spaces/mamba-130m

Test 3: Inspecting layer structure...
✓ Layer structure identified

... (more tests)

==================================================================
ALL TESTS PASSED! ✓
==================================================================
```

## Alternative: Using Different PyTorch/CUDA Versions

### For CUDA 12.1

Edit `pyproject.toml` and change the PyTorch index URL:

```toml
[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu121"  # Changed from cu118
explicit = true
```

Then run:
```bash
uv sync --reinstall-package torch
```

### For CPU-Only

Edit `pyproject.toml` and change the PyTorch index URL:

```toml
[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

Then run:
```bash
uv sync --reinstall-package torch
```

### For ROCm (AMD GPUs)

Edit `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/rocm5.7"
explicit = true
```

Then run:
```bash
uv sync --reinstall-package torch
```

## What Was Already Created For You

The following files and directories are already set up:

### ✓ Directory Structure
```
mamba_analysis/         # Core Mamba analysis code
util_ssm/              # SSM-specific utilities
experiments_ssm/       # Experimental scripts
configs/mamba/         # Configuration files
notebooks/             # Jupyter notebooks
```

### ✓ Implemented Files

1. **[mamba_analysis/mamba_models.py](mamba_analysis/mamba_models.py)**
   - `MambaModelAndTokenizer` class for loading Mamba models
   - `load_mamba_model()` convenience function
   - Layer identification and structure parsing

2. **[util_ssm/ssm_nethook.py](util_ssm/ssm_nethook.py)**
   - `mamba_layername()` - Get layer names for Mamba
   - `trace_mamba_layer()` - Hook single layer
   - `trace_multiple_mamba_layers()` - Hook multiple layers
   - Wraps ROME's nethook utilities for SSM use

3. **[util_ssm/mamba_layernames.py](util_ssm/mamba_layernames.py)**
   - `parse_mamba_architecture()` - Analyze model structure
   - `get_layer_components()` - List available components in a layer
   - Helper functions for layer name parsing

4. **[test_phase1.py](test_phase1.py)**
   - Complete test suite for Phase 1
   - Verifies all infrastructure is working

5. **[README.md](README.md)**
   - Full project documentation
   - Usage examples
   - Troubleshooting guide

## UV Commands Cheat Sheet

### Essential Commands

```bash
# Install/sync dependencies
uv sync                           # Install all dependencies
uv sync --dev                     # Include dev dependencies (jupyter, etc.)

# Add new packages
uv add package-name               # Add to dependencies
uv add --dev package-name         # Add to dev dependencies

# Remove packages
uv remove package-name

# Update packages
uv lock --upgrade                 # Update lock file
uv sync                          # Install updated versions

# Run commands in the environment
uv run python test_phase1.py     # Run without activating
uv run jupyter notebook          # Start jupyter

# Show installed packages
uv pip list

# Clean cache
uv cache clean
```

### Working with the Environment

```bash
# Activate environment
source .venv/bin/activate         # Linux/macOS
# .venv\Scripts\activate          # Windows

# Once activated, use python normally
python test_phase1.py
jupyter notebook

# Deactivate
deactivate
```

## Troubleshooting

### Problem: `uv: command not found`

**Solution**: Install UV first:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Then restart your terminal or run:
source $HOME/.cargo/env
```

### Problem: `uv sync` fails with CUDA errors

**Solution**: Your CUDA version may not match. Check your CUDA version:
```bash
nvcc --version
```

Then update `pyproject.toml` with the correct PyTorch index (see "Alternative" section above).

### Problem: Test script fails with import errors

**Solution**: Make sure environment is activated:
```bash
source .venv/bin/activate
python test_phase1.py
```

Or run directly with UV:
```bash
uv run python test_phase1.py
```

### Problem: CUDA out of memory

**Solution**: Use smaller model or CPU:
```python
# In your code, use:
mt = load_mamba_model("state-spaces/mamba-130m", device="cpu")
```

### Problem: Slow package installation

**Solution**: UV is already fast, but you can:
```bash
# Use parallel downloads (default)
uv sync

# Clear cache and reinstall
uv cache clean
uv sync --reinstall
```

## Comparison: UV vs Conda

| Feature | UV | Conda |
|---------|----|----|
| Speed | ⚡ Very fast (Rust-based) | 🐌 Slower (Python-based) |
| Install time | ~10-30 seconds | ~5-10 minutes |
| Disk space | Smaller | Larger |
| Environment | `.venv` (standard) | Custom conda env |
| Configuration | `pyproject.toml` | `environment.yml` |
| Package resolution | Fast | Can be slow |

## Next Steps After Setup

Once `test_phase1.py` passes all tests, you're ready for Phase 2:

### Phase 2: Basic Causal Tracing

The following files still need to be implemented:

1. **mamba_analysis/mamba_repr_tools.py**
   - Extract hidden states at specific token positions
   - Similar to ROME's `repr_tools.py` but for Mamba

2. **mamba_analysis/mamba_causal_trace.py**
   - Core causal tracing algorithm
   - Adapt ROME's `trace_with_patch()` for SSMs

3. **experiments_ssm/run_causal_trace.py**
   - Main experimental script
   - Run causal tracing on factual prompts

Would you like me to implement these Phase 2 components next?

## Quick Reference

### Always activate the environment first:
```bash
source .venv/bin/activate

# Or run without activating:
uv run python script.py
```

### Check if environment is active:
```bash
which python
# Should show: /Users/ajbailey4@ad.wisc.edu/ssms/.venv/bin/python
```

### Deactivate environment when done:
```bash
deactivate
```

### Re-sync dependencies anytime:
```bash
uv sync
```

## File Locations

All code is in: `/Users/ajbailey4@ad.wisc.edu/ssms/`

- Implementation plan: `IMPLEMENTATION_PLAN.md`
- This setup guide: `SETUP_INSTRUCTIONS.md`
- Full documentation: `README.md`
- Test script: `test_phase1.py`
- Dependencies: `pyproject.toml`
- Python version: `.python-version`

## Questions?

1. Check [README.md](README.md) for detailed documentation
2. Check the implementation plan in `IMPLEMENTATION_PLAN.md`
3. UV documentation: https://docs.astral.sh/uv/

## Advantages of UV for This Project

1. **Fast**: Install all dependencies in seconds
2. **Deterministic**: Lock file ensures reproducible builds
3. **Modern**: Uses standard `pyproject.toml` format
4. **Simple**: One tool for everything (pip + virtualenv replacement)
5. **Cross-platform**: Works on macOS, Linux, Windows
6. **Compatible**: Integrates with existing Python tools
