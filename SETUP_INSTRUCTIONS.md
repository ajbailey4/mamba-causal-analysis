# Setup Instructions for Mamba Causal Analysis

## Quick Start (Copy and Paste)

### Step 1: Create and Activate Conda Environment

Open your VSCode terminal and run:

```bash
# Navigate to project directory
cd /Users/ajbailey4@ad.wisc.edu/ssms

# Create conda environment
conda create -n mamba-causal python=3.10 -y

# Activate it
conda activate mamba-causal
```

### Step 2: Install PyTorch

Choose one based on your GPU:

**If you have an NVIDIA GPU (CUDA 11.8):**
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

**If you have an NVIDIA GPU (CUDA 12.1):**
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

**If you only have CPU:**
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

### Step 3: Install Other Dependencies

```bash
pip install mamba-ssm transformers datasets matplotlib numpy jupyter ipykernel tqdm
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

5. **[README_MAMBA_CAUSAL.md](README_MAMBA_CAUSAL.md)**
   - Full project documentation
   - Usage examples
   - Troubleshooting guide

## Troubleshooting

### Problem: `conda: command not found`

**Solution**: Conda is not in your PATH. Either:
- Restart your terminal
- Run: `source ~/.bashrc` or `source ~/.zshrc`
- Or use the full path to conda

### Problem: `pip install mamba-ssm` fails

**Solutions**:
1. Make sure CUDA toolkit is installed:
   ```bash
   conda install cudatoolkit=11.8 -c conda-forge
   ```

2. Try installing without building from source:
   ```bash
   pip install mamba-ssm --no-build-isolation
   ```

3. Check if you have compatible CUDA version:
   ```bash
   nvcc --version
   ```

### Problem: Test script fails with import errors

**Solution**: Make sure you're in the right directory and environment:
```bash
cd /Users/ajbailey4@ad.wisc.edu/ssms
conda activate mamba-causal
python test_phase1.py
```

### Problem: CUDA out of memory

**Solution**: Use smaller model or CPU:
```python
# In your code, use:
mt = load_mamba_model("state-spaces/mamba-130m", device="cpu")
```

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
conda activate mamba-causal
```

### Check if environment is active:
```bash
# Your prompt should show: (mamba-causal)
# Or check with:
conda env list
```

### Deactivate environment when done:
```bash
conda deactivate
```

### Re-run tests anytime:
```bash
python test_phase1.py
```

## File Locations

All code is in: `/Users/ajbailey4@ad.wisc.edu/ssms/`

- Implementation plan: `.claude/plans/woolly-stargazing-ocean.md`
- This setup guide: `SETUP_INSTRUCTIONS.md`
- Full documentation: `README_MAMBA_CAUSAL.md`
- Test script: `test_phase1.py`

## Questions?

1. Check [README_MAMBA_CAUSAL.md](README_MAMBA_CAUSAL.md) for detailed documentation
2. Check the implementation plan in `.claude/plans/woolly-stargazing-ocean.md`
3. Look at ROME's original code in the `rome/` directory for reference
