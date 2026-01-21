# Quick Start Guide - Mamba Causal Analysis

## One-Page Reference

### Setup (First Time)

```bash
# 1. Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone or navigate to project
cd /Users/ajbailey4@ad.wisc.edu/ssms

# 3. Install dependencies (creates .venv and installs everything)
uv sync

# 4. Activate and test
source .venv/bin/activate
python test_phase1.py
```

### Daily Workflow

```bash
# Activate environment
source .venv/bin/activate

# Work on code...

# Run tests
python test_phase1.py

# Deactivate when done
deactivate
```

### Alternative: Use UV Run (No Activation Needed)

```bash
# Run commands without activating
uv run python test_phase1.py
uv run jupyter notebook
```

### Git Workflow

```bash
# Initial setup
git init
git add .
git commit -m "Initial commit"
git remote add origin YOUR_REPO_URL
git push -u origin main

# Daily usage
git pull                          # Get latest
# ... make changes ...
git add .
git commit -m "Description"
git push                          # Upload changes

# On another computer
git clone https://github.com/ajbailey4/mamba-causal-analysis.git
cd mamba-causal-analysis
uv sync                          # Install dependencies
source .venv/bin/activate
python test_phase1.py
```

### Basic Usage

```python
# Load model
from mamba_analysis.mamba_models import load_mamba_model
mt = load_mamba_model("state-spaces/mamba-130m")

# Prepare input
inputs = mt.tokenizer("The Eiffel Tower is in", return_tensors="pt").to(mt.device)

# Run model
import torch
with torch.no_grad():
    outputs = mt.model(**inputs)

# Hook a layer
from util_ssm import ssm_nethook
with ssm_nethook.trace_mamba_layer(mt.model, 5, 'mixer') as trace:
    with torch.no_grad():
        outputs = mt.model(**inputs)
    hidden = trace.output
```

### File Organization

```
ssms/
├── mamba_analysis/      # Core code (Phase 1 ✓)
├── util_ssm/           # Utilities (Phase 1 ✓)
├── experiments_ssm/    # Experiments (TODO)
├── configs/            # Configs
├── notebooks/          # Notebooks
├── test_phase1.py      # Tests
├── pyproject.toml      # Dependencies (UV)
├── .python-version     # Python version
└── *.md               # Documentation
```

### Documentation Quick Links

| File | Purpose |
|------|---------|
| [README.md](README.md) | Project overview |
| [SETUP_INSTRUCTIONS.md](SETUP_INSTRUCTIONS.md) | Detailed setup |
| [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) | Full roadmap |
| [README_MAMBA_CAUSAL.md](README_MAMBA_CAUSAL.md) | API docs |
| [GIT_SETUP.md](GIT_SETUP.md) | Git workflow |

### Troubleshooting

| Problem | Solution |
|---------|----------|
| `uv: command not found` | Run: `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| `uv sync` fails | Check CUDA version, edit `pyproject.toml` PyTorch index |
| Import errors | `source .venv/bin/activate` or use `uv run` |
| CUDA out of memory | Use `device="cpu"` or smaller model |

### UV Commands Cheat Sheet

```bash
# Package management
uv sync                           # Install all dependencies
uv add package-name               # Add new package
uv remove package-name            # Remove package
uv lock --upgrade                 # Update lock file

# Run commands
uv run python script.py           # Run without activating
uv run jupyter notebook           # Start jupyter

# Environment info
uv pip list                       # List installed packages
which python                      # Check python location
```

### PyTorch/CUDA Configuration

Edit `pyproject.toml` to change CUDA version:

```toml
# For CUDA 11.8 (default)
[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu118"

# For CUDA 12.1
url = "https://download.pytorch.org/whl/cu121"

# For CPU-only
url = "https://download.pytorch.org/whl/cpu"

# For ROCm (AMD)
url = "https://download.pytorch.org/whl/rocm5.7"
```

Then run: `uv sync --reinstall-package torch`

### Next Steps

- [ ] Setup complete? Run `python test_phase1.py`
- [ ] Git initialized? See [GIT_SETUP.md](GIT_SETUP.md)
- [ ] Ready for Phase 2? See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)

### Essential Commands

```bash
# Environment
source .venv/bin/activate         # Activate
deactivate                        # Deactivate
which python                      # Check location

# UV shortcuts
uv sync                           # Sync dependencies
uv run CMD                        # Run without activating
uv add PKG                        # Add package

# Testing
python test_phase1.py             # Run tests
uv run python test_phase1.py     # Run without activating

# Git
git status                        # Check status
git add . && git commit -m "msg" # Save changes
git push / git pull              # Sync

# Models
# mamba-130m  - Development (fast, small)
# mamba-370m  - Testing
# mamba-790m  - Experiments
# mamba-1.4b  - Production (slow, large)
```

### Why UV?

| Feature | Benefit |
|---------|---------|
| **Speed** | 10-100x faster than pip/conda |
| **Simple** | One command: `uv sync` |
| **Standard** | Uses `pyproject.toml` (PEP standard) |
| **Reliable** | Lock file ensures reproducibility |
| **Modern** | Rust-based, actively developed |

### Contact

Questions? Check:
1. Documentation in `*.md` files
2. UV docs: https://docs.astral.sh/uv/
3. ROME repo: https://github.com/kmeng01/rome
4. Mamba repo: https://github.com/state-spaces/mamba

---

**Current Status:** Phase 1 Complete ✓

**Package Manager:** UV (fast Python package manager)

**Last Updated:** 2026-01-20
