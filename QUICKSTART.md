# Quick Start Guide - Mamba Causal Analysis

## One-Page Reference

### Setup (First Time)

```bash
# 1. Clone or navigate to project
cd /Users/ajbailey4@ad.wisc.edu/ssms

# 2. Create environment (choose one)
conda env create -f environment.yml                    # Recommended
# OR
conda create -n mamba-causal python=3.10 -y && \
conda activate mamba-causal && \
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y && \
pip install -r requirements.txt

# 3. Activate and test
conda activate mamba-causal
python test_phase1.py
```

### Daily Workflow

```bash
# Activate environment
conda activate mamba-causal

# Work on code...

# Run tests
python test_phase1.py
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
git clone --recursive YOUR_REPO_URL
cd mamba-causal-analysis
conda env create -f environment.yml
conda activate mamba-causal
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
| `mamba-ssm` install fails | `conda install cudatoolkit=11.8 -c conda-forge` |
| CUDA out of memory | Use `device="cpu"` or `torch_dtype=torch.float16` |
| Import errors | `conda activate mamba-causal` and `cd` to project root |
| Tests fail | Check environment: `conda list \| grep torch` |

### Next Steps

- [ ] Setup complete? Run `python test_phase1.py`
- [ ] Git initialized? See [GIT_SETUP.md](GIT_SETUP.md)
- [ ] Ready for Phase 2? See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)

### Essential Commands

```bash
# Environment
conda activate mamba-causal        # Activate
conda deactivate                   # Deactivate
conda env list                     # List envs
conda list                         # List packages

# Testing
python test_phase1.py              # Run tests

# Git
git status                         # Check status
git add . && git commit -m "msg"  # Save changes
git push / git pull               # Sync

# Models
# mamba-130m  - Development
# mamba-370m  - Testing
# mamba-790m  - Experiments
# mamba-1.4b  - Production
```

### Contact

Questions? Check:
1. Documentation in `*.md` files
2. ROME repo: https://github.com/kmeng01/rome
3. Mamba repo: https://github.com/state-spaces/mamba

---

**Current Status:** Phase 1 Complete ✓

**Last Updated:** 2026-01-20
