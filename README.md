# Causal Mediation Analysis for Mamba State Space Models

Adapting the causal mediation analysis methodology from [ROME (Meng et al., 2022)](https://github.com/kmeng01/rome) to understand how Mamba State Space Models store and retrieve factual information.

[![Phase 1](https://img.shields.io/badge/Phase%201-Complete-brightgreen)]() [![Phase 2](https://img.shields.io/badge/Phase%202-In%20Progress-yellow)]()

## 🎯 Overview

This project extends causal tracing techniques from "Locating and Editing Factual Associations in GPT" to work with Mamba SSMs, aiming to understand:

- **Where** Mamba stores factual information (which layers?)
- **How** Mamba stores information (recurrent state vs. selection mechanism?)
- **Differences** between SSM and transformer memory mechanisms

## ⚡ Quick Start

```bash
# 1. Clone repository
git clone https://github.com/ajbailey4/mamba-causal-analysis.git
cd mamba-causal-analysis

# 2. Install UV (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Install dependencies
uv sync

# 4. Run tests
source .venv/bin/activate
python test_phase1.py
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed setup instructions.

## 📁 Project Structure

```
mamba_analysis/          # Core analysis code
├── mamba_models.py      # ✓ Model loading and layer identification
├── mamba_repr_tools.py  # TODO: Representation extraction
└── mamba_causal_trace.py # TODO: Causal tracing

util_ssm/               # SSM-specific utilities
├── ssm_nethook.py      # ✓ Hooking utilities
└── mamba_layernames.py # ✓ Layer naming

experiments_ssm/        # Experimental scripts
└── run_causal_trace.py # TODO: Main experiment runner

test_phase1.py          # ✓ Phase 1 verification
CONTRIBUTING.md         # Setup and development guide
IMPLEMENTATION_PLAN.md  # Technical roadmap
```

## 🎓 Usage

### Loading a Mamba Model

```python
from mamba_analysis.mamba_models import load_mamba_model

# Load model
mt = load_mamba_model("state-spaces/mamba-130m")
print(f"Loaded {mt.num_layers} layer model")

# Run inference
inputs = mt.tokenizer("The Eiffel Tower is in", return_tensors="pt").to(mt.device)
outputs = mt.model(**inputs)
```

### Hooking Layers

```python
from util_ssm import ssm_nethook
import torch

# Hook a specific layer
with ssm_nethook.trace_mamba_layer(mt.model, layer_num=5, component='mixer') as trace:
    with torch.no_grad():
        outputs = mt.model(**inputs)
    hidden_states = trace.output

print(f"Hidden states: {hidden_states.shape}")
```

### Multiple Layers

```python
# Hook multiple layers simultaneously
layer_specs = [(0, 'mixer'), (5, 'mixer'), (10, 'mixer')]

with ssm_nethook.trace_multiple_mamba_layers(mt.model, layer_specs) as traces:
    with torch.no_grad():
        outputs = mt.model(**inputs)

    for layer_num, component in layer_specs:
        layer_name = ssm_nethook.mamba_layername(mt.model, layer_num, component)
        hidden = traces[layer_name].output
        print(f"Layer {layer_num}: {hidden.shape}")
```

## 📊 Current Status

### ✅ Phase 1: Infrastructure (Complete)

- [x] Mamba model loading and inspection
- [x] Layer hooking and state extraction
- [x] Architecture parsing utilities
- [x] Comprehensive test suite

### 🚧 Phase 2: Basic Causal Tracing (In Progress)

- [ ] Token representation extraction
- [ ] Causal tracing algorithm for SSMs
- [ ] Experimental scripts and heatmap generation

### 🔮 Future Phases

- **Phase 3:** SSM-specific state tracing (internal states h_t, selection parameters B/C/Δt)
- **Phase 4:** Analysis and visualization tools
- **Phase 5:** Scale to larger models and diverse tasks

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the complete roadmap.

## 🔧 Available Models

| Model | Parameters | Use Case | HuggingFace ID |
|-------|-----------|----------|----------------|
| Mamba-130m | 130M | Development | `state-spaces/mamba-130m` |
| Mamba-370m | 370M | Testing | `state-spaces/mamba-370m` |
| Mamba-790m | 790M | Experiments | `state-spaces/mamba-790m` |
| Mamba-1.4b | 1.4B | Production | `state-spaces/mamba-1.4b` |

## 🧪 Testing

```bash
# Run test suite
python test_phase1.py

# Or without activating environment
uv run python test_phase1.py
```

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup
- Git workflow
- Coding guidelines
- How to submit changes

**Current priorities:**
1. Implement Phase 2 (basic causal tracing)
2. Test on diverse factual prompts
3. Compare with ROME's GPT-2 results

## 📖 References

### Papers

- **ROME**: Meng et al., "Locating and Editing Factual Associations in GPT" (NeurIPS 2022)
  [[Paper]](https://arxiv.org/abs/2202.05262) [[Code]](https://github.com/kmeng01/rome)

- **Mamba**: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023)
  [[Paper]](https://arxiv.org/abs/2312.00752) [[Code]](https://github.com/state-spaces/mamba)

### Attribution

This project includes `nethook.py` from the ROME repository:
- Original: https://github.com/kmeng01/rome
- License: MIT
- See `util_ssm/nethook.py` for attribution

## 📄 License

MIT License - see LICENSE file for details.

This project builds upon ROME. Please respect the original ROME license.

## 💬 Support

- **Issues**: Open an issue on GitHub
- **Documentation**: See [CONTRIBUTING.md](CONTRIBUTING.md) and [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)
- **ROME**: https://github.com/kmeng01/rome

---

**Status:** Phase 1 Complete ✓ | Phase 2 In Progress 🚧
**Package Manager:** UV (10-100x faster than conda/pip)
**Last Updated:** 2026-01-20
