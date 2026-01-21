# Causal Mediation Analysis for Mamba State Space Models

Adapting the causal mediation analysis methodology from [ROME (Meng et al., 2022)](https://github.com/kmeng01/rome) to understand how Mamba State Space Models store and retrieve factual information.

## 🎯 Project Overview

This project extends the causal tracing techniques from "Locating and Editing Factual Associations in GPT" to work with Mamba SSMs. The goal is to understand:

- **Where** Mamba stores factual information (which layers?)
- **How** Mamba stores information (recurrent state vs. selection mechanism?)
- **Differences** between SSM and transformer memory mechanisms

## 🚀 Quick Start

### Prerequisites

- Conda or Miniconda installed
- NVIDIA GPU with CUDA support (recommended) or CPU
- Git

### Installation

1. **Clone this repository:**
   ```bash
   git clone <your-repo-url>
   cd ssms
   ```

2. **Create conda environment:**
   ```bash
   # Option 1: Using environment.yml (recommended)
   conda env create -f environment.yml

   # Option 2: Manual setup
   conda create -n mamba-causal python=3.10 -y
   conda activate mamba-causal
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
   pip install -r requirements.txt
   ```

3. **Activate environment:**
   ```bash
   conda activate mamba-causal
   ```

4. **Verify installation:**
   ```bash
   python test_phase1.py
   ```

   You should see:
   ```
   ==================================================================
   ALL TESTS PASSED! ✓
   ==================================================================
   ```

### Alternative: Quick Setup Script

```bash
# One-command setup (if conda is available)
conda env create -f environment.yml && conda activate mamba-causal && python test_phase1.py
```

## 📁 Project Structure

```
ssms/
├── mamba_analysis/          # Core Mamba analysis code
│   ├── mamba_models.py      # ✓ Model loading and layer identification
│   ├── mamba_repr_tools.py  # TODO: Representation extraction
│   ├── mamba_causal_trace.py # TODO: Causal tracing implementation
│   └── mamba_hparams.py     # TODO: Hyperparameters
├── util_ssm/                # SSM-specific utilities
│   ├── ssm_nethook.py       # ✓ Hooking utilities (wraps ROME's nethook)
│   └── mamba_layernames.py  # ✓ Layer naming conventions
├── experiments_ssm/         # Experimental scripts
│   ├── run_causal_trace.py  # TODO: Main experiment runner
│   └── visualize_flow.py    # TODO: Visualization tools
├── configs/mamba/           # Configuration files for different models
├── notebooks/               # Jupyter notebooks for analysis
├── rome/                    # Original ROME repository (git submodule)
├── test_phase1.py           # ✓ Phase 1 verification tests
├── IMPLEMENTATION_PLAN.md   # Detailed implementation roadmap
├── SETUP_INSTRUCTIONS.md    # Step-by-step setup guide
├── requirements.txt         # Python dependencies
├── environment.yml          # Conda environment specification
└── README.md               # This file
```

## 📚 Documentation

- **[SETUP_INSTRUCTIONS.md](SETUP_INSTRUCTIONS.md)** - Detailed setup guide with troubleshooting
- **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)** - Complete implementation roadmap
- **[README_MAMBA_CAUSAL.md](README_MAMBA_CAUSAL.md)** - Detailed usage and API documentation

## 🔬 Current Status

### ✅ Phase 1: Infrastructure Setup (COMPLETE)

- [x] Directory structure created
- [x] `mamba_models.py` - Load and inspect Mamba models
- [x] `ssm_nethook.py` - Hook Mamba layers for intervention
- [x] `mamba_layernames.py` - Parse Mamba architecture
- [x] `test_phase1.py` - Comprehensive test suite
- [x] Documentation and setup files

**Capabilities:**
- Load Mamba models (130m, 370m, 790m, 1.4b)
- Identify layer structure programmatically
- Hook layers to extract/modify activations
- Extract hidden states at specific token positions

### 🔜 Phase 2: Basic Causal Tracing (NEXT)

**TODO:**
- [ ] `mamba_repr_tools.py` - Extract representations at token positions
- [ ] `mamba_causal_trace.py` - Core causal tracing algorithm
- [ ] `run_causal_trace.py` - Experimental script
- [ ] Generate causal trace heatmaps

**Goal:** Identify which (layer, position) pairs are critical for predictions

### 🔮 Future Phases

- **Phase 3:** SSM-Specific State Tracing (internal states, selection parameters)
- **Phase 4:** Analysis and Visualization tools
- **Phase 5:** Scale up to larger models and diverse tasks

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for details.

## 🎓 Usage Examples

### Loading a Mamba Model

```python
from mamba_analysis.mamba_models import load_mamba_model

# Load model
mt = load_mamba_model("state-spaces/mamba-130m")
print(f"Loaded {mt.num_layers} layer model on {mt.device}")

# Tokenize input
inputs = mt.tokenizer("The Eiffel Tower is in", return_tensors="pt").to(mt.device)

# Run forward pass
import torch
with torch.no_grad():
    outputs = mt.model(**inputs)

# Get prediction
next_token_id = outputs.logits[0, -1].argmax()
next_token = mt.tokenizer.decode([next_token_id])
print(f"Predicted: {next_token}")
```

### Hooking Layers

```python
from util_ssm import ssm_nethook
import torch

# Hook a specific layer
layer_num = 5
with ssm_nethook.trace_mamba_layer(mt.model, layer_num, 'mixer') as trace:
    with torch.no_grad():
        outputs = mt.model(**inputs)
    hidden_states = trace.output

print(f"Hidden states shape: {hidden_states.shape}")
```

### Hooking Multiple Layers

```python
# Hook multiple layers at once
layer_specs = [(0, 'mixer'), (5, 'mixer'), (10, 'mixer')]

with ssm_nethook.trace_multiple_mamba_layers(mt.model, layer_specs) as traces:
    with torch.no_grad():
        outputs = mt.model(**inputs)

    for layer_num, component in layer_specs:
        layer_name = ssm_nethook.mamba_layername(mt.model, layer_num, component)
        hidden = traces[layer_name].output
        print(f"Layer {layer_num}: {hidden.shape}")
```

See [README_MAMBA_CAUSAL.md](README_MAMBA_CAUSAL.md) for more examples.

## 🧪 Testing

Run the test suite to verify everything is working:

```bash
# Activate environment first
conda activate mamba-causal

# Run Phase 1 tests
python test_phase1.py

# Run specific tests (once Phase 2 is implemented)
# python -m pytest tests/
```

## 🔧 Configuration

### Available Mamba Models

| Model | Parameters | HuggingFace ID |
|-------|-----------|----------------|
| Mamba-130m | 130M | `state-spaces/mamba-130m` |
| Mamba-370m | 370M | `state-spaces/mamba-370m` |
| Mamba-790m | 790M | `state-spaces/mamba-790m` |
| Mamba-1.4b | 1.4B | `state-spaces/mamba-1.4b` |

Start with `mamba-130m` for development and testing.

### System Requirements

**Minimum:**
- CPU: Any modern CPU
- RAM: 8 GB
- Storage: 5 GB for models and dependencies

**Recommended:**
- GPU: NVIDIA GPU with 8+ GB VRAM (for 130m/370m models)
- RAM: 16 GB
- Storage: 20 GB

**For larger models (1.4b):**
- GPU: NVIDIA GPU with 16+ GB VRAM
- RAM: 32 GB

## 🤝 Contributing

This is a research project adapting ROME's methodology to SSMs. Contributions are welcome!

**Current priorities:**
1. Implement Phase 2 (basic causal tracing)
2. Test on diverse factual prompts
3. Compare results with ROME's GPT-2 findings

## 📖 References

### Papers

- **ROME**: Meng et al., "Locating and Editing Factual Associations in GPT" (NeurIPS 2022)
  - Paper: https://arxiv.org/abs/2202.05262
  - Code: https://github.com/kmeng01/rome

- **Mamba**: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023)
  - Paper: https://arxiv.org/abs/2312.00752
  - Code: https://github.com/state-spaces/mamba

### Related Work

- Causal mediation analysis in NLP
- Mechanistic interpretability of neural networks
- State space models for sequence modeling

## 📄 License

This project builds upon the ROME repository. Please respect the original ROME license.

## 🐛 Troubleshooting

### Common Issues

**1. `mamba-ssm` installation fails**
```bash
# Make sure CUDA toolkit is installed
conda install cudatoolkit=11.8 -c conda-forge
pip install mamba-ssm
```

**2. CUDA out of memory**
```python
# Use smaller model
mt = load_mamba_model("state-spaces/mamba-130m", device="cpu")
# Or use half precision
mt = load_mamba_model("state-spaces/mamba-130m", torch_dtype=torch.float16)
```

**3. Import errors**
```bash
# Make sure you're in the project directory
cd /path/to/ssms
# And environment is activated
conda activate mamba-causal
```

See [SETUP_INSTRUCTIONS.md](SETUP_INSTRUCTIONS.md) for more troubleshooting.

## 💬 Contact

For questions about this project, please open an issue or refer to:
- Implementation plan: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)
- Setup guide: [SETUP_INSTRUCTIONS.md](SETUP_INSTRUCTIONS.md)
- ROME documentation: https://github.com/kmeng01/rome

---

**Status:** Phase 1 Complete ✓ | Phase 2 In Progress 🚧

Last updated: 2026-01-20
