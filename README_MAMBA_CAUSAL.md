# Causal Mediation Analysis for Mamba SSMs

Adapting the causal mediation analysis methodology from Meng et al's "Locating and Editing Factual Associations in GPT" (ROME) to work with Mamba State Space Models.

## Project Goal

Understand how Mamba SSMs store and retrieve factual information through causal tracing analysis, comparing their memory mechanisms to traditional transformer architectures.

## Setup Instructions

### 1. Create Conda Environment

```bash
# Create new conda environment
conda create -n mamba-causal python=3.10 -y
conda activate mamba-causal
```

### 2. Install Dependencies

```bash
# Install PyTorch (adjust CUDA version as needed)
# For CUDA 11.8:
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# For CPU only:
# conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# Install other dependencies
pip install mamba-ssm transformers datasets matplotlib numpy jupyter ipykernel tqdm

# Register Jupyter kernel (optional, for notebooks)
python -m ipykernel install --user --name mamba-causal --display-name "Mamba Causal Analysis"
```

### 3. Verify Installation

Run the Phase 1 test script:

```bash
python test_phase1.py
```

This should output:
- ✓ All modules imported successfully
- ✓ Model loaded successfully
- ✓ Layer structure identified
- ✓ All tests passing

## Project Structure

```
ssms/
├── mamba_analysis/          # Mamba-specific analysis code
│   ├── __init__.py
│   ├── mamba_models.py      # Model loading and layer identification ✓
│   ├── mamba_repr_tools.py  # Representation extraction (TODO: Phase 2)
│   ├── mamba_causal_trace.py # Causal tracing (TODO: Phase 2)
│   └── mamba_hparams.py     # Hyperparameters (TODO: Phase 3)
├── util_ssm/                # SSM-specific utilities
│   ├── __init__.py
│   ├── ssm_nethook.py       # Hooking utilities ✓
│   └── mamba_layernames.py  # Layer naming helpers ✓
├── experiments_ssm/         # Experimental scripts
│   ├── __init__.py
│   ├── run_causal_trace.py  # Main experiment script (TODO: Phase 2)
│   └── visualize_flow.py    # Visualization (TODO: Phase 4)
├── configs/                 # Configuration files
│   └── mamba/               # Mamba-specific configs (TODO: Phase 2)
├── notebooks/               # Analysis notebooks
│   └── mamba_exploration.ipynb (TODO: Phase 4)
├── rome/                    # Original ROME repository
└── test_phase1.py           # Phase 1 verification script ✓
```

## Implementation Phases

### ✓ Phase 1: Infrastructure Setup (COMPLETE)

**Completed:**
- ✓ Created directory structure
- ✓ Implemented `mamba_models.py` - Model loading and layer identification
- ✓ Implemented `ssm_nethook.py` - SSM-aware hooking utilities
- ✓ Implemented `mamba_layernames.py` - Layer naming conventions
- ✓ Created `test_phase1.py` - Verification script

**Verification:**
```bash
python test_phase1.py
```

### Phase 2: Basic Causal Tracing (TODO)

**Next Steps:**
1. Implement `mamba_repr_tools.py` - Extract hidden states at specific token positions
2. Implement `mamba_causal_trace.py` - Core causal tracing algorithm
3. Create `experiments_ssm/run_causal_trace.py` - Experimental script
4. Test on factual prompts and generate heatmaps

### Phase 3: SSM-Specific State Tracing (TODO)

**Goals:**
- Extract internal SSM states (h_t)
- Extract selection parameters (B, C, Δt)
- Compare different intervention types

### Phase 4: Analysis and Visualization (TODO)

**Goals:**
- Create visualization tools
- Compare Mamba vs GPT-2 information flow
- Generate insights about SSM memory structure

### Phase 5: Scale Up and Iterate (TODO)

**Goals:**
- Test on larger models (370m, 790m, 1.4b)
- Diverse test cases
- Advanced analysis

## Quick Start Guide

### Loading a Mamba Model

```python
from mamba_analysis.mamba_models import load_mamba_model

# Load model
mt = load_mamba_model("state-spaces/mamba-130m")

print(f"Loaded {mt.num_layers} layer model")
print(f"Device: {mt.device}")
```

### Hooking Layers

```python
from util_ssm import ssm_nethook
import torch

# Prepare input
inputs = mt.tokenizer("The Eiffel Tower is located in", return_tensors="pt").to(mt.device)

# Hook a single layer
with ssm_nethook.trace_mamba_layer(mt.model, layer_num=5, component='mixer') as trace:
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

## Available Mamba Models

Start with smaller models and scale up:

1. **mamba-130m** (recommended for development): `state-spaces/mamba-130m`
2. **mamba-370m**: `state-spaces/mamba-370m`
3. **mamba-790m**: `state-spaces/mamba-790m`
4. **mamba-1.4b** (comparable to GPT-2 XL): `state-spaces/mamba-1.4b`

## Key Differences: Transformer vs Mamba

| Aspect | Transformer (GPT-2) | Mamba SSM |
|--------|-------------------|-----------|
| Layer structure | `transformer.h.{i}.{attn\|mlp}` | `backbone.layers.{i}.mixer` |
| Processing | Parallel (all positions at once) | Sequential (position t depends on 0...t-1) |
| Memory | Explicit (attention weights) | Compressed (recurrent state h_t) |
| Selection | Attention scores | Input-dependent gating (B, C, Δt) |

## Reusable Components from ROME

The following components from ROME work with any PyTorch model:
- `util/nethook.py`: `Trace`, `TraceDict`, `recursive_copy`
- Noise injection patterns
- Input preparation utilities
- Dataset loading

We've created SSM-specific wrappers around these in `util_ssm/`.

## Research Questions

1. **Memory Localization**: Which layers store factual information in Mamba?
2. **Storage Mechanism**: Is information in the recurrent state h_t or selection parameters B, C, Δt?
3. **Information Flow**: How does Mamba's sequential processing differ from transformers?
4. **Layer Specialization**: Do Mamba layers specialize like transformer layers?
5. **Efficiency**: How does Mamba's compressed state compare to attention for factual recall?

## Troubleshooting

### `mamba-ssm` Installation Issues

If `pip install mamba-ssm` fails:

```bash
# Make sure you have the right CUDA toolkit
conda install cudatoolkit=11.8 -c conda-forge

# Try installing with specific versions
pip install mamba-ssm==1.2.0
```

### CUDA Out of Memory

For GPU memory issues:

```python
# Use smaller model
mt = load_mamba_model("state-spaces/mamba-130m")

# Or use CPU
mt = load_mamba_model("state-spaces/mamba-130m", device="cpu")

# Or use half precision
mt = load_mamba_model("state-spaces/mamba-130m", torch_dtype=torch.float16)
```

### Import Errors

Make sure you're in the project directory and conda environment is activated:

```bash
cd /Users/ajbailey4@ad.wisc.edu/ssms
conda activate mamba-causal
python test_phase1.py
```

## Next Steps

After Phase 1 is verified:

1. Implement Phase 2 components (representation extraction and causal tracing)
2. Run experiments on factual prompts
3. Compare results with ROME's GPT-2 results
4. Extend to SSM-specific interventions

## References

- **ROME Paper**: "Locating and Editing Factual Associations in GPT" (Meng et al., 2022)
- **ROME Code**: https://github.com/kmeng01/rome
- **Mamba Paper**: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Dao, 2023)
- **Mamba Code**: https://github.com/state-spaces/mamba

## Contact & Support

For issues specific to this adaptation, check:
1. The plan file: `.claude/plans/woolly-stargazing-ocean.md`
2. ROME's original documentation
3. Mamba's documentation

## License

This code builds upon the ROME repository. Please respect the original ROME license and cite both ROME and Mamba if you use this work.
