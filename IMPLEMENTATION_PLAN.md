# Causal Mediation Analysis for Mamba SSMs - Implementation Plan

## Project Goal
Adapt the causal mediation analysis methodology from Meng et al's "Locating and Editing Factual Associations in GPT" (ROME) to work with Mamba State Space Models, with the goal of understanding SSM internal memory structure.

## Background
The ROME repository implements causal tracing for transformers by:
1. Corrupting input embeddings with noise
2. Selectively restoring clean hidden states at specific (layer, token position) pairs
3. Measuring the effect on output predictions
4. Creating heatmaps showing which states are critical for factual recall

This plan adapts this methodology for Mamba's recurrent state space architecture.

## Key Architectural Differences: Transformer vs Mamba

**Transformer (GPT-2):**
- Layer structure: `transformer.h.{layer}.{attn|mlp}`
- Parallel processing: all positions computed simultaneously
- Explicit memory: attention weights store all previous positions

**Mamba SSM:**
- Layer structure: `backbone.layers.{layer}.mixer`
- Sequential processing: state at position t depends on all positions 0...t
- Compressed memory: recurrent state h_t + selective gating (B, C, Δt)
- Novel components: causal convolution, input-dependent selection

## Directory Structure

```
ssms/
├── mamba_analysis/          # New directory (parallel to rome/)
│   ├── __init__.py
│   ├── mamba_models.py      # Model loading and layer identification
│   ├── mamba_repr_tools.py  # SSM-specific representation extraction
│   ├── mamba_causal_trace.py # Causal tracing for SSMs
│   └── mamba_hparams.py     # Hyperparameters config
├── util_ssm/                # SSM-specific utilities
│   ├── __init__.py
│   ├── ssm_nethook.py       # Adapted hooking for SSM layers
│   └── mamba_layernames.py  # Layer naming conventions
├── experiments_ssm/         # SSM experiments
│   ├── __init__.py
│   ├── run_causal_trace.py  # Main experimental script
│   └── visualize_flow.py    # Visualization tools
├── configs/                 # Configuration files
│   └── mamba/
│       ├── mamba-130m.json
│       ├── mamba-370m.json
│       └── mamba-790m.json
└── notebooks/              # Analysis notebooks
    └── mamba_exploration.ipynb
```

## Implementation Phases

### Phase 1: Infrastructure Setup ✓ COMPLETE

**Goal:** Load Mamba models and enable basic hooking

**Completed Tasks:**
1. ✓ Set up directory structure
2. ✓ Create `mamba_analysis/mamba_models.py`
   - `MambaModelAndTokenizer` class (similar to ROME's `ModelAndTokenizer`)
   - Load model: `state-spaces/mamba-130m` (start small)
   - Identify layer structure programmatically
   - Map layer names (e.g., `backbone.layers.{i}.mixer`)

3. ✓ Create `util_ssm/ssm_nethook.py`
   - Extend ROME's `Trace`/`TraceDict` (which are architecture-agnostic)
   - `ssm_layername(model, num, component=None)` function
   - Components: None (full mixer), 'in_proj', 'out_proj', 'conv1d', 'ssm'
   - Test hooking: verify we can extract hidden states

4. ✓ Create `util_ssm/mamba_layernames.py`
   - Helper functions to resolve Mamba-specific module paths
   - Handle different Mamba variants (Mamba-1 vs Mamba-2)

**Verification:**
```bash
python test_phase1.py  # All tests should pass ✓
```

**Critical Files:**
- ✓ `mamba_analysis/mamba_models.py`
- ✓ `util_ssm/ssm_nethook.py`
- ✓ `util_ssm/mamba_layernames.py`
- ✓ `test_phase1.py`

### Phase 2: Basic Causal Tracing ✓ COMPLETE

**Goal:** Implement causal tracing for Mamba hidden states and residual stream

**Completed Tasks:**
1. ✓ Created `mamba_analysis/mamba_repr_tools.py`
   - `find_token_range()`: Locate subject tokens in prompts

2. ✓ Created `mamba_analysis/mamba_causal_trace.py`
   - `collect_states()`: Extract hidden states AND residuals from full blocks
   - `trace_with_patch_mamba()`: Core causal tracing with embedding corruption + state restoration
   - Supports both per-layer and per-position patching modes
   - Properly handles Mamba's residual stream (both hidden_states and residuals from blocks)

3. ✓ Created `calculate_hidden_flow()`: Main entry point
   - Per-layer mode (default): Test which layers are important
   - Per-position mode: Test which (layer, position) pairs matter
   - Returns scores ready for visualization

4. ✓ Created `notebooks/causal_tracing_visualization.ipynb`
   - Interactive visualization of causal tracing results
   - Supports both 1D (per-layer) and 2D (per-position) heatmaps
   - Tested on factual prompts

**Key Finding:** Per-layer mode shows early layers (especially layer 0) are critical because Mamba's residual stream flows forward through layers. This is different from transformers where late layers can be restored independently.

**Verification:** ✓ Per-layer heatmaps working correctly in notebook

**Critical Files:**
- ✓ `mamba_causal_analysis/mamba_causal_trace.py`
- ✓ `mamba_causal_analysis/mamba_repr_tools.py`
- ✓ `notebooks/causal_tracing_visualization.ipynb`

### Phase 3: Per-Position SSM State Tracing (IN PROGRESS)

**Goal:** Trace internal SSM states (h_t, conv_state) at per-position granularity

**Current Status:** Planning complete, implementing Option 3 (token-by-token prototype)

**Approach Analysis:**
- **Option 1 (Hook within selective_scan)**: ❌ Not feasible - it's a compiled CUDA kernel
- **Option 2 (Hybrid batching)**: ✅ Best performance (10-50x faster) but complex (3-5 hours to implement)
- **Option 3 (Token-by-token)**: ✅ Simplest (1-2 hours to implement) but slow (30-60 min runtime)

**Selected Approach:** Start with Option 3 for fastest time-to-first-heatmap (~2-3 hours total), then optimize with Option 2 if needed.

**Implementation Plan (Option 3 - Token-by-Token Prototype):**

1. Create `collect_ssm_states_sequential()` in `mamba_causal_trace.py`
   - Use `InferenceParams` to process tokens one-by-one
   - Save conv_state and ssm_state after each token at each layer
   - Returns: {layer_idx: {position: (conv_state, ssm_state)}}

2. Create `trace_with_ssm_state_patch_sequential()`
   - Corrupt embeddings
   - Process token-by-token through all layers
   - At target (layer, position), inject clean SSM states
   - Measure effect on final output

3. Extend `calculate_hidden_flow()` with new mode
   - Add `mode="per_position_sequential"` parameter
   - Returns 2D scores array [num_layers, seq_len]

4. Update notebook with per-position visualization
   - Add 2D heatmap plotting
   - Compare with per-layer results

**Technical Details:**
- Uses Mamba's `InferenceParams` with `key_value_memory_dict`
- `seqlen_offset` tracks current position
- States: `conv_state` [batch, d_inner, d_conv], `ssm_state` [batch, d_inner, d_state]
- Inject states by modifying `key_value_memory_dict[layer_idx]`

**Performance:**
- Estimated runtime: 30-60 minutes for 24 layers × 8 positions × 10 samples
- Acceptable for short sequences (<10 tokens)
- Will optimize with Option 2 (hybrid batching) if needed

**Next Steps After Prototype:**
1. Validate results are meaningful (early layers × early positions should matter)
2. Compare with per-layer results for consistency
3. Decide if Option 2 optimization is needed

**Verification:** Can selectively restore different components of SSM computation and measure differential effects

**Critical Files:**
- `mamba_analysis/mamba_repr_tools.py` (enhanced)
- `mamba_analysis/mamba_causal_trace.py` (enhanced)

### Phase 4: Analysis and Visualization (TODO)

**Goal:** Tools for understanding SSM memory structure

**Tasks:**
1. Create `experiments_ssm/visualize_flow.py`
   - Multi-level heatmaps (hidden state, SSM state, selection params)
   - Comparative visualization: Mamba vs GPT-2 on same prompts
   - Selection parameter evolution plots (how B, C, Δt change with corruption)

2. Create analysis functions in `mamba_analysis/state_analysis.py`
   - Memory decay analysis: how long does information persist in h_t?
   - Selection mechanism analysis: what patterns in B, C, Δt correlate with recall?
   - Layer specialization: do different layers serve different roles?

3. Create `notebooks/mamba_exploration.ipynb`
   - Interactive exploration of causal tracing results
   - Compare factual recall patterns between Mamba and transformers
   - Visualize SSM state evolution over sequences

4. Documentation
   - README with usage examples
   - Comparison with ROME methodology
   - Findings about SSM memory structure

**Verification:** Clear visualizations and insights about where and how Mamba stores factual information

**Critical Files:**
- `experiments_ssm/visualize_flow.py`
- `notebooks/mamba_exploration.ipynb`

### Phase 5: Scale Up and Iterate (TODO)

**Goal:** Larger models and refined analysis

**Tasks:**
1. Test on larger Mamba models
   - Mamba-370m, Mamba-790m, Mamba-1.4b
   - Update configs in `configs/mamba/`
   - Optimize memory usage (gradient checkpointing, batching)

2. Diverse test cases
   - Factual recall (like ROME)
   - Long-range dependencies (where SSMs should excel)
   - Multi-hop reasoning

3. Advanced analysis
   - State space visualization (project h_t into 2D/3D)
   - Eigenvalue analysis of A matrix
   - Clustering of selection patterns

**Verification:** Comprehensive understanding of Mamba's memory mechanisms across model sizes and tasks

## Mamba Model Setup

**Models to use (in order):**
1. **mamba-130m** (start here): `state-spaces/mamba-130m`
2. **mamba-370m**: `state-spaces/mamba-370m`
3. **mamba-790m**: `state-spaces/mamba-790m`
4. **mamba-1.4b** (comparable to GPT-2 XL): `state-spaces/mamba-1.4b`

**Loading code:**
```python
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import AutoTokenizer

model = MambaLMHeadModel.from_pretrained("state-spaces/mamba-130m")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")  # Mamba uses GPT-NeoX tokenizer
model.eval().cuda()
```

## Attribution: ROME

This project builds upon the ROME methodology and includes `nethook.py` from the ROME repository:
- **Original ROME**: https://github.com/kmeng01/rome
- **Paper**: "Locating and Editing Factual Associations in GPT" (Meng et al., NeurIPS 2022)
- **License**: MIT

**What we use from ROME:**
- `util_ssm/nethook.py`: Copy of ROME's model instrumentation utilities (`Trace`, `TraceDict`, `recursive_copy`)
- Conceptual framework for causal tracing methodology

**What we've adapted for Mamba:**
- `mamba_models.py`: Mamba-specific `ModelAndTokenizer` class
- `ssm_nethook.py`: Wrappers for Mamba layer naming conventions
- `mamba_layernames.py`: Mamba architecture parsing
- Future: `mamba_causal_trace.py` will adapt `trace_with_patch()` for recurrent states

## Key Research Questions

1. **Memory Localization:** Which layers store factual information in Mamba?
2. **Storage Mechanism:** Is information in the recurrent state h_t or in the selection parameters B, C, Δt?
3. **Information Flow:** How does information propagate through Mamba's sequential processing vs transformer's parallel processing?
4. **Layer Specialization:** Do Mamba layers specialize (early=encoding, middle=processing, late=recall) like transformers?
5. **Efficiency Trade-offs:** How does Mamba's compressed state compare to attention's explicit memory for factual recall?

## Technical Challenges and Solutions

### Challenge 1: Recurrent State Dependencies
**Problem:** In SSMs, state at position t depends on all previous positions. Restoring h_t at position t affects all positions > t.

**Solution:**
- Start by restoring hidden states (mixer outputs) like ROME does - this is simpler
- Later, explore restoring internal SSM state h_t with full awareness of downstream effects
- Consider "local" interventions within the conv window only

### Challenge 2: Accessing Internal SSM States
**Problem:** The SSM recurrent state h_t is computed inside `selective_scan_fn` and not directly accessible.

**Solution:**
- Hook at sub-module level to intercept SSM computation
- May need to monkey-patch or reimplement selective_scan to expose h_t
- Start with publicly accessible tensors (hidden states, B, C, Δt projections)

### Challenge 3: Selection Mechanism Complexity
**Problem:** B, C, Δt are computed from hidden states, creating circular dependencies when restoring.

**Solution:**
- Separate experiments: restore parameters vs states independently
- Measure correlation between parameter changes and output changes
- Use partial interventions to disentangle effects

## Success Criteria

**Phase 1:** ✓ Successfully load Mamba-130m and extract hidden states at arbitrary layers
**Phase 2:** Generate causal trace heatmaps showing (layer, position) importance for Mamba predictions
**Phase 3:** Compare effects of restoring hidden states vs internal SSM states vs selection parameters
**Phase 4:** Clear visualizations showing differences between Mamba and transformer information flow
**Phase 5:** Comprehensive analysis across multiple Mamba model sizes with documented findings

## Progress Tracking

- [x] Phase 1: Infrastructure Setup - COMPLETE
  - [x] Directory structure
  - [x] `mamba_models.py`
  - [x] `ssm_nethook.py`
  - [x] `mamba_layernames.py`
  - [x] `test_phase1.py`
  - [x] Documentation
- [x] Phase 2: Basic Causal Tracing - COMPLETE
  - [x] `mamba_repr_tools.py`
  - [x] `mamba_causal_trace.py` (per-layer + per-position modes)
  - [x] `notebooks/causal_tracing_visualization.ipynb`
  - [x] Initial experiments and per-layer heatmaps
  - [x] Fixed residual stream patching (patch both hidden_states and residuals)
- [ ] Phase 3: Per-Position SSM State Tracing - IN PROGRESS
  - [ ] `collect_ssm_states_sequential()` - Token-by-token state collection
  - [ ] `trace_with_ssm_state_patch_sequential()` - SSM state injection
  - [ ] Update `calculate_hidden_flow()` with per-position mode
  - [ ] 2D heatmap visualization (layers × positions)
  - [ ] Validation and comparison with per-layer results
- [ ] Phase 4: Analysis and Visualization
- [ ] Phase 5: Scale Up and Iterate

## References

- **ROME Paper**: "Locating and Editing Factual Associations in GPT" (Meng et al., 2022)
- **ROME Code**: https://github.com/kmeng01/rome
- **Mamba Paper**: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Dao, 2023)
- **Mamba Code**: https://github.com/state-spaces/mamba
