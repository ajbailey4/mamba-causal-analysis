# Vertical Restoration Feature Summary

## What Was Added

### 1. Vertical Restoration Implementation
Added a new `mode="vertical"` to `calculate_hidden_flow()` in [mamba_causal_trace.py](mamba_causal_analysis/mamba_causal_trace.py).

**What it does:**
- Restores ALL layers at each position (vertical restoration)
- Answers: "How important is the information at position P across all model layers?"
- Complements the existing horizontal restoration (all positions in a layer)

**Code changes:**
- Lines 335-398: Added `trace_vertical_restoration()` helper function (not currently used, can be removed)
- Lines 529-535: Added vertical mode handling in `calculate_hidden_flow()`
- Lines 570-596: Added vertical restoration loop that creates `states_to_patch` for all layers at each position

### 2. Notebook Cells
Added 3 new cells to [causal_tracing_visualization.ipynb](notebooks/causal_tracing_visualization.ipynb):

**Cell after "4b. Horizontal vs Vertical Restoration" markdown:**
1. **Vertical Restoration Execution** - Runs vertical restoration and shows position-wise importance
2. **Comparison Visualization** - Side-by-side plots of horizontal vs vertical restoration
3. **Key Insights** - Explains the difference between horizontal and vertical restoration

## How It Works

### Horizontal Restoration (Existing)
- **Mode:** `mode="per_layer"` with `patch_entire_layer=True`
- **What it restores:** All time steps (positions) in a single layer
- **Question:** "Which layer is most important?"
- **Implementation:** `clean_hidden_states[:, layer_idx, :, :]` restores ALL positions
- **Why it works:** Hooks process the full sequence AFTER it's computed through the layer

### Vertical Restoration (NEW)
- **Mode:** `mode="vertical"`
- **What it restores:** All layers at a single time step (position)
- **Question:** "Which position contains the most critical information?"
- **Implementation:** For each position, creates `[(layer0, pos), (layer1, pos), ..., (layerN, pos)]`
- **Uses:** `patch_entire_layer=False` for per-position patching

## Verification

Horizontal restoration DOES work correctly:
- Checked code at lines 148-161 in mamba_causal_trace.py
- When `patch_entire_layer=True`, we restore `clean_hidden_states[:, layer_idx, :, :]`
- The `:` in the sequence dimension means ALL positions are restored
- The hook runs AFTER the full sequence is processed, so all positions are available

## Test Results

From `test_vertical_restoration.py`:
```
Prompt: "The Eiffel Tower is located in"

Horizontal (per-layer) best: Layer 0
  Score: 0.5299

Vertical (all layers at position) best: Position 7
  Token: ' in'
  Score: 0.5299
```

This makes sense:
- **Layer 0** (horizontal): Restoring early layer cleans entire residual stream forward
- **Position 7** (vertical): Last position in autoregressive model has full context

## Usage Example

```python
# Run vertical restoration
result_vertical = calculate_hidden_flow(
    mt,
    mt.tokenizer,
    prompt="The Eiffel Tower is located in",
    subject="Eiffel Tower",
    samples=5,
    noise_level=3.0,
    mode="vertical",  # NEW!
)

# Result shape: [seq_len] - one score per position
print(result_vertical['scores'].shape)  # (8,)
```

## Key Findings

1. **Horizontal restoration works as expected** - Restores all positions in a layer simultaneously
2. **Vertical restoration complements horizontal** - Tests which positions carry critical info
3. **Both modes provide different insights:**
   - Horizontal: Layer importance (residual stream flow)
   - Vertical: Position importance (temporal/recurrent flow)

## Files Modified

1. `mamba_causal_analysis/mamba_causal_trace.py`
   - Added vertical mode support
   - Updated docstrings

2. `notebooks/causal_tracing_visualization.ipynb`
   - Added 3 cells demonstrating vertical restoration
   - Added comparison visualization

3. `test_vertical_restoration.py` (new)
   - Validation script for vertical restoration feature

4. `vertical_restoration_test.png` (generated)
   - Visualization output from test script
