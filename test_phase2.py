"""
Quick test for Phase 2: Basic Causal Tracing

Tests that the core causal tracing infrastructure works before running
full experiments.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from mamba_causal_analysis.mamba_models import load_mamba_model
from mamba_causal_analysis import mamba_repr_tools
from mamba_causal_analysis import mamba_causal_trace

print("=" * 70)
print("PHASE 2 TEST: Basic Causal Tracing")
print("=" * 70)
print()

# Test 1: Load model
print("Test 1: Loading model...")
try:
    mt = load_mamba_model("state-spaces/mamba-130m")
    print(f"✓ Model loaded: {mt.num_layers} layers")
except Exception as e:
    print(f"✗ Model loading failed: {e}")
    sys.exit(1)

print()

# Test 2: Extract states at tokens
print("Test 2: Extracting hidden states at specific tokens...")
try:
    contexts = ["The Eiffel Tower is located in"]
    layers = [0, mt.num_layers // 2, mt.num_layers - 1]

    states = mamba_repr_tools.get_mamba_states_at_tokens(
        mt.model,
        mt.tokenizer,
        contexts,
        layers,
        token_positions=[-1],  # Last token
        device=mt.device
    )

    print(f"✓ Extracted states shape: {states.shape}")
    print(f"  Expected: [1 context, {len(layers)} layers, hidden_size]")
    print(f"  Hidden size: {states.shape[-1]}")
except Exception as e:
    print(f"✗ State extraction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 3: Collect clean states
print("Test 3: Collecting clean states from all layers...")
try:
    prompt = "The Eiffel Tower is located in"
    tokens = mt.tokenizer(prompt, return_tensors="pt").to(mt.device)
    input_ids = tokens["input_ids"]

    clean_states = mamba_causal_trace.collect_states(
        mt.model,
        input_ids,
        layers=list(range(mt.num_layers)),
        component="mixer"
    )

    print(f"✓ Collected clean states shape: {clean_states.shape}")
    print(f"  Expected: [batch=1, {mt.num_layers} layers, seq_len, hidden_size]")
except Exception as e:
    print(f"✗ State collection failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 4: Single patch test (quick)
print("Test 4: Testing single state restoration...")
try:
    # Test restoring one (layer, position) pair
    test_layer = mt.num_layers // 2
    test_position = input_ids.shape[1] // 2

    logits = mamba_causal_trace.trace_with_patch_mamba(
        mt.model,
        input_ids,
        states_to_patch=[(test_layer, test_position)],
        clean_states=clean_states,
        noise_level=3.0,
    )

    print(f"✓ Single patch successful")
    print(f"  Patched: layer {test_layer}, position {test_position}")
    print(f"  Output logits shape: {logits.shape}")

    # Get top prediction
    probs = torch.softmax(logits[0, -1, :], dim=0)
    top_token = probs.argmax().item()
    top_prob = probs[top_token].item()
    token_str = mt.tokenizer.decode([top_token])
    print(f"  Top prediction: '{token_str}' (prob: {top_prob:.4f})")

except Exception as e:
    print(f"✗ Single patch test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 5: Mini causal trace (2 layers x 3 positions - very quick)
print("Test 5: Running mini causal trace (2 layers x 3 positions)...")
print("  This is a smoke test - full trace will be much larger")
try:
    # Override calculate_hidden_flow to do a mini version
    prompt = "The Eiffel Tower is"
    tokens = mt.tokenizer(prompt, return_tensors="pt").to(mt.device)
    input_ids = tokens["input_ids"]
    seq_len = input_ids.shape[1]

    # Test just 2 layers at 3 positions
    test_layers = [mt.num_layers // 3, 2 * mt.num_layers // 3]
    test_positions = [0, seq_len // 2, seq_len - 1]

    # Collect clean states
    clean_states = mamba_causal_trace.collect_states(
        mt.model,
        input_ids,
        layers=test_layers,
    )

    # Get clean baseline
    with torch.no_grad():
        clean_logits = mt.model(input_ids)
        if hasattr(clean_logits, 'logits'):
            clean_logits = clean_logits.logits
        clean_probs = torch.softmax(clean_logits[0, -1, :], dim=0)
        target_token = clean_probs.argmax().item()
        high_score = clean_probs[target_token].item()

    print(f"  Clean prediction: '{mt.tokenizer.decode([target_token])}' ({high_score:.4f})")

    # Get corrupted baseline
    corrupted_logits = mamba_causal_trace.trace_with_patch_mamba(
        mt.model,
        input_ids,
        states_to_patch=[],
        clean_states=clean_states,
        noise_level=3.0,
    )
    corrupted_probs = torch.softmax(corrupted_logits[0, -1, :], dim=0)
    low_score = corrupted_probs[target_token].item()

    print(f"  Corrupted prediction prob: {low_score:.4f}")
    print(f"  Effect of corruption: {high_score - low_score:.4f}")

    # Test a few restoration points
    print(f"  Testing {len(test_layers)} x {len(test_positions)} = {len(test_layers) * len(test_positions)} restorations...")

    for layer_idx, layer in enumerate(test_layers):
        for pos in test_positions:
            logits = mamba_causal_trace.trace_with_patch_mamba(
                mt.model,
                input_ids,
                states_to_patch=[(layer, pos)],
                clean_states=clean_states,
                noise_level=3.0,
            )
            probs = torch.softmax(logits[0, -1, :], dim=0)
            score = probs[target_token].item()

    print(f"✓ Mini causal trace successful")
    print(f"  All {len(test_layers) * len(test_positions)} restoration tests passed")

except Exception as e:
    print(f"✗ Mini causal trace failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 70)
print("ALL PHASE 2 TESTS PASSED! ✓")
print("=" * 70)
print()
print("Phase 2 basic infrastructure is working. You can now:")
print("  1. Run full causal trace:")
print("     python experiments_ssm/run_causal_trace.py \\")
print("       --prompt \"The Eiffel Tower is located in\" \\")
print("       --subject \"Eiffel Tower\" \\")
print("       --samples 10")
print()
print("  2. Use the Jupyter notebook:")
print("     jupyter lab notebooks/causal_tracing_visualization.ipynb")
print()
print("Note: Full causal tracing on all layers/positions takes several minutes.")
print("      The notebook provides an interactive way to visualize results.")
