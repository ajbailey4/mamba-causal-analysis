"""
Test causal tracing functionality.

This tests the core causal tracing implementation including:
- State collection
- Embedding corruption
- State restoration/patching
- All three tracing modes

Usage:
    python tests/test_causal_tracing.py
    python -m pytest tests/test_causal_tracing.py -v
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test configuration
TEST_PROMPT = "The Eiffel Tower is located in"
TEST_SUBJECT = "Eiffel Tower"


def get_model():
    """Load the model (cached for efficiency)."""
    from mamba_causal_analysis.mamba_models import load_mamba_model
    return load_mamba_model("state-spaces/mamba-130m")


def test_state_extraction():
    """Test extracting hidden states at specific tokens."""
    from mamba_causal_analysis import mamba_repr_tools

    mt = get_model()
    layers = [0, mt.num_layers // 2, mt.num_layers - 1]

    states = mamba_repr_tools.get_mamba_states_at_tokens(
        mt.model,
        mt.tokenizer,
        [TEST_PROMPT],
        layers,
        token_positions=[-1],
        device=mt.device
    )

    assert states.shape[0] == 1, f"Expected batch size 1, got {states.shape[0]}"
    assert states.shape[1] == len(layers), f"Expected {len(layers)} layers, got {states.shape[1]}"
    print(f"  State extraction: shape={states.shape}")


def test_state_collection():
    """Test collecting clean states from all layers."""
    from mamba_causal_analysis import mamba_causal_trace

    mt = get_model()
    tokens = mt.tokenizer(TEST_PROMPT, return_tensors="pt").to(mt.device)
    input_ids = tokens["input_ids"]

    clean_states = mamba_causal_trace.collect_states(
        mt.model,
        input_ids,
        layers=list(range(mt.num_layers)),
        component="mixer"
    )

    assert clean_states.shape[0] == 1, "Expected batch size 1"
    assert clean_states.shape[1] == mt.num_layers, f"Expected {mt.num_layers} layers"
    assert clean_states.shape[2] == input_ids.shape[1], "Sequence length mismatch"
    print(f"  State collection: shape={clean_states.shape}")
    return mt, input_ids, clean_states


def test_single_patch():
    """Test patching a single (layer, position) pair."""
    from mamba_causal_analysis import mamba_causal_trace

    mt, input_ids, clean_states = test_state_collection()

    test_layer = mt.num_layers // 2
    test_position = input_ids.shape[1] // 2

    logits = mamba_causal_trace.trace_with_patch_mamba(
        mt.model,
        input_ids,
        states_to_patch=[(test_layer, test_position)],
        clean_states=clean_states,
        noise_level=3.0,
    )

    assert logits.shape[0] == 1
    assert logits.shape[1] == input_ids.shape[1]
    print(f"  Single patch: layer={test_layer}, position={test_position}, logits shape={logits.shape}")


def test_corruption_effect():
    """Test that corruption actually affects the output."""
    from mamba_causal_analysis import mamba_causal_trace

    mt = get_model()
    tokens = mt.tokenizer(TEST_PROMPT, return_tensors="pt").to(mt.device)
    input_ids = tokens["input_ids"]

    # Get clean prediction
    with torch.no_grad():
        clean_output = mt.model(input_ids)
        clean_logits = clean_output.logits if hasattr(clean_output, 'logits') else clean_output
        clean_probs = torch.softmax(clean_logits[0, -1, :], dim=0)
        target_id = clean_probs.argmax().item()
        clean_prob = clean_probs[target_id].item()

    # Get corrupted prediction
    clean_states = mamba_causal_trace.collect_states(
        mt.model, input_ids, list(range(mt.num_layers))
    )

    corrupted_logits = mamba_causal_trace.trace_with_patch_mamba(
        mt.model,
        input_ids,
        states_to_patch=[],
        clean_states=clean_states,
        noise_level=3.0,
    )
    corrupted_probs = torch.softmax(corrupted_logits[0, -1, :], dim=0)
    corrupted_prob = corrupted_probs[target_id].item()

    effect = clean_prob - corrupted_prob
    assert effect > 0.1, f"Corruption should significantly reduce probability, got effect={effect:.4f}"
    print(f"  Corruption effect: clean={clean_prob:.4f}, corrupted={corrupted_prob:.4f}, effect={effect:.4f}")


def test_subject_only_corruption():
    """Test that only subject tokens are corrupted (bug fix validation)."""
    from mamba_causal_analysis.mamba_causal_trace import calculate_hidden_flow

    mt = get_model()

    result = calculate_hidden_flow(
        mt,
        mt.tokenizer,
        prompt=TEST_PROMPT,
        subject=TEST_SUBJECT,
        samples=3,
        noise_level=3.0,
        mode="per_layer",
    )

    # Validation: corruption should work (low corrupted prob)
    assert result['low_score'] < 0.1, f"Corrupted prob should be low, got {result['low_score']:.4f}"

    # Validation: restoration should work (layer 0 should recover prediction)
    assert result['scores'][0] > 0.4, f"Layer 0 should restore well, got {result['scores'][0]:.4f}"

    print(f"  Subject-only corruption: corrupted={result['low_score']:.4f}, layer0_restored={result['scores'][0]:.4f}")


def test_per_layer_mode():
    """Test per-layer causal tracing mode."""
    from mamba_causal_analysis.mamba_causal_trace import calculate_hidden_flow

    mt = get_model()

    result = calculate_hidden_flow(
        mt,
        mt.tokenizer,
        prompt=TEST_PROMPT,
        subject=TEST_SUBJECT,
        samples=3,
        noise_level=3.0,
        mode="per_layer",
    )

    assert result['scores'].ndim == 1, f"Expected 1D scores, got shape {result['scores'].shape}"
    assert len(result['scores']) == 24, f"Expected 24 layers, got {len(result['scores'])}"
    assert result['scores'][0] > 0.4, f"Layer 0 should restore well, got {result['scores'][0]:.4f}"

    best_layer = result['scores'].argmax()
    print(f"  Per-layer mode: shape={result['scores'].shape}, best_layer={best_layer}")


def test_per_position_mode():
    """Test per-position (hidden state) causal tracing mode."""
    from mamba_causal_analysis.mamba_causal_trace import calculate_hidden_flow

    mt = get_model()

    result = calculate_hidden_flow(
        mt,
        mt.tokenizer,
        prompt=TEST_PROMPT,
        subject=TEST_SUBJECT,
        samples=3,
        noise_level=3.0,
        mode="per_position",
    )

    assert result['scores'].ndim == 2, f"Expected 2D scores, got shape {result['scores'].shape}"
    assert result['scores'].shape[0] == 24, f"Expected 24 layers, got {result['scores'].shape[0]}"
    assert result['scores'].max() > 0.1, f"Should have some restoration effect, max={result['scores'].max():.4f}"

    max_layer, max_pos = np.unravel_index(result['scores'].argmax(), result['scores'].shape)
    print(f"  Per-position mode: shape={result['scores'].shape}, best=(layer={max_layer}, pos={max_pos})")


def test_per_position_ssm_mode(run_slow=False):
    """Test per-position SSM state causal tracing mode.

    Note: This test is slow (15-30 minutes). Set run_slow=True to run it.
    """
    if not run_slow:
        print("  Per-position SSM mode: SKIPPED (slow test, set run_slow=True)")
        return

    from mamba_causal_analysis.mamba_causal_trace import calculate_hidden_flow

    mt = get_model()

    result = calculate_hidden_flow(
        mt,
        mt.tokenizer,
        prompt=TEST_PROMPT,
        subject=TEST_SUBJECT,
        samples=2,  # Minimal samples for testing
        noise_level=3.0,
        mode="per_position_ssm",
    )

    assert result['scores'].ndim == 2, f"Expected 2D scores, got shape {result['scores'].shape}"
    assert result['scores'].shape[0] == 24, f"Expected 24 layers, got {result['scores'].shape[0]}"

    max_layer, max_pos = np.unravel_index(result['scores'].argmax(), result['scores'].shape)
    print(f"  Per-position SSM mode: shape={result['scores'].shape}, best=(layer={max_layer}, pos={max_pos})")


def run_all_tests(include_slow=False):
    """Run all causal tracing tests."""
    print("=" * 70)
    print("CAUSAL TRACING TESTS")
    print("=" * 70)
    print()

    print("Test 1: State extraction")
    test_state_extraction()
    print("  PASSED")
    print()

    print("Test 2: State collection")
    test_state_collection()
    print("  PASSED")
    print()

    print("Test 3: Single patch")
    test_single_patch()
    print("  PASSED")
    print()

    print("Test 4: Corruption effect")
    test_corruption_effect()
    print("  PASSED")
    print()

    print("Test 5: Subject-only corruption (bug fix validation)")
    test_subject_only_corruption()
    print("  PASSED")
    print()

    print("Test 6: Per-layer mode")
    test_per_layer_mode()
    print("  PASSED")
    print()

    print("Test 7: Per-position mode (hidden states)")
    test_per_position_mode()
    print("  PASSED")
    print()

    print("Test 8: Per-position SSM mode")
    test_per_position_ssm_mode(run_slow=include_slow)
    if include_slow:
        print("  PASSED")
    print()

    print("=" * 70)
    print("ALL CAUSAL TRACING TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-slow", action="store_true", help="Include slow SSM tests")
    args = parser.parse_args()

    run_all_tests(include_slow=args.include_slow)
