"""
Test infrastructure: Model loading, layer identification, and hooking.

This tests the foundational components needed for causal tracing.

Usage:
    python tests/test_infrastructure.py
    python -m pytest tests/test_infrastructure.py -v
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test that all modules can be imported."""
    from mamba_causal_analysis.mamba_models import MambaModelAndTokenizer, load_mamba_model
    from util_ssm import ssm_nethook
    from util_ssm import mamba_layernames
    print("  All modules imported successfully")


def test_model_loading():
    """Test loading a Mamba model."""
    from mamba_causal_analysis.mamba_models import load_mamba_model

    mt = load_mamba_model(
        model_name="state-spaces/mamba-130m",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    assert mt.num_layers == 24, f"Expected 24 layers, got {mt.num_layers}"
    assert mt.model is not None
    assert mt.tokenizer is not None
    print(f"  Model loaded: {mt.num_layers} layers on {mt.device}")
    return mt


def test_layer_structure(mt):
    """Test layer structure identification."""
    assert mt.get_layer_name(0) is not None
    assert mt.get_layer_name(mt.num_layers - 1) is not None
    assert mt.get_layer_name(0, 'mixer') is not None
    assert mt.get_embedding_layer_name() is not None
    assert mt.get_lm_head_name() is not None
    print(f"  Layer structure: {mt.get_layer_name(0)} to {mt.get_layer_name(mt.num_layers - 1)}")


def test_architecture_parsing(mt):
    """Test architecture parsing with mamba_layernames."""
    from util_ssm import mamba_layernames

    arch_info = mamba_layernames.parse_mamba_architecture(mt.model)

    assert arch_info['num_layers'] == 24
    assert arch_info['layer_prefix'] is not None
    assert len(arch_info['layer_names']) == 24
    print(f"  Architecture: {arch_info['num_layers']} layers, prefix={arch_info['layer_prefix']}")


def test_forward_pass(mt):
    """Test running a forward pass."""
    test_prompt = "The Eiffel Tower is located in"
    inputs = mt.tokenizer(test_prompt, return_tensors="pt").to(mt.device)
    model_inputs = {k: v for k, v in inputs.items() if k == 'input_ids'}

    with torch.no_grad():
        outputs = mt.model(**model_inputs)

    assert outputs.logits.shape[0] == 1
    assert outputs.logits.shape[1] == inputs['input_ids'].shape[1]

    next_token_id = outputs.logits[0, -1].argmax().item()
    next_token = mt.tokenizer.decode([next_token_id])
    print(f"  Forward pass: '{test_prompt}' -> '{next_token}'")


def test_single_layer_hook(mt):
    """Test hooking a single layer."""
    from util_ssm import ssm_nethook

    test_prompt = "The Eiffel Tower is located in"
    inputs = mt.tokenizer(test_prompt, return_tensors="pt").to(mt.device)
    model_inputs = {k: v for k, v in inputs.items() if k == 'input_ids'}

    test_layer = mt.num_layers // 2

    with ssm_nethook.trace_mamba_layer(mt.model, test_layer, 'mixer') as trace:
        with torch.no_grad():
            _ = mt.model(**model_inputs)
        hidden_states = trace.output

    assert hidden_states.shape[0] == 1
    assert hidden_states.shape[1] == inputs['input_ids'].shape[1]
    print(f"  Single layer hook: layer {test_layer}, shape={hidden_states.shape}")


def test_multiple_layer_hooks(mt):
    """Test hooking multiple layers."""
    from util_ssm import ssm_nethook

    test_prompt = "The Eiffel Tower is located in"
    inputs = mt.tokenizer(test_prompt, return_tensors="pt").to(mt.device)
    model_inputs = {k: v for k, v in inputs.items() if k == 'input_ids'}

    layer_specs = [
        (0, 'mixer'),
        (mt.num_layers // 2, 'mixer'),
        (mt.num_layers - 1, 'mixer'),
    ]

    with ssm_nethook.trace_multiple_mamba_layers(mt.model, layer_specs) as traces:
        with torch.no_grad():
            _ = mt.model(**model_inputs)

    for spec in layer_specs:
        layer_num, component = spec
        layer_name = ssm_nethook.mamba_layername(mt.model, layer_num, component)
        assert layer_name in traces
        assert traces[layer_name].output is not None

    print(f"  Multiple layer hooks: {len(layer_specs)} layers hooked successfully")


def run_all_tests():
    """Run all infrastructure tests."""
    print("=" * 70)
    print("INFRASTRUCTURE TESTS")
    print("=" * 70)
    print()

    print("Test 1: Imports")
    test_imports()
    print("  PASSED")
    print()

    print("Test 2: Model loading")
    mt = test_model_loading()
    print("  PASSED")
    print()

    print("Test 3: Layer structure")
    test_layer_structure(mt)
    print("  PASSED")
    print()

    print("Test 4: Architecture parsing")
    test_architecture_parsing(mt)
    print("  PASSED")
    print()

    print("Test 5: Forward pass")
    test_forward_pass(mt)
    print("  PASSED")
    print()

    print("Test 6: Single layer hook")
    test_single_layer_hook(mt)
    print("  PASSED")
    print()

    print("Test 7: Multiple layer hooks")
    test_multiple_layer_hooks(mt)
    print("  PASSED")
    print()

    print("=" * 70)
    print("ALL INFRASTRUCTURE TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
