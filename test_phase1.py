"""
Test script for Phase 1: Infrastructure Setup

This script tests that we can:
1. Load a Mamba model
2. Identify its layer structure
3. Hook layers and extract hidden states

Run this after setting up the conda environment and installing dependencies.

Usage:
    conda activate mamba-causal
    python test_phase1.py
"""

import torch
import sys
from pathlib import Path

# Add project directories to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("PHASE 1 TEST: Mamba Infrastructure Setup")
print("=" * 70)
print()

# Test 1: Import modules
print("Test 1: Importing modules...")
try:
    from mamba_causal_analysis.mamba_models import MambaModelAndTokenizer, load_mamba_model
    from util_ssm import ssm_nethook
    from util_ssm import mamba_layernames
    print("✓ All modules imported successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    print("\nMake sure you've installed dependencies:")
    print("  pip install mamba-ssm transformers torch")
    sys.exit(1)

print()

# Test 2: Load Mamba model
print("Test 2: Loading Mamba-130m...")
print("(This may take a minute on first run to download the model)")
try:
    mt = load_mamba_model(
        model_name="state-spaces/mamba-130m",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print(f"✓ Model loaded successfully")
    print(f"  Device: {mt.device}")
    print(f"  Layers: {mt.num_layers}")
    print(f"  Model: {mt.model_name}")
except Exception as e:
    print(f"✗ Model loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 3: Inspect layer structure
print("Test 3: Inspecting layer structure...")
try:
    print(f"  Total layers: {mt.num_layers}")
    print(f"  First layer: {mt.get_layer_name(0)}")
    print(f"  Last layer: {mt.get_layer_name(mt.num_layers - 1)}")
    print(f"  Example mixer: {mt.get_layer_name(0, 'mixer')}")
    print(f"  Embedding: {mt.get_embedding_layer_name()}")
    print(f"  LM head: {mt.get_lm_head_name()}")
    print("✓ Layer structure identified")
except Exception as e:
    print(f"✗ Layer inspection failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 4: Parse architecture
print("Test 4: Parsing architecture with mamba_layernames...")
try:
    arch_info = mamba_layernames.parse_mamba_architecture(mt.model)
    print(f"  Layer prefix: {arch_info['layer_prefix']}")
    print(f"  Has backbone: {arch_info['has_backbone']}")
    print(f"  Number of layers: {arch_info['num_layers']}")
    print(f"  Embedding name: {arch_info['embedding_name']}")
    print(f"  LM head name: {arch_info['lm_head_name']}")

    # Show first few layer names
    print(f"  First 3 layers: {arch_info['layer_names'][:3]}")
    print("✓ Architecture parsed successfully")
except Exception as e:
    print(f"✗ Architecture parsing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 5: Tokenize and run forward pass
print("Test 5: Running forward pass...")
try:
    test_prompt = "The Eiffel Tower is located in"
    inputs = mt.tokenizer(test_prompt, return_tensors="pt").to(mt.device)

    # Mamba doesn't use attention_mask (it's a state space model, not attention-based)
    model_inputs = {k: v for k, v in inputs.items() if k == 'input_ids'}

    with torch.no_grad():
        outputs = mt.model(**model_inputs)

    print(f"  Input: '{test_prompt}'")
    print(f"  Input shape: {inputs['input_ids'].shape}")
    print(f"  Output logits shape: {outputs.logits.shape}")

    # Get predicted token
    next_token_id = outputs.logits[0, -1].argmax().item()
    next_token = mt.tokenizer.decode([next_token_id])
    print(f"  Predicted next token: '{next_token}'")
    print("✓ Forward pass successful")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 6: Hook a layer and extract hidden states
print("Test 6: Hooking layer and extracting hidden states...")
try:
    test_layer = mt.num_layers // 2  # Middle layer

    with ssm_nethook.trace_mamba_layer(mt.model, test_layer, 'mixer') as trace:
        with torch.no_grad():
            outputs = mt.model(**model_inputs)
        hidden_states = trace.output

    print(f"  Hooked layer: {test_layer} (mixer)")
    print(f"  Hidden states shape: {hidden_states.shape}")
    print(f"  Hidden states dtype: {hidden_states.dtype}")
    print(f"  Hidden states device: {hidden_states.device}")
    print("✓ Layer hooking successful")
except Exception as e:
    print(f"✗ Layer hooking failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 7: Hook multiple layers
print("Test 7: Hooking multiple layers...")
try:
    layer_specs = [
        (0, 'mixer'),
        (mt.num_layers // 2, 'mixer'),
        (mt.num_layers - 1, 'mixer'),
    ]

    with ssm_nethook.trace_multiple_mamba_layers(mt.model, layer_specs) as traces:
        with torch.no_grad():
            outputs = mt.model(**model_inputs)

    print(f"  Hooked {len(layer_specs)} layers")
    for spec in layer_specs:
        layer_num, component = spec
        layer_name = ssm_nethook.mamba_layername(mt.model, layer_num, component)
        hidden = traces[layer_name].output
        print(f"    Layer {layer_num} ({component}): shape {hidden.shape}")

    print("✓ Multiple layer hooking successful")
except Exception as e:
    print(f"✗ Multiple layer hooking failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 8: Extract specific token representations
print("Test 8: Extracting specific token representations...")
try:
    # Get representation of last token from middle layer
    target_token_pos = inputs['input_ids'].shape[1] - 1

    with ssm_nethook.trace_mamba_layer(mt.model, test_layer, 'mixer') as trace:
        with torch.no_grad():
            outputs = mt.model(**model_inputs)
        hidden_states = trace.output

    # Extract last token
    last_token_repr = hidden_states[0, target_token_pos]

    print(f"  Token position: {target_token_pos}")
    print(f"  Token representation shape: {last_token_repr.shape}")
    print(f"  Representation norm: {last_token_repr.norm().item():.4f}")
    print("✓ Token representation extraction successful")
except Exception as e:
    print(f"✗ Token representation extraction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 70)
print("ALL TESTS PASSED! ✓")
print("=" * 70)
print()
print("Phase 1 is complete. You can now proceed to Phase 2:")
print("  - Implement mamba_repr_tools.py")
print("  - Implement mamba_causal_trace.py")
print("  - Create run_causal_trace.py experiment script")
print()
print("The infrastructure is working correctly:")
print("  ✓ Can load Mamba models")
print("  ✓ Can identify layer structure")
print("  ✓ Can hook layers")
print("  ✓ Can extract hidden states")
print("  ✓ Can extract specific token representations")
