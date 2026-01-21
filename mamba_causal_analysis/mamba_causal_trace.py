"""
Causal tracing for Mamba State Space Models.

Adapted from ROME's causal_trace.py methodology for Mamba's recurrent architecture.
Implements the core algorithm:
1. Run clean input and save hidden states
2. Corrupt input embeddings with noise
3. Selectively restore clean states at specific (layer, position) pairs
4. Measure effect on output predictions

Key difference from ROME: While transformers process all positions in parallel,
Mamba processes sequentially with recurrent state. We start by tracing hidden
states (mixer outputs) similar to ROME, leaving internal SSM state tracing for Phase 3.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from util_ssm import ssm_nethook


def trace_with_patch_mamba(
    model,
    input_ids: torch.Tensor,
    states_to_patch: List[Tuple[int, int]],  # List of (layer, token_position) pairs
    clean_states: Optional[torch.Tensor] = None,
    clean_input_ids: Optional[torch.Tensor] = None,
    tokens_to_mix: Optional[int] = None,
    noise_level: float = 3.0,
    uniform_noise: bool = False,
    component: str = "mixer",
):
    """
    Perform causal tracing on Mamba with selective state restoration.

    This implements the core ROME methodology adapted for Mamba:
    1. If clean_states not provided, run clean input to collect them
    2. Corrupt input embeddings with noise
    3. Run corrupted input while restoring clean states at specified locations
    4. Return model output for analysis

    Args:
        model: Mamba model instance
        input_ids: Input token IDs to trace [batch_size, seq_len]
        states_to_patch: List of (layer, position) tuples to restore clean states
        clean_states: Pre-computed clean states [batch_size, num_layers, seq_len, hidden_size]
                     If None, will run clean forward pass first
        clean_input_ids: Clean input_ids for computing clean states
                         Required if clean_states is None
        tokens_to_mix: Number of tokens to corrupt (from start). If None, corrupts all
        noise_level: Standard deviations of noise to add (default: 3.0 like ROME)
        uniform_noise: If True, use uniform noise instead of Gaussian
        component: Which component to patch ('mixer', 'norm', etc.)

    Returns:
        torch.Tensor: Model logits after patching [batch_size, seq_len, vocab_size]

    Example:
        >>> # Test effect of restoring layer 10 at position 5
        >>> clean_ids = tokenizer("The Eiffel Tower is", return_tensors="pt").input_ids
        >>> corrupt_ids = clean_ids.clone()  # We'll corrupt embeddings, not IDs
        >>> logits = trace_with_patch_mamba(
        ...     model,
        ...     corrupt_ids,
        ...     states_to_patch=[(10, 5)],
        ...     clean_input_ids=clean_ids,
        ...     noise_level=3.0
        ... )
    """
    # Handle model wrapper
    if hasattr(model, 'model'):
        raw_model = model.model
        device = model.device
    else:
        raw_model = model
        device = next(model.parameters()).device

    input_ids = input_ids.to(device)

    # Get unique layers we need (for indexing into clean_states)
    layers_needed = sorted(set(layer for layer, _ in states_to_patch))

    # Step 1: Get clean states if not provided
    if clean_states is None:
        if clean_input_ids is None:
            raise ValueError("Must provide either clean_states or clean_input_ids")

        clean_input_ids = clean_input_ids.to(device)

        # Run clean forward pass and collect states
        clean_states = collect_states(
            raw_model,
            clean_input_ids,
            layers=layers_needed,
            component=component,
        )

    # Step 2: Corrupt embeddings with noise
    # We'll use a hook to inject noise into the embedding layer
    embedding_layer_name = ssm_nethook.mamba_layername(raw_model, 0, "embed")
    embedding_layer = ssm_nethook.get_module(raw_model, embedding_layer_name)

    # Determine how many tokens to corrupt
    seq_len = input_ids.shape[1]
    if tokens_to_mix is None:
        tokens_to_mix = seq_len

    def corrupt_embeddings(output, layer_name):
        """Hook function to add noise to embeddings."""
        # output shape: [batch, seq_len, embed_size]
        if isinstance(output, tuple):
            embeddings = output[0]
        else:
            embeddings = output

        # Create noise
        if uniform_noise:
            noise = torch.randn_like(embeddings[:, :tokens_to_mix]) * noise_level
        else:
            noise = torch.randn_like(embeddings[:, :tokens_to_mix]) * (
                noise_level * embeddings[:, :tokens_to_mix].std()
            )

        # Add noise to first N tokens
        corrupted = embeddings.clone()
        corrupted[:, :tokens_to_mix] += noise

        if isinstance(output, tuple):
            return (corrupted,) + output[1:]
        return corrupted

    # Step 3: Set up hooks to restore clean states at specified locations
    patch_specs = {}  # Map layer_name -> dict of positions to patch

    for layer_num, position in states_to_patch:
        layer_name = ssm_nethook.mamba_layername(raw_model, layer_num, component)

        if layer_name not in patch_specs:
            patch_specs[layer_name] = {}

        # Store clean state for this position
        # clean_states shape: [batch, num_layers, seq_len, hidden_size]
        # Need to map layer_num to index in clean_states
        layer_idx = layers_needed.index(layer_num)
        patch_specs[layer_name][position] = clean_states[:, layer_idx, position, :]

    def make_patch_hook(layer_name):
        """Create a hook function that patches specific positions."""
        positions_to_patch = patch_specs[layer_name]

        def patch_hook(output, layer):
            # Handle tuple output
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
            else:
                hidden_states = output
                rest = ()

            # Patch specified positions
            patched = hidden_states.clone()
            for pos, clean_state in positions_to_patch.items():
                # clean_state shape: [batch, hidden_size]
                # Expand to match if needed
                patched[:, pos, :] = clean_state

            if rest:
                return (patched,) + rest
            return patched

        return patch_hook

    # Step 4: Run forward pass with noise corruption and state patching
    with torch.no_grad():
        with ssm_nethook.TraceDict(
            raw_model,
            layers=[embedding_layer_name] + list(patch_specs.keys()),
            edit_output=lambda output, layer: (
                corrupt_embeddings(output, layer)
                if layer == embedding_layer_name
                else make_patch_hook(layer)(output, layer)
            ),
        ) as trace:
            # Run forward pass
            outputs = raw_model(input_ids)

    # Return logits
    if hasattr(outputs, 'logits'):
        return outputs.logits
    return outputs


def collect_states(
    model,
    input_ids: torch.Tensor,
    layers: List[int],
    component: str = "mixer",
) -> torch.Tensor:
    """
    Collect hidden states from specified layers during a forward pass.

    Args:
        model: Mamba model (raw, not wrapped)
        input_ids: Input token IDs [batch_size, seq_len]
        layers: List of layer numbers to collect from
        component: Component to collect from

    Returns:
        torch.Tensor: States [batch_size, num_layers, seq_len, hidden_size]
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    layer_names = [
        ssm_nethook.mamba_layername(model, layer_num, component)
        for layer_num in layers
    ]

    with torch.no_grad():
        with ssm_nethook.TraceDict(
            model, layer_names, retain_output=True
        ) as traces:
            _ = model(input_ids)

            # Collect states from all layers
            all_states = []
            for layer_name in layer_names:
                output = traces[layer_name].output

                # Handle tuple output
                if isinstance(output, tuple):
                    output = output[0]

                # output shape: [batch, seq_len, hidden]
                all_states.append(output)

            # Stack: [batch, num_layers, seq_len, hidden]
            all_states = torch.stack(all_states, dim=1)

    return all_states


def calculate_hidden_flow(
    model,
    tokenizer,
    prompt: str,
    subject: str,
    samples: int = 10,
    noise_level: float = 3.0,
    component: str = "mixer",
    batch_size: int = 1,
) -> Dict[str, Any]:
    """
    Calculate information flow for causal tracing analysis.

    This is the main function for generating heatmap data, similar to ROME's
    calculate_hidden_flow(). It:
    1. Runs clean input
    2. For each (layer, position) pair:
        a. Corrupts embeddings
        b. Restores clean state at that position
        c. Measures effect on target token probability
    3. Returns results as a matrix suitable for heatmap visualization

    Args:
        model: Mamba model instance
        tokenizer: Tokenizer
        prompt: Text prompt (e.g., "The Eiffel Tower is located in")
        subject: Subject phrase to locate (e.g., "Eiffel Tower")
        samples: Number of noise samples to average over
        noise_level: Standard deviations of noise
        component: Component to trace
        batch_size: Batch size for processing

    Returns:
        Dict containing:
        - 'scores': np.array [num_layers, seq_len] of probability differences
        - 'low_score': Baseline corrupted probability
        - 'high_score': Clean probability
        - 'input_ids': Token IDs
        - 'input_tokens': List of token strings
        - 'subject_range': (start, end) token indices for subject
        - 'target_token_id': ID of target token (next token after prompt)

    Example:
        >>> result = calculate_hidden_flow(
        ...     mt, mt.tokenizer,
        ...     prompt="The Eiffel Tower is located in",
        ...     subject="Eiffel Tower",
        ...     samples=10
        ... )
        >>> # result['scores'] is ready for heatmap plotting
    """
    # Handle model wrapper
    if hasattr(model, 'model'):
        raw_model = model.model
        device = model.device
    else:
        raw_model = model
        device = next(model.parameters()).device

    # Tokenize prompt
    tokens = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = tokens["input_ids"]

    batch_size_actual = input_ids.shape[0]
    seq_len = input_ids.shape[1]

    # Find subject token range
    from . import mamba_repr_tools
    try:
        subject_start, subject_end = mamba_repr_tools.find_token_range(
            tokenizer, prompt, subject
        )
    except ValueError:
        # If subject not found, use middle tokens as fallback
        subject_start = seq_len // 3
        subject_end = 2 * seq_len // 3

    # Get number of layers
    if hasattr(model, 'num_layers'):
        num_layers = model.num_layers
    else:
        # Count layers in model
        num_layers = len([
            name for name, _ in raw_model.named_modules()
            if 'backbone.layers.' in name and '.mixer' in name
        ])

    # Run clean forward pass to get target token
    with torch.no_grad():
        clean_outputs = raw_model(input_ids)
        if hasattr(clean_outputs, 'logits'):
            clean_logits = clean_outputs.logits
        else:
            clean_logits = clean_outputs

        # Get probability of target token (next token after prompt)
        clean_probs = torch.softmax(clean_logits[0, -1, :], dim=0)
        target_token_id = clean_probs.argmax().item()
        high_score = clean_probs[target_token_id].item()

    # Collect clean states for all layers
    all_layers = list(range(num_layers))
    clean_states = collect_states(raw_model, input_ids, all_layers, component)

    # Initialize score matrix: [num_layers, seq_len]
    scores = np.zeros((num_layers, seq_len))

    # Run corrupted baseline (no restoration)
    corrupted_probs_samples = []
    for _ in range(samples):
        with torch.no_grad():
            logits = trace_with_patch_mamba(
                raw_model,
                input_ids,
                states_to_patch=[],  # No patching
                clean_states=clean_states,
                noise_level=noise_level,
                component=component,
            )
            probs = torch.softmax(logits[0, -1, :], dim=0)
            corrupted_probs_samples.append(probs[target_token_id].item())

    low_score = np.mean(corrupted_probs_samples)

    # For each (layer, position), restore and measure effect
    print(f"Tracing {num_layers} layers x {seq_len} positions...")

    for layer in range(num_layers):
        for position in range(seq_len):
            position_probs = []

            for _ in range(samples):
                with torch.no_grad():
                    logits = trace_with_patch_mamba(
                        raw_model,
                        input_ids,
                        states_to_patch=[(layer, position)],
                        clean_states=clean_states,
                        noise_level=noise_level,
                        component=component,
                    )
                    probs = torch.softmax(logits[0, -1, :], dim=0)
                    position_probs.append(probs[target_token_id].item())

            # Average across samples
            scores[layer, position] = np.mean(position_probs)

        if (layer + 1) % 5 == 0:
            print(f"  Completed layer {layer + 1}/{num_layers}")

    # Get token strings for visualization
    input_tokens = [
        tokenizer.decode([token_id]) for token_id in input_ids[0].tolist()
    ]

    return {
        'scores': scores,
        'low_score': low_score,
        'high_score': high_score,
        'input_ids': input_ids.cpu(),
        'input_tokens': input_tokens,
        'subject_range': (subject_start, subject_end),
        'target_token_id': target_token_id,
        'target_token': tokenizer.decode([target_token_id]),
    }
