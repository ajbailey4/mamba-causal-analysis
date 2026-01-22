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
    patch_entire_layer: bool = True,  # NEW: If True, patch entire layer instead of per-position
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
                        If patch_entire_layer=True, position is ignored
        clean_states: Pre-computed clean states [batch_size, num_layers, seq_len, hidden_size]
                     If None, will run clean forward pass first
        clean_input_ids: Clean input_ids for computing clean states
                         Required if clean_states is None
        tokens_to_mix: Number of tokens to corrupt (from start). If None, corrupts all
        noise_level: Standard deviations of noise to add (default: 3.0 like ROME)
        uniform_noise: If True, use uniform noise instead of Gaussian
        component: Which component to patch ('mixer', 'norm', etc.)
        patch_entire_layer: If True, restore ALL positions in a layer (appropriate for Mamba's
                           fused SSM computation). If False, restore only specific positions
                           (less appropriate but matches ROME's approach).

    Returns:
        torch.Tensor: Model logits after patching [batch_size, seq_len, vocab_size]

    Example:
        >>> # Test effect of restoring layer 10 (entire layer)
        >>> clean_ids = tokenizer("The Eiffel Tower is", return_tensors="pt").input_ids
        >>> logits = trace_with_patch_mamba(
        ...     model,
        ...     clean_ids,
        ...     states_to_patch=[(10, 0)],  # position ignored when patch_entire_layer=True
        ...     clean_input_ids=clean_ids,
        ...     noise_level=3.0,
        ...     patch_entire_layer=True
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
    patch_specs = {}  # Map layer_name -> clean states to patch
    patch_specs_residuals = {}  # For full blocks, also store residuals

    # Check if clean_states is a tuple (for full blocks)
    if isinstance(clean_states, tuple):
        clean_hidden_states, clean_residuals = clean_states
    else:
        clean_hidden_states = clean_states
        clean_residuals = None

    if patch_entire_layer:
        # Per-layer patching: restore entire layer output
        for layer_num, _ in states_to_patch:
            layer_name = ssm_nethook.mamba_layername(raw_model, layer_num, component)

            if layer_name not in patch_specs:
                # Store clean states for ALL positions in this layer
                # clean_hidden_states shape: [batch, num_layers, seq_len, hidden_size]
                layer_idx = layers_needed.index(layer_num)
                patch_specs[layer_name] = clean_hidden_states[:, layer_idx, :, :]  # [batch, seq_len, hidden]

                # If we have residuals (full block mode), store those too
                if clean_residuals is not None:
                    patch_specs_residuals[layer_name] = clean_residuals[:, layer_idx, :, :]
    else:
        # Per-position patching: restore specific (layer, position) pairs
        patch_specs_positions = {}  # Map layer_name -> dict of positions to patch
        patch_specs_positions_residuals = {}  # Map layer_name -> dict of positions for residuals

        for layer_num, position in states_to_patch:
            layer_name = ssm_nethook.mamba_layername(raw_model, layer_num, component)

            if layer_name not in patch_specs_positions:
                patch_specs_positions[layer_name] = {}
                if clean_residuals is not None:
                    patch_specs_positions_residuals[layer_name] = {}

            # Store clean state for this position
            layer_idx = layers_needed.index(layer_num)
            patch_specs_positions[layer_name][position] = clean_hidden_states[:, layer_idx, position, :]

            # Store clean residual for this position if available
            if clean_residuals is not None:
                patch_specs_positions_residuals[layer_name][position] = clean_residuals[:, layer_idx, position, :]

        patch_specs = patch_specs_positions

    def make_patch_hook(layer_name):
        """Create a hook function that patches states."""

        if patch_entire_layer:
            # Restore entire layer
            clean_layer_states = patch_specs[layer_name]
            clean_layer_residual = patch_specs_residuals.get(layer_name, None)

            def patch_hook(output, layer):
                # Handle tuple output (for full blocks)
                if isinstance(output, tuple):
                    hidden_states = output[0]
                    residual = output[1] if len(output) > 1 else None
                    rest = output[2:] if len(output) > 2 else ()

                    # Replace hidden states with clean
                    patched_hidden = clean_layer_states.clone()

                    # If we have clean residual, replace that too
                    if clean_layer_residual is not None and residual is not None:
                        patched_residual = clean_layer_residual.clone()
                        return (patched_hidden, patched_residual) + rest
                    else:
                        return (patched_hidden, residual) + rest
                else:
                    # Single tensor output
                    patched = clean_layer_states.clone()
                    return patched

            return patch_hook
        else:
            # Restore specific positions
            positions_to_patch = patch_specs[layer_name]
            positions_to_patch_residuals = patch_specs_positions_residuals.get(layer_name, {})

            def patch_hook(output, layer):
                # Handle tuple output
                if isinstance(output, tuple):
                    hidden_states = output[0]
                    residual = output[1] if len(output) > 1 else None
                    rest = output[2:] if len(output) > 2 else ()

                    # Patch specified positions in hidden_states
                    patched_hidden = hidden_states.clone()
                    for pos, clean_state in positions_to_patch.items():
                        # clean_state shape: [batch, hidden_size]
                        patched_hidden[:, pos, :] = clean_state

                    # Patch specified positions in residual if available
                    if residual is not None and positions_to_patch_residuals:
                        patched_residual = residual.clone()
                        for pos, clean_residual in positions_to_patch_residuals.items():
                            patched_residual[:, pos, :] = clean_residual
                        return (patched_hidden, patched_residual) + rest
                    else:
                        return (patched_hidden, residual) + rest
                else:
                    # Single tensor output - patch positions
                    patched = output.clone()
                    for pos, clean_state in positions_to_patch.items():
                        patched[:, pos, :] = clean_state
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
        component: Component to collect from (None for full block)

    Returns:
        If component is None (full block): tuple of (hidden_states, residuals)
            - hidden_states: [batch_size, num_layers, seq_len, hidden_size]
            - residuals: [batch_size, num_layers, seq_len, hidden_size]
        Otherwise: torch.Tensor [batch_size, num_layers, seq_len, hidden_size]
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
            all_residuals = []
            for layer_name in layer_names:
                output = traces[layer_name].output

                # Handle tuple output (for full blocks)
                if isinstance(output, tuple) and component is None:
                    # Full block returns (hidden_states, residual)
                    hidden_states, residual = output
                    all_states.append(hidden_states)
                    all_residuals.append(residual if residual is not None else torch.zeros_like(hidden_states))
                elif isinstance(output, tuple):
                    # Component returns tuple, take first element
                    all_states.append(output[0])
                else:
                    # Single tensor output
                    all_states.append(output)

            # Stack: [batch, num_layers, seq_len, hidden]
            all_states = torch.stack(all_states, dim=1)

            if component is None and all_residuals:
                # Return both hidden states and residuals for full block
                all_residuals = torch.stack(all_residuals, dim=1)
                return (all_states, all_residuals)

    return all_states


def trace_vertical_restoration(
    model,
    tokenizer,
    prompt: str,
    subject: str,
    position: int,  # Position to restore across all layers
    samples: int = 10,
    noise_level: float = 3.0,
    component: str = None,
) -> float:
    """
    Restore a specific position across ALL layers (vertical restoration).

    This tests: "What if we restore position P in every layer?"
    This is a cross-layer intervention at a single time step.

    Args:
        model: Mamba model instance
        tokenizer: Tokenizer
        prompt: Text prompt
        subject: Subject phrase to locate
        position: Token position to restore in all layers
        samples: Number of noise samples
        noise_level: Standard deviations of noise
        component: Component to trace (None for full block)

    Returns:
        Average probability of target token across samples
    """
    # Handle model wrapper
    if hasattr(model, 'model'):
        raw_model = model.model
        device = model.device
    else:
        raw_model = model
        device = next(model.parameters()).device

    # Tokenize
    tokens = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = tokens["input_ids"]
    seq_len = input_ids.shape[1]

    # Find subject range
    from . import mamba_repr_tools
    try:
        subject_start, subject_end = mamba_repr_tools.find_token_range(
            tokenizer, prompt, subject
        )
    except ValueError:
        subject_start = seq_len // 3
        subject_end = 2 * seq_len // 3

    tokens_to_mix = subject_end

    # Get number of layers
    if hasattr(model, 'num_layers'):
        num_layers = model.num_layers
    else:
        num_layers = len([
            name for name, _ in raw_model.named_modules()
            if 'backbone.layers.' in name and '.mixer' in name
        ])

    # Get target token from clean run
    with torch.no_grad():
        clean_outputs = raw_model(input_ids)
        if hasattr(clean_outputs, 'logits'):
            clean_logits = clean_outputs.logits
        else:
            clean_logits = clean_outputs
        clean_probs = torch.softmax(clean_logits[0, -1, :], dim=0)
        target_token_id = clean_probs.argmax().item()

    # Collect clean states for all layers
    all_layers = list(range(num_layers))
    clean_states = collect_states(raw_model, input_ids, all_layers, component)

    # Create states_to_patch: restore this position in ALL layers
    states_to_patch = [(layer, position) for layer in range(num_layers)]

    # Run samples with vertical restoration
    probs_samples = []
    for _ in range(samples):
        with torch.no_grad():
            logits = trace_with_patch_mamba(
                raw_model,
                input_ids,
                states_to_patch=states_to_patch,
                clean_states=clean_states,
                tokens_to_mix=tokens_to_mix,
                noise_level=noise_level,
                component=component,
                patch_entire_layer=False,  # Must be False for per-position
            )
            probs = torch.softmax(logits[0, -1, :], dim=0)
            probs_samples.append(probs[target_token_id].item())

    return np.mean(probs_samples)


def calculate_hidden_flow(
    model,
    tokenizer,
    prompt: str,
    subject: str,
    samples: int = 10,
    noise_level: float = 3.0,
    component: str = None,  # None means patch after residual (default)
    batch_size: int = 1,
    patch_entire_layer: bool = True,  # Use per-layer patching by default
    mode: str = None,  # None (auto), "per_layer", "per_position", "per_position_ssm", "vertical"
) -> Dict[str, Any]:
    """
    Calculate information flow for causal tracing analysis.

    This is the main function for generating heatmap data, similar to ROME's
    calculate_hidden_flow(). It:
    1. Runs clean input
    2. For each layer (or layer,position pair):
        a. Corrupts embeddings
        b. Restores clean state for that layer/position
        c. Measures effect on target token probability
    3. Returns results as a matrix suitable for visualization

    Args:
        model: Mamba model instance
        tokenizer: Tokenizer
        prompt: Text prompt (e.g., "The Eiffel Tower is located in")
        subject: Subject phrase to locate (e.g., "Eiffel Tower")
        samples: Number of noise samples to average over
        noise_level: Standard deviations of noise
        component: Component to trace (for hidden state modes)
        batch_size: Batch size for processing
        patch_entire_layer: If True, restore entire layers. If False, per-position.
        mode: Tracing mode:
            - None (auto): Use patch_entire_layer to decide
            - "per_layer": Test each layer (1D output)
            - "per_position": Test each (layer, position) with hidden states (2D output)
            - "per_position_ssm": Test each (layer, position) with SSM states (2D output, slow)

    Returns:
        Dict containing:
        - 'scores': np.array [num_layers], [num_layers, seq_len], or [seq_len] depending on mode
        - 'low_score': Baseline corrupted probability
        - 'high_score': Clean probability
        - 'input_ids': Token IDs
        - 'input_tokens': List of token strings
        - 'subject_range': (start, end) token indices for subject
        - 'target_token_id': ID of target token (next token after prompt)
        - 'mode': Mode used for tracing

    Examples:
        >>> # Per-layer mode (fast, default) - horizontal restoration
        >>> result = calculate_hidden_flow(
        ...     mt, mt.tokenizer,
        ...     prompt="The Eiffel Tower is located in",
        ...     subject="Eiffel Tower",
        ...     samples=10,
        ...     mode="per_layer"
        ... )
        >>> # result['scores'] has shape [num_layers]

        >>> # Vertical restoration mode - restore all layers at each position
        >>> result = calculate_hidden_flow(
        ...     mt, mt.tokenizer,
        ...     prompt="The Eiffel Tower is located in",
        ...     subject="Eiffel Tower",
        ...     samples=10,
        ...     mode="vertical"
        ... )
        >>> # result['scores'] has shape [seq_len]

        >>> # Per-position SSM state mode (slow, most detailed)
        >>> result = calculate_hidden_flow(
        ...     mt, mt.tokenizer,
        ...     prompt="The Eiffel Tower is located in",
        ...     subject="Eiffel Tower",
        ...     samples=10,
        ...     mode="per_position_ssm"
        ... )
        >>> # result['scores'] has shape [num_layers, seq_len]
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

    # Determine tokens to corrupt (subject tokens only, following ROME)
    tokens_to_mix = subject_end

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

    # Determine mode and set patch_entire_layer accordingly
    if mode is None:
        mode = "per_layer" if patch_entire_layer else "per_position"
    elif mode == "per_layer":
        patch_entire_layer = True
    elif mode == "per_position":
        patch_entire_layer = False
    elif mode == "vertical":
        patch_entire_layer = False  # Vertical uses per-position patching
    # per_position_ssm doesn't use patch_entire_layer

    # Collect clean states for all layers
    all_layers = list(range(num_layers))

    if mode == "vertical":
        # Vertical mode: test each position (restoring ALL layers at that position)
        clean_states = collect_states(raw_model, input_ids, all_layers, component)
        scores = np.zeros(seq_len)  # 1D: just positions
        print(f"Tracing {seq_len} positions (vertical restoration - all layers at each position)...")
    elif mode == "per_position_ssm":
        # Use sequential SSM state collection
        print(f"Collecting clean SSM states (token-by-token, slow)...")
        conv_clean, ssm_clean = collect_ssm_states_sequential(raw_model, input_ids, all_layers)
        clean_states = (conv_clean, ssm_clean)
        scores = np.zeros((num_layers, seq_len))  # 2D: layers × positions
        print(f"Tracing {num_layers} layers x {seq_len} positions (per-position SSM state mode)...")
    else:
        # Use regular state collection (hooks)
        clean_states = collect_states(raw_model, input_ids, all_layers, component)

        # Initialize score matrix
        if mode == "per_layer":
            scores = np.zeros(num_layers)  # 1D: just layers
            print(f"Tracing {num_layers} layers (per-layer mode)...")
        else:  # mode == "per_position"
            scores = np.zeros((num_layers, seq_len))  # 2D: layers × positions
            print(f"Tracing {num_layers} layers x {seq_len} positions (per-position mode)...")

    # Run corrupted baseline (no restoration)
    corrupted_probs_samples = []
    for _ in range(samples):
        with torch.no_grad():
            logits = trace_with_patch_mamba(
                raw_model,
                input_ids,
                states_to_patch=[],  # No patching
                clean_states=clean_states,
                tokens_to_mix=tokens_to_mix,  # Only corrupt subject tokens
                noise_level=noise_level,
                component=component,
                patch_entire_layer=patch_entire_layer,
            )
            probs = torch.softmax(logits[0, -1, :], dim=0)
            corrupted_probs_samples.append(probs[target_token_id].item())

    low_score = np.mean(corrupted_probs_samples)

    # Trace each layer (and optionally each position)
    if mode == "vertical":
        # Vertical restoration: restore all layers at each position
        for position in range(seq_len):
            position_probs = []

            # Create states_to_patch for this position across all layers
            states_to_patch = [(layer, position) for layer in range(num_layers)]

            for _ in range(samples):
                with torch.no_grad():
                    logits = trace_with_patch_mamba(
                        raw_model,
                        input_ids,
                        states_to_patch=states_to_patch,
                        clean_states=clean_states,
                        tokens_to_mix=tokens_to_mix,
                        noise_level=noise_level,
                        component=component,
                        patch_entire_layer=False,  # Must use per-position for vertical
                    )
                    probs = torch.softmax(logits[0, -1, :], dim=0)
                    position_probs.append(probs[target_token_id].item())

            # Average across samples
            scores[position] = np.mean(position_probs)

            if (position + 1) % 2 == 0:
                print(f"  Completed position {position + 1}/{seq_len}")
    elif mode == "per_position_ssm":
        # Per-position SSM state mode: restore SSM states using token-by-token
        # First get corrupted baseline
        corrupted_probs_samples_ssm = []
        for _ in range(samples):
            with torch.no_grad():
                # For baseline, just corrupt and run without state restoration
                logits = trace_with_ssm_state_patch_sequential(
                    raw_model,
                    input_ids,
                    layer_to_patch=-1,  # Invalid layer, no patching
                    position_to_patch=-1,
                    clean_states=clean_states,
                    tokens_to_mix=tokens_to_mix,  # Only corrupt subject tokens
                    noise_level=noise_level,
                )
                probs = torch.softmax(logits[0, -1, :], dim=0)
                corrupted_probs_samples_ssm.append(probs[target_token_id].item())

        low_score_ssm = np.mean(corrupted_probs_samples_ssm)

        # Trace each (layer, position) pair
        for layer in range(num_layers):
            for position in range(seq_len):
                position_probs = []

                for _ in range(samples):
                    with torch.no_grad():
                        logits = trace_with_ssm_state_patch_sequential(
                            raw_model,
                            input_ids,
                            layer_to_patch=layer,
                            position_to_patch=position,
                            clean_states=clean_states,
                            tokens_to_mix=tokens_to_mix,  # Only corrupt subject tokens
                            noise_level=noise_level,
                        )
                        probs = torch.softmax(logits[0, -1, :], dim=0)
                        position_probs.append(probs[target_token_id].item())

                # Average across samples
                scores[layer, position] = np.mean(position_probs)

            if (layer + 1) % 5 == 0:
                print(f"  Completed layer {layer + 1}/{num_layers}")
    elif patch_entire_layer:
        # Per-layer mode: restore entire layers
        for layer in range(num_layers):
            layer_probs = []

            for _ in range(samples):
                with torch.no_grad():
                    logits = trace_with_patch_mamba(
                        raw_model,
                        input_ids,
                        states_to_patch=[(layer, 0)],  # position ignored
                        clean_states=clean_states,
                        tokens_to_mix=tokens_to_mix,  # Only corrupt subject tokens
                        noise_level=noise_level,
                        component=component,
                        patch_entire_layer=True,
                    )
                    probs = torch.softmax(logits[0, -1, :], dim=0)
                    layer_probs.append(probs[target_token_id].item())

            # Average across samples
            scores[layer] = np.mean(layer_probs)

            if (layer + 1) % 5 == 0:
                print(f"  Completed layer {layer + 1}/{num_layers}")
    else:
        # Per-position mode: restore individual (layer, position) pairs
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
                            tokens_to_mix=tokens_to_mix,  # Only corrupt subject tokens
                            noise_level=noise_level,
                            component=component,
                            patch_entire_layer=False,
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
        'mode': mode,
        'patch_entire_layer': patch_entire_layer,
        'prompt': prompt,
    }


def collect_ssm_states_sequential(
    model,
    input_ids: torch.Tensor,
    layers: List[int],
) -> Tuple[Dict[int, Dict[int, torch.Tensor]], Dict[int, Dict[int, torch.Tensor]]]:
    """
    Collect SSM states (conv_state, ssm_state) at all positions using token-by-token processing.

    This uses Mamba's inference mode to process tokens sequentially and save the internal
    SSM states after each token.

    Args:
        model: Mamba model (raw, not wrapped)
        input_ids: Input token IDs [batch_size, seq_len]
        layers: List of layer indices to collect states from

    Returns:
        Tuple of (conv_states, ssm_states) where each is:
            {layer_idx: {position: state_tensor}}
        conv_states: {layer: {pos: [batch, d_inner, d_conv]}}
        ssm_states: {layer: {pos: [batch, d_inner, d_state]}}
    """
    from mamba_ssm.utils.generation import InferenceParams

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    batch_size, seq_len = input_ids.shape

    # Initialize storage
    conv_states_all = {layer: {} for layer in layers}
    ssm_states_all = {layer: {} for layer in layers}

    # Allocate inference cache
    inference_params = InferenceParams(
        max_seqlen=seq_len,
        max_batch_size=batch_size,
    )

    # Allocate cache for all layers
    cache = model.allocate_inference_cache(batch_size, max_seqlen=seq_len)
    inference_params.key_value_memory_dict = cache

    # Process tokens one by one
    for position in range(seq_len):
        # Get single token
        token = input_ids[:, position:position+1]

        # Forward pass (updates states in cache)
        with torch.no_grad():
            _ = model(token, inference_params=inference_params)

        # Save states after this token for requested layers
        for layer_idx in layers:
            if layer_idx in cache:
                conv_state, ssm_state = cache[layer_idx]
                # Clone to save snapshot
                conv_states_all[layer_idx][position] = conv_state.clone()
                ssm_states_all[layer_idx][position] = ssm_state.clone()

        # Update offset for next token
        inference_params.seqlen_offset += 1

    return conv_states_all, ssm_states_all


def trace_with_ssm_state_patch_sequential(
    model,
    input_ids: torch.Tensor,
    layer_to_patch: int,
    position_to_patch: int,
    clean_states: Tuple[Dict, Dict],  # (conv_states, ssm_states)
    tokens_to_mix: int = None,  # Number of tokens to corrupt (subject tokens)
    noise_level: float = 3.0,
) -> torch.Tensor:
    """
    Patch SSM state at (layer, position) using token-by-token processing.

    This corrupts subject embeddings and processes the sequence token-by-token, injecting
    clean SSM states at the specified (layer, position) pair.

    Args:
        model: Mamba model (raw, not wrapped)
        input_ids: Input token IDs [batch_size, seq_len]
        layer_to_patch: Layer index to patch
        position_to_patch: Token position to patch
        clean_states: Tuple of (conv_states_dict, ssm_states_dict) from collect_ssm_states_sequential
        tokens_to_mix: Number of tokens to corrupt (from start). If None, corrupts all.
        noise_level: Standard deviations of noise to add to embeddings

    Returns:
        torch.Tensor: Model logits [batch_size, seq_len, vocab_size]
    """
    from mamba_ssm.utils.generation import InferenceParams

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    batch_size, seq_len = input_ids.shape

    conv_clean, ssm_clean = clean_states

    # Determine how many tokens to corrupt
    if tokens_to_mix is None:
        tokens_to_mix = seq_len

    # Get embeddings and corrupt subject tokens only
    embedding_layer = model.backbone.embedding
    with torch.no_grad():
        clean_embeds = embedding_layer(input_ids)
        noise = torch.randn_like(clean_embeds[:, :tokens_to_mix]) * noise_level * clean_embeds[:, :tokens_to_mix].std()
        corrupted_embeds = clean_embeds.clone()
        corrupted_embeds[:, :tokens_to_mix] += noise

    # Initialize inference
    inference_params = InferenceParams(
        max_seqlen=seq_len,
        max_batch_size=batch_size,
    )

    # Allocate cache
    cache = model.allocate_inference_cache(batch_size, max_seqlen=seq_len)
    inference_params.key_value_memory_dict = cache

    # Process token-by-token with manual embedding input
    for position in range(seq_len):
        # Check if we should inject clean state at this position
        if position == position_to_patch and layer_to_patch in cache:
            # Inject clean SSM states
            cache[layer_to_patch] = (
                conv_clean[layer_to_patch][position].clone(),
                ssm_clean[layer_to_patch][position].clone()
            )

        # Get corrupted embedding for this token
        token_embed = corrupted_embeds[:, position:position+1, :]

        # Process through backbone layers
        hidden_states = token_embed
        residual = None

        # Forward through layers with inference_params
        for layer_idx, layer in enumerate(model.backbone.layers):
            hidden_states, residual = layer(
                hidden_states,
                residual,
                inference_params=inference_params
            )

        # Update offset
        inference_params.seqlen_offset += 1

    # Apply final norm and LM head
    with torch.no_grad():
        if residual is not None:
            hidden_states = hidden_states + residual
        hidden_states = model.backbone.norm_f(hidden_states)
        logits = model.lm_head(hidden_states)

    return logits
