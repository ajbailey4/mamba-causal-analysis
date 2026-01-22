"""
Causal tracing for Mamba State Space Models - Refactored Version.

Simplified API with two orthogonal parameters:
- component: What to restore ('mixer_output' or 'recurrent_state')
- granularity: How to restore ('per_layer' or 'per_position')

This gives 4 modes:
1. component='mixer_output', granularity='per_layer'
2. component='mixer_output', granularity='per_position'
3. component='recurrent_state', granularity='per_layer'
4. component='recurrent_state', granularity='per_position'
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from util_ssm import ssm_nethook


def trace_important_states(
    model,
    tokenizer,
    prompt: str,
    subject: str,
    component: str = 'mixer_output',
    granularity: str = 'per_position',
    samples: int = 10,
    noise_level: float = 3.0,
) -> Dict[str, Any]:
    """
    Main entry point for causal tracing on Mamba models.

    Args:
        model: Mamba model instance
        tokenizer: Tokenizer
        prompt: Text prompt (e.g., "The Eiffel Tower is located in")
        subject: Subject phrase to corrupt (e.g., "Eiffel Tower")
        component: What to restore:
            - 'mixer_output': Restore mixer outputs (what goes into residual stream)
            - 'recurrent_state': Restore SSM internal cache (conv_state, ssm_state)
        granularity: How to restore:
            - 'per_layer': Restore all positions in each layer
            - 'per_position': Restore specific (layer, position) pairs
        samples: Number of noise samples to average over
        noise_level: Standard deviations of noise (default: 3.0 like ROME)

    Returns:
        dict with keys:
            - 'scores': numpy array of restoration effects
                - per_layer: shape (num_layers,)
                - per_position: shape (num_layers, seq_len)
            - 'high_score': clean baseline probability
            - 'low_score': corrupted baseline probability
            - 'input_tokens': list of token strings
            - 'subject_range': (start, end) indices of subject
            - 'target_token': predicted token string
            - 'prompt': original prompt
            - 'component': component used
            - 'granularity': granularity used
    """
    # Validate parameters
    if component not in ['mixer_output', 'recurrent_state']:
        raise ValueError(f"component must be 'mixer_output' or 'recurrent_state', got '{component}'")
    if granularity not in ['per_layer', 'per_position']:
        raise ValueError(f"granularity must be 'per_layer' or 'per_position', got '{granularity}'")

    # Get model device
    if hasattr(model, 'model'):
        raw_model = model.model
        device = model.device
    else:
        raw_model = model
        device = next(model.parameters()).device

    # Tokenize and get number of layers
    tokens = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = tokens["input_ids"]
    seq_len = input_ids.shape[1]
    num_layers = _get_num_layers(model, raw_model)

    # Find subject token range
    from . import mamba_repr_tools
    try:
        subject_start, subject_end = mamba_repr_tools.find_token_range(
            tokenizer, prompt, subject
        )
    except ValueError:
        # Fallback: use middle third
        subject_start = seq_len // 3
        subject_end = 2 * seq_len // 3

    # Get clean baseline and target
    with torch.no_grad():
        clean_logits = raw_model(input_ids)
        if hasattr(clean_logits, 'logits'):
            clean_logits = clean_logits.logits

        clean_probs = torch.softmax(clean_logits[0, -1, :], dim=0)
        target_token_id = clean_probs.argmax().item()
        high_score = clean_probs[target_token_id].item()
        target_token = tokenizer.decode([target_token_id])

    # Get input token strings
    input_tokens = [tokenizer.decode([t]) for t in input_ids[0]]

    print(f"Causal Tracing: component={component}, granularity={granularity}")
    print(f"  Clean probability: {high_score:.4f} for '{target_token}'")
    print(f"  Subject: '{subject}' at positions {subject_start}-{subject_end}")

    # Route to appropriate tracing function
    if component == 'mixer_output':
        if granularity == 'per_layer':
            scores, low_score = _trace_mixer_output_per_layer(
                raw_model, input_ids, subject_end, num_layers,
                samples, noise_level, target_token_id
            )
        else:  # per_position
            scores, low_score = _trace_mixer_output_per_position(
                raw_model, input_ids, subject_end, num_layers, seq_len,
                samples, noise_level, target_token_id
            )
    else:  # recurrent_state
        if granularity == 'per_layer':
            scores, low_score = _trace_recurrent_state_per_layer(
                raw_model, input_ids, subject_end, num_layers,
                samples, noise_level, target_token_id
            )
        else:  # per_position
            scores, low_score = _trace_recurrent_state_per_position(
                raw_model, input_ids, subject_end, num_layers, seq_len,
                samples, noise_level, target_token_id
            )

    print(f"  Corrupted probability: {low_score:.4f}")
    print(f"  Effect size: {high_score - low_score:.4f}")

    return {
        'scores': scores,
        'high_score': high_score,
        'low_score': low_score,
        'input_tokens': input_tokens,
        'subject_range': (subject_start, subject_end),
        'target_token': target_token,
        'prompt': prompt,
        'component': component,
        'granularity': granularity,
    }


def _get_num_layers(model, raw_model):
    """Get number of Mamba layers in the model."""
    if hasattr(model, 'num_layers'):
        return model.num_layers
    # Count manually
    return len([
        name for name, _ in raw_model.named_modules()
        if 'backbone.layers.' in name and '.mixer' in name
    ])


def _trace_mixer_output_per_layer(
    raw_model, input_ids, tokens_to_corrupt, num_layers,
    samples, noise_level, target_token_id
):
    """
    Trace mixer outputs with per-layer granularity.
    Restore all positions in each layer.
    """
    from . import mamba_causal_trace as old_trace  # Import for collect_states

    device = next(raw_model.parameters()).device
    all_layers = list(range(num_layers))

    # Collect clean states (mixer outputs)
    clean_states = old_trace.collect_states(
        raw_model, input_ids, all_layers, component='mixer'
    )

    scores = np.zeros(num_layers)
    print(f"Tracing {num_layers} layers...")

    # Corrupted baseline
    low_score = _measure_corruption_baseline(
        raw_model, input_ids, tokens_to_corrupt,
        noise_level, samples, target_token_id
    )

    # Test each layer
    for layer_idx in range(num_layers):
        layer_scores = []

        for _ in range(samples):
            logits = old_trace.trace_with_patch_mamba(
                raw_model,
                input_ids,
                states_to_patch=[(layer_idx, 0)],  # position ignored
                clean_states=clean_states,
                tokens_to_mix=tokens_to_corrupt,
                noise_level=noise_level,
                component='mixer',
                patch_entire_layer=True,
            )

            if hasattr(logits, 'logits'):
                logits = logits.logits

            probs = torch.softmax(logits[0, -1, :], dim=0)
            layer_scores.append(probs[target_token_id].item())

        scores[layer_idx] = np.mean(layer_scores)

        if (layer_idx + 1) % 5 == 0:
            print(f"  Completed layer {layer_idx + 1}/{num_layers}")

    return scores, low_score


def _trace_mixer_output_per_position(
    raw_model, input_ids, tokens_to_corrupt, num_layers, seq_len,
    samples, noise_level, target_token_id
):
    """
    Trace mixer outputs with per-position granularity.
    Restore specific (layer, position) pairs.
    """
    from . import mamba_causal_trace as old_trace

    all_layers = list(range(num_layers))

    # Collect clean states
    clean_states = old_trace.collect_states(
        raw_model, input_ids, all_layers, component='mixer'
    )

    scores = np.zeros((num_layers, seq_len))
    print(f"Tracing {num_layers} layers × {seq_len} positions...")

    # Corrupted baseline
    low_score = _measure_corruption_baseline(
        raw_model, input_ids, tokens_to_corrupt,
        noise_level, samples, target_token_id
    )

    # Test each (layer, position) pair
    for layer_idx in range(num_layers):
        for pos in range(seq_len):
            pos_scores = []

            for _ in range(samples):
                logits = old_trace.trace_with_patch_mamba(
                    raw_model,
                    input_ids,
                    states_to_patch=[(layer_idx, pos)],
                    clean_states=clean_states,
                    tokens_to_mix=tokens_to_corrupt,
                    noise_level=noise_level,
                    component='mixer',
                    patch_entire_layer=False,
                )

                if hasattr(logits, 'logits'):
                    logits = logits.logits

                probs = torch.softmax(logits[0, -1, :], dim=0)
                pos_scores.append(probs[target_token_id].item())

            scores[layer_idx, pos] = np.mean(pos_scores)

        if (layer_idx + 1) % 5 == 0:
            print(f"  Completed layer {layer_idx + 1}/{num_layers}")

    return scores, low_score


def _trace_recurrent_state_per_layer(
    raw_model, input_ids, tokens_to_corrupt, num_layers,
    samples, noise_level, target_token_id
):
    """
    Trace SSM recurrent states with per-layer granularity.
    Restore SSM states for all positions in each layer.
    """
    from . import mamba_causal_trace as old_trace

    all_layers = list(range(num_layers))
    seq_len = input_ids.shape[1]

    # Collect clean SSM states (slow - token by token)
    print("Collecting clean SSM states (token-by-token)...")
    conv_clean, ssm_clean = old_trace.collect_ssm_states_sequential(
        raw_model, input_ids, all_layers
    )

    scores = np.zeros(num_layers)
    print(f"Tracing {num_layers} layers...")

    # Corrupted baseline
    low_score = _measure_corruption_baseline(
        raw_model, input_ids, tokens_to_corrupt,
        noise_level, samples, target_token_id
    )

    # Test each layer (restoring SSM states at all positions)
    for layer_idx in range(num_layers):
        layer_scores = []

        for _ in range(samples):
            # For per-layer, we restore SSM states at all positions in this layer
            states_to_patch = [(layer_idx, pos) for pos in range(seq_len)]

            logits = old_trace.trace_with_ssm_state_patch_sequential(
                raw_model,
                input_ids,
                states_to_patch=states_to_patch,
                conv_clean=conv_clean,
                ssm_clean=ssm_clean,
                tokens_to_mix=tokens_to_corrupt,
                noise_level=noise_level,
            )

            if hasattr(logits, 'logits'):
                logits = logits.logits

            probs = torch.softmax(logits[0, -1, :], dim=0)
            layer_scores.append(probs[target_token_id].item())

        scores[layer_idx] = np.mean(layer_scores)

        if (layer_idx + 1) % 5 == 0:
            print(f"  Completed layer {layer_idx + 1}/{num_layers}")

    return scores, low_score


def _trace_recurrent_state_per_position(
    raw_model, input_ids, tokens_to_corrupt, num_layers, seq_len,
    samples, noise_level, target_token_id
):
    """
    Trace SSM recurrent states with per-position granularity.
    Restore SSM states at specific (layer, position) pairs.
    """
    from . import mamba_causal_trace as old_trace

    all_layers = list(range(num_layers))

    # Collect clean SSM states
    print("Collecting clean SSM states (token-by-token)...")
    conv_clean, ssm_clean = old_trace.collect_ssm_states_sequential(
        raw_model, input_ids, all_layers
    )

    scores = np.zeros((num_layers, seq_len))
    print(f"Tracing {num_layers} layers × {seq_len} positions...")

    # Corrupted baseline
    low_score = _measure_corruption_baseline(
        raw_model, input_ids, tokens_to_corrupt,
        noise_level, samples, target_token_id
    )

    # Test each (layer, position) pair
    for layer_idx in range(num_layers):
        for pos in range(seq_len):
            pos_scores = []

            for _ in range(samples):
                logits = old_trace.trace_with_ssm_state_patch_sequential(
                    raw_model,
                    input_ids,
                    states_to_patch=[(layer_idx, pos)],
                    conv_clean=conv_clean,
                    ssm_clean=ssm_clean,
                    tokens_to_mix=tokens_to_corrupt,
                    noise_level=noise_level,
                )

                if hasattr(logits, 'logits'):
                    logits = logits.logits

                probs = torch.softmax(logits[0, -1, :], dim=0)
                pos_scores.append(probs[target_token_id].item())

            scores[layer_idx, pos] = np.mean(pos_scores)

        if (layer_idx + 1) % 5 == 0:
            print(f"  Completed layer {layer_idx + 1}/{num_layers}")

    return scores, low_score


def _measure_corruption_baseline(
    raw_model, input_ids, tokens_to_corrupt, noise_level, samples, target_token_id
):
    """Measure baseline probability with corruption but no restoration."""
    from . import mamba_causal_trace as old_trace

    corrupt_scores = []
    for _ in range(samples):
        logits = old_trace.trace_with_patch_mamba(
            raw_model,
            input_ids,
            states_to_patch=[],  # No restoration
            tokens_to_mix=tokens_to_corrupt,
            noise_level=noise_level,
            component='mixer',
            patch_entire_layer=True,
        )

        if hasattr(logits, 'logits'):
            logits = logits.logits

        probs = torch.softmax(logits[0, -1, :], dim=0)
        corrupt_scores.append(probs[target_token_id].item())

    return np.mean(corrupt_scores)
