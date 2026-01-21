"""
Representation extraction tools for Mamba SSMs.

Similar to ROME's repr_tools.py but adapted for Mamba's recurrent architecture.
Provides utilities to extract hidden states at specific token positions.
"""

import torch
from typing import List, Optional, Tuple
from util_ssm import ssm_nethook


def get_mamba_states_at_tokens(
    model,
    tokenizer,
    contexts: List[str],
    layers: List[int],
    token_positions: Optional[List[int]] = None,
    component: str = "mixer",
    batch_size: int = 1,
    device: str = "cuda",
):
    """
    Extract Mamba hidden states at specific token positions.

    This is analogous to ROME's get_reprs_at_word_tokens() but adapted for Mamba.

    Args:
        model: Mamba model instance (MambaModelAndTokenizer or raw model)
        tokenizer: Tokenizer instance
        contexts: List of text strings to process
        layers: List of layer numbers to extract states from
        token_positions: List of token positions to extract (one per context)
                        If None, extracts from last token of each context
        component: Which component to hook ('mixer', 'norm', etc.)
        batch_size: Number of contexts to process at once
        device: Device to run on

    Returns:
        torch.Tensor: Shape [len(contexts), len(layers), hidden_size]
                     Hidden states at specified (context, layer, position) combinations

    Example:
        >>> mt = load_mamba_model("state-spaces/mamba-130m")
        >>> contexts = ["The Eiffel Tower is located in"]
        >>> layers = [10, 15, 20]
        >>> states = get_mamba_states_at_tokens(
        ...     mt.model, mt.tokenizer, contexts, layers
        ... )
        >>> states.shape
        torch.Size([1, 3, 768])  # 1 context, 3 layers, 768 hidden size
    """
    # Handle model wrapper vs raw model
    if hasattr(model, 'model'):
        raw_model = model.model
    else:
        raw_model = model

    # Default to last token if positions not specified
    if token_positions is None:
        token_positions = [-1] * len(contexts)

    if len(token_positions) != len(contexts):
        raise ValueError(
            f"Length mismatch: {len(contexts)} contexts but {len(token_positions)} positions"
        )

    # Storage for extracted states
    all_states = []

    # Process in batches
    for batch_start in range(0, len(contexts), batch_size):
        batch_end = min(batch_start + batch_size, len(contexts))
        batch_contexts = contexts[batch_start:batch_end]
        batch_positions = token_positions[batch_start:batch_end]

        # Tokenize batch
        tokens = tokenizer(
            batch_contexts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        # Set up hooks for all requested layers
        layer_names = [
            ssm_nethook.mamba_layername(raw_model, layer_num, component)
            for layer_num in layers
        ]

        with torch.no_grad():
            with ssm_nethook.TraceDict(
                raw_model, layer_names, retain_output=True
            ) as traces:
                # Run forward pass (only need input_ids for Mamba)
                _ = raw_model(tokens["input_ids"])

                # Extract states at specified positions for this batch
                batch_states = []
                for ctx_idx, pos in enumerate(batch_positions):
                    context_states = []
                    for layer_name in layer_names:
                        # Get output of this layer: shape [batch, seq_len, hidden_size]
                        layer_output = traces[layer_name].output

                        # Handle tuple output (some layers return (hidden, residual))
                        if isinstance(layer_output, tuple):
                            layer_output = layer_output[0]

                        # Extract state at specified position
                        # pos can be negative (e.g., -1 for last token)
                        state = layer_output[ctx_idx, pos, :]
                        context_states.append(state)

                    # Stack states across layers: [num_layers, hidden_size]
                    context_states = torch.stack(context_states, dim=0)
                    batch_states.append(context_states)

                # Stack across batch: [batch_size, num_layers, hidden_size]
                batch_states = torch.stack(batch_states, dim=0)
                all_states.append(batch_states)

    # Concatenate all batches: [total_contexts, num_layers, hidden_size]
    all_states = torch.cat(all_states, dim=0)

    return all_states


def get_mamba_embedding_at_tokens(
    model,
    tokenizer,
    contexts: List[str],
    token_positions: Optional[List[int]] = None,
    batch_size: int = 1,
    device: str = "cuda",
):
    """
    Extract Mamba embedding vectors at specific token positions.

    Args:
        model: Mamba model instance
        tokenizer: Tokenizer instance
        contexts: List of text strings
        token_positions: List of positions (one per context), or None for last token
        batch_size: Batch size for processing
        device: Device to run on

    Returns:
        torch.Tensor: Shape [len(contexts), embed_size]
                     Embedding vectors at specified positions
    """
    # Handle model wrapper
    if hasattr(model, 'model'):
        raw_model = model.model
    else:
        raw_model = model

    if token_positions is None:
        token_positions = [-1] * len(contexts)

    if len(token_positions) != len(contexts):
        raise ValueError(
            f"Length mismatch: {len(contexts)} contexts but {len(token_positions)} positions"
        )

    all_embeddings = []

    for batch_start in range(0, len(contexts), batch_size):
        batch_end = min(batch_start + batch_size, len(contexts))
        batch_contexts = contexts[batch_start:batch_end]
        batch_positions = token_positions[batch_start:batch_end]

        # Tokenize
        tokens = tokenizer(
            batch_contexts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        # Get embedding layer
        try:
            embedding_name = ssm_nethook.mamba_layername(raw_model, 0, "embed")
        except ValueError:
            # Fallback: try to find it directly
            if hasattr(raw_model, 'backbone') and hasattr(raw_model.backbone, 'embedding'):
                embedding_layer = raw_model.backbone.embedding
            elif hasattr(raw_model, 'embedding'):
                embedding_layer = raw_model.embedding
            else:
                raise ValueError("Could not find embedding layer")
        else:
            embedding_layer = ssm_nethook.get_module(raw_model, embedding_name)

        with torch.no_grad():
            # Get embeddings: [batch, seq_len, embed_size]
            embeddings = embedding_layer(tokens["input_ids"])

            # Extract at specified positions
            batch_embeddings = []
            for ctx_idx, pos in enumerate(batch_positions):
                emb = embeddings[ctx_idx, pos, :]
                batch_embeddings.append(emb)

            batch_embeddings = torch.stack(batch_embeddings, dim=0)
            all_embeddings.append(batch_embeddings)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings


def make_inputs(tokenizer, prompts: List[str], device: str = "cuda"):
    """
    Tokenize prompts for Mamba models.

    Args:
        tokenizer: Tokenizer instance
        prompts: List of text prompts
        device: Device to place tensors on

    Returns:
        dict: Tokenized inputs with 'input_ids' key (Mamba doesn't use attention_mask)
    """
    tokens = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    # Mamba only needs input_ids
    return {"input_ids": tokens["input_ids"]}


def find_token_range(tokenizer, text: str, substring: str) -> Tuple[int, int]:
    """
    Find the token range for a substring within a text.

    Useful for identifying which tokens correspond to specific words/phrases
    in causal tracing experiments.

    Args:
        tokenizer: Tokenizer instance
        text: Full text string
        substring: Substring to locate

    Returns:
        Tuple[int, int]: (start_token_idx, end_token_idx) inclusive range

    Example:
        >>> tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        >>> text = "The Eiffel Tower is located in Paris"
        >>> start, end = find_token_range(tokenizer, text, "Eiffel Tower")
        >>> # Returns indices of tokens that make up "Eiffel Tower"
    """
    # Find character positions
    char_start = text.find(substring)
    if char_start == -1:
        raise ValueError(f"Substring '{substring}' not found in '{text}'")

    char_end = char_start + len(substring)

    # Tokenize full text
    tokens = tokenizer(text, return_tensors="pt")
    token_ids = tokens["input_ids"][0]

    # Tokenize text up to start and end positions
    prefix_tokens = tokenizer(text[:char_start], return_tensors="pt")["input_ids"][0]
    prefix_and_substr_tokens = tokenizer(text[:char_end], return_tensors="pt")["input_ids"][0]

    start_token_idx = len(prefix_tokens)
    end_token_idx = len(prefix_and_substr_tokens) - 1  # Inclusive end

    return start_token_idx, end_token_idx

