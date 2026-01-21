"""
SSM-aware hooking utilities.

Extends ROME's nethook.py to work with Mamba SSM architecture.
The core Trace and TraceDict classes from ROME are architecture-agnostic
and work with any PyTorch module, so we mainly add convenience functions
for Mamba-specific layer naming.
"""

import sys
from pathlib import Path

# Add rome directory to path to import nethook
rome_path = Path(__file__).parent.parent / "rome"
if str(rome_path) not in sys.path:
    sys.path.insert(0, str(rome_path))

from util import nethook  # ROME's nethook utilities

# Re-export ROME's utilities for convenience
Trace = nethook.Trace
TraceDict = nethook.TraceDict
StopForward = nethook.StopForward
recursive_copy = nethook.recursive_copy
get_module = nethook.get_module
get_parameter = nethook.get_parameter
replace_module = nethook.replace_module
set_requires_grad = nethook.set_requires_grad


def mamba_layername(model, layer_num, component=None):
    """
    Generate layer names for Mamba models.

    This is the SSM equivalent of ROME's layername() function,
    adapted for Mamba's architecture.

    Args:
        model: Mamba model instance
        layer_num: Layer number (int)
        component: Optional component name:
            - None: Full layer (e.g., 'backbone.layers.5')
            - 'embed': Embedding layer
            - 'mixer': SSM mixer block
            - 'norm': Layer normalization
            - 'in_proj': Input projection within mixer
            - 'out_proj': Output projection within mixer
            - 'conv1d': Causal convolution
            - 'x_proj': Projection to SSM parameters
            - 'dt_proj': Delta projection

    Returns:
        str: Full module name

    Examples:
        >>> mamba_layername(model, 5)
        'backbone.layers.5'
        >>> mamba_layername(model, 5, 'mixer')
        'backbone.layers.5.mixer'
        >>> mamba_layername(model, 0, 'embed')
        'backbone.embedding'
    """
    # Handle embedding layer
    if component == "embed":
        # Try common embedding layer names
        for name in ["backbone.embedding", "embedding", "backbone.embed_tokens"]:
            if hasattr_nested(model, name):
                return name
        raise ValueError("Could not find embedding layer")

    # Handle LM head
    if component == "lm_head":
        for name in ["lm_head", "head", "output"]:
            if hasattr(model, name):
                return name
        raise ValueError("Could not find LM head")

    # Try backbone.layers.{i} pattern first (most common for Mamba)
    base_name = f"backbone.layers.{layer_num}"

    # If that doesn't exist, try just layers.{i}
    if not hasattr_nested(model, base_name):
        base_name = f"layers.{layer_num}"

    # If still doesn't exist, raise error
    if not hasattr_nested(model, base_name):
        raise ValueError(
            f"Could not find layer {layer_num}. "
            f"Tried 'backbone.layers.{layer_num}' and 'layers.{layer_num}'"
        )

    # Return full layer or specific component
    if component is None:
        return base_name
    elif component == "mixer":
        return f"{base_name}.mixer"
    elif component == "norm":
        return f"{base_name}.norm"
    elif component in ["in_proj", "out_proj", "conv1d", "x_proj", "dt_proj"]:
        return f"{base_name}.mixer.{component}"
    else:
        raise ValueError(f"Unknown component: {component}")


def hasattr_nested(obj, attr_path):
    """
    Check if nested attribute exists.

    Args:
        obj: Object to check
        attr_path: Dot-separated attribute path (e.g., 'backbone.layers.0')

    Returns:
        bool: True if attribute exists

    Example:
        >>> hasattr_nested(model, 'backbone.layers.0.mixer')
        True
    """
    parts = attr_path.split(".")
    current = obj

    for part in parts:
        if not hasattr(current, part):
            return False
        current = getattr(current, part)

    return True


def get_mamba_module(model, layer_num, component=None):
    """
    Get a Mamba module by layer number and component.

    Convenience wrapper around get_module that uses mamba_layername.

    Args:
        model: Mamba model
        layer_num: Layer number
        component: Component name (see mamba_layername for options)

    Returns:
        torch.nn.Module: The requested module

    Example:
        >>> mixer = get_mamba_module(model, 5, 'mixer')
        >>> isinstance(mixer, torch.nn.Module)
        True
    """
    layer_name = mamba_layername(model, layer_num, component)
    return get_module(model, layer_name)


def trace_mamba_layer(
    model,
    layer_num,
    component=None,
    retain_input=False,
    retain_output=True,
    clone=False,
    detach=False,
    edit_output=None,
    stop=False,
):
    """
    Convenience function to trace a Mamba layer.

    Args:
        model: Mamba model
        layer_num: Layer number
        component: Component to trace (see mamba_layername)
        retain_input: Whether to retain input
        retain_output: Whether to retain output
        clone: Whether to clone tensors
        detach: Whether to detach tensors
        edit_output: Function to edit output
        stop: Whether to stop forward pass after this layer

    Returns:
        Trace context manager

    Example:
        >>> with trace_mamba_layer(model, 5, 'mixer') as t:
        ...     output = model(input_ids)
        ...     hidden_state = t.output
    """
    layer_name = mamba_layername(model, layer_num, component)

    return Trace(
        model,
        layer=layer_name,
        retain_input=retain_input,
        retain_output=retain_output,
        clone=clone,
        detach=detach,
        edit_output=edit_output,
        stop=stop,
    )


def trace_multiple_mamba_layers(
    model,
    layer_specs,
    retain_input=False,
    retain_output=True,
    clone=False,
    detach=False,
    edit_output=None,
    stop=False,
):
    """
    Trace multiple Mamba layers at once.

    Args:
        model: Mamba model
        layer_specs: List of (layer_num, component) tuples, or just layer numbers
        retain_input: Whether to retain inputs
        retain_output: Whether to retain outputs
        clone: Whether to clone tensors
        detach: Whether to detach tensors
        edit_output: Function to edit outputs (gets layer name as argument)
        stop: Whether to stop after last layer

    Returns:
        TraceDict context manager

    Example:
        >>> specs = [(5, 'mixer'), (10, 'mixer'), (15, 'mixer')]
        >>> with trace_multiple_mamba_layers(model, specs) as td:
        ...     output = model(input_ids)
        ...     mixer_5 = td['backbone.layers.5.mixer'].output
    """
    # Convert specs to layer names
    layer_names = []
    for spec in layer_specs:
        if isinstance(spec, tuple):
            layer_num, component = spec
        else:
            layer_num = spec
            component = None

        layer_names.append(mamba_layername(model, layer_num, component))

    return TraceDict(
        model,
        layers=layer_names,
        retain_input=retain_input,
        retain_output=retain_output,
        clone=clone,
        detach=detach,
        edit_output=edit_output,
        stop=stop,
    )


# Example usage
if __name__ == "__main__":
    print("SSM NetHook Utilities")
    print("=" * 50)
    print("This module provides Mamba-specific wrappers around ROME's nethook utilities.")
    print()
    print("Key functions:")
    print("  - mamba_layername(model, layer_num, component)")
    print("  - get_mamba_module(model, layer_num, component)")
    print("  - trace_mamba_layer(model, layer_num, component, ...)")
    print("  - trace_multiple_mamba_layers(model, layer_specs, ...)")
    print()
    print("Inherits from ROME:")
    print("  - Trace, TraceDict, StopForward")
    print("  - get_module, get_parameter, replace_module")
    print("  - set_requires_grad, recursive_copy")
