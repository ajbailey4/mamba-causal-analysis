"""
Helper utilities for Mamba layer naming conventions.

Provides functions to help navigate Mamba's module structure and
handle different naming conventions across Mamba variants.
"""

import re
from typing import List, Optional, Tuple


def parse_mamba_architecture(model) -> dict:
    """
    Analyze a Mamba model's architecture and return structural information.

    Args:
        model: Mamba model instance

    Returns:
        dict with keys:
            - 'num_layers': Number of Mamba layers
            - 'layer_prefix': Prefix for layer names ('backbone.layers' or 'layers')
            - 'has_backbone': Whether model uses 'backbone' prefix
            - 'embedding_name': Name of embedding layer
            - 'lm_head_name': Name of LM head
            - 'layer_pattern': Regex pattern for matching layers
            - 'mixer_pattern': Regex pattern for matching mixer blocks
    """
    info = {
        'num_layers': 0,
        'layer_prefix': None,
        'has_backbone': False,
        'embedding_name': None,
        'lm_head_name': None,
        'layer_names': [],
        'mixer_names': [],
    }

    # Check for backbone prefix
    if hasattr(model, 'backbone'):
        info['has_backbone'] = True
        info['layer_prefix'] = 'backbone.layers'
        info['layer_pattern'] = r'^backbone\.layers\.\d+$'
        info['mixer_pattern'] = r'^backbone\.layers\.\d+\.mixer$'
    else:
        info['layer_prefix'] = 'layers'
        info['layer_pattern'] = r'^layers\.\d+$'
        info['mixer_pattern'] = r'^layers\.\d+\.mixer$'

    # Find layers
    for name, module in model.named_modules():
        if re.match(info['layer_pattern'], name):
            info['layer_names'].append(name)
        if re.match(info['mixer_pattern'], name):
            info['mixer_names'].append(name)

    info['num_layers'] = len(info['layer_names'])

    # Find embedding
    for emb_name in ['backbone.embedding', 'embedding', 'backbone.embed_tokens', 'embed_tokens']:
        if _has_attr_nested(model, emb_name):
            info['embedding_name'] = emb_name
            break

    # Find LM head
    for head_name in ['lm_head', 'head', 'output']:
        if hasattr(model, head_name):
            info['lm_head_name'] = head_name
            break

    return info


def get_layer_components(model, layer_num: int) -> List[str]:
    """
    Get all available components within a Mamba layer.

    Args:
        model: Mamba model
        layer_num: Layer number

    Returns:
        List of component names available in this layer

    Example:
        >>> get_layer_components(model, 0)
        ['norm', 'mixer', 'mixer.in_proj', 'mixer.conv1d', 'mixer.out_proj', ...]
    """
    info = parse_mamba_architecture(model)

    if layer_num >= info['num_layers']:
        raise ValueError(f"Layer {layer_num} out of range (model has {info['num_layers']} layers)")

    base_name = f"{info['layer_prefix']}.{layer_num}"
    components = []

    # Get the layer module
    try:
        layer_module = _get_module_nested(model, base_name)
    except AttributeError:
        return components

    # Find all sub-modules
    for name, module in layer_module.named_modules():
        if name:  # Skip the layer itself (empty string)
            components.append(name)

    return sorted(components)


def format_layer_name(layer_num: int, component: Optional[str] = None, has_backbone: bool = True) -> str:
    """
    Format a layer name according to Mamba conventions.

    Args:
        layer_num: Layer number
        component: Optional component (e.g., 'mixer', 'norm')
        has_backbone: Whether model uses 'backbone' prefix

    Returns:
        Formatted layer name

    Example:
        >>> format_layer_name(5, 'mixer', has_backbone=True)
        'backbone.layers.5.mixer'
        >>> format_layer_name(5, has_backbone=False)
        'layers.5'
    """
    prefix = 'backbone.layers' if has_backbone else 'layers'
    base = f"{prefix}.{layer_num}"

    if component is None:
        return base
    else:
        return f"{base}.{component}"


def extract_layer_number(layer_name: str) -> Optional[int]:
    """
    Extract layer number from a layer name.

    Args:
        layer_name: Full layer name (e.g., 'backbone.layers.5.mixer')

    Returns:
        Layer number (int) or None if not a layer name

    Example:
        >>> extract_layer_number('backbone.layers.5.mixer')
        5
        >>> extract_layer_number('backbone.embedding')
        None
    """
    # Try backbone.layers pattern
    match = re.search(r'backbone\.layers\.(\d+)', layer_name)
    if match:
        return int(match.group(1))

    # Try plain layers pattern
    match = re.search(r'^layers\.(\d+)', layer_name)
    if match:
        return int(match.group(1))

    return None


def extract_component(layer_name: str) -> Optional[str]:
    """
    Extract component name from a layer name.

    Args:
        layer_name: Full layer name

    Returns:
        Component name or None if just the base layer

    Example:
        >>> extract_component('backbone.layers.5.mixer')
        'mixer'
        >>> extract_component('backbone.layers.5.mixer.in_proj')
        'mixer.in_proj'
        >>> extract_component('backbone.layers.5')
        None
    """
    # Remove layer prefix
    layer_num = extract_layer_number(layer_name)
    if layer_num is None:
        return None

    # Find where the component starts
    patterns = [
        f'backbone.layers.{layer_num}.',
        f'layers.{layer_num}.',
    ]

    for pattern in patterns:
        if pattern in layer_name:
            component = layer_name.split(pattern, 1)[1]
            return component if component else None

    return None


def is_mixer_layer(layer_name: str) -> bool:
    """Check if a layer name refers to a mixer block."""
    return layer_name.endswith('.mixer') or '.mixer.' in layer_name


def is_embedding_layer(layer_name: str) -> bool:
    """Check if a layer name refers to an embedding layer."""
    return 'embedding' in layer_name or 'embed_tokens' in layer_name


def is_lm_head(layer_name: str) -> bool:
    """Check if a layer name refers to the LM head."""
    return layer_name in ['lm_head', 'head', 'output']


def get_mixer_subcomponents() -> List[str]:
    """
    Get list of typical sub-components within a Mamba mixer block.

    Returns:
        List of component names

    Note:
        Actual components may vary by Mamba version and implementation.
    """
    return [
        'in_proj',      # Input projection
        'conv1d',       # Causal convolution
        'x_proj',       # Projection to SSM parameters (B, C, dt)
        'dt_proj',      # Delta (timescale) projection
        'out_proj',     # Output projection
        'A_log',        # Log of A matrix (state transition)
        'D',            # Skip connection parameter
    ]


# Helper functions
def _has_attr_nested(obj, attr_path: str) -> bool:
    """Check if nested attribute exists."""
    parts = attr_path.split('.')
    current = obj
    for part in parts:
        if not hasattr(current, part):
            return False
        current = getattr(current, part)
    return True


def _get_module_nested(obj, attr_path: str):
    """Get nested module."""
    parts = attr_path.split('.')
    current = obj
    for part in parts:
        current = getattr(current, part)
    return current


# Example usage and testing
if __name__ == "__main__":
    print("Mamba Layer Naming Utilities")
    print("=" * 60)

    # Test layer name formatting
    print("\nLayer name formatting:")
    print(f"  Layer 5 (with backbone): {format_layer_name(5, has_backbone=True)}")
    print(f"  Layer 5 mixer (with backbone): {format_layer_name(5, 'mixer', True)}")
    print(f"  Layer 5 (no backbone): {format_layer_name(5, has_backbone=False)}")

    # Test layer number extraction
    print("\nLayer number extraction:")
    test_names = [
        'backbone.layers.5.mixer',
        'layers.10',
        'backbone.embedding',
    ]
    for name in test_names:
        num = extract_layer_number(name)
        comp = extract_component(name)
        print(f"  '{name}' -> layer={num}, component={comp}")

    # Test component identification
    print("\nComponent identification:")
    test_layers = [
        'backbone.layers.5.mixer',
        'backbone.embedding',
        'lm_head',
    ]
    for layer in test_layers:
        print(f"  '{layer}':")
        print(f"    is_mixer: {is_mixer_layer(layer)}")
        print(f"    is_embedding: {is_embedding_layer(layer)}")
        print(f"    is_lm_head: {is_lm_head(layer)}")

    # Show typical mixer components
    print("\nTypical mixer sub-components:")
    for comp in get_mixer_subcomponents():
        print(f"  - {comp}")
