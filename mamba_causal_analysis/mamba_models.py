"""
Model and tokenizer wrapper for Mamba SSMs.
Similar to ROME's ModelAndTokenizer but adapted for Mamba architecture.
"""

import re
from typing import Optional

import torch
from transformers import AutoTokenizer

try:
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
except ImportError:
    MambaLMHeadModel = None
    print("Warning: mamba_ssm not installed. Run: pip install mamba-ssm")


class MambaModelAndTokenizer:
    """
    Wrapper for Mamba language models and their tokenizers.

    Provides convenient access to:
    - The model itself
    - Tokenizer (uses GPT-NeoX tokenizer)
    - Layer names and structure
    - Number of layers

    Similar to ROME's ModelAndTokenizer but for Mamba architecture.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        model: Optional[torch.nn.Module] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        torch_dtype: Optional[torch.dtype] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize Mamba model and tokenizer.

        Args:
            model_name: HuggingFace model name (e.g., 'state-spaces/mamba-130m')
            model: Pre-loaded model (if None, loads from model_name)
            tokenizer: Pre-loaded tokenizer (if None, loads GPT-NeoX tokenizer)
            torch_dtype: Data type for model (default: float32, or float16 for large models)
            device: Device to load model on
        """
        if MambaLMHeadModel is None:
            raise ImportError(
                "mamba_ssm not installed. Install with: pip install mamba-ssm"
            )

        # Load tokenizer (Mamba uses GPT-NeoX tokenizer)
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
            # Set pad token to eos token (standard practice)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer

        # Load model
        if model is None:
            if model_name is None:
                raise ValueError("Must provide either model_name or model")

            print(f"Loading Mamba model: {model_name}")
            self.model = MambaLMHeadModel.from_pretrained(model_name)
            self.model.eval()

            # Convert dtype if specified
            if torch_dtype is not None:
                self.model = self.model.to(dtype=torch_dtype)

            self.model.to(device)

            # Disable gradients for efficiency
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            self.model = model

        self.model_name = model_name
        self.device = device

        # Identify layer structure
        self._identify_layers()

    def _identify_layers(self):
        """
        Identify and catalog Mamba layer structure.

        Mamba models typically have structure:
        - backbone.embedding
        - backbone.layers.0, backbone.layers.1, ..., backbone.layers.N
        - backbone.norm_f
        - lm_head

        Each layer contains:
        - norm (LayerNorm)
        - mixer (Mamba block) - this is the main SSM component
        """
        self.layer_names = []
        self.mixer_names = []

        # Find all Mamba layers
        for name, module in self.model.named_modules():
            # Match pattern: backbone.layers.{i}
            if re.match(r"^backbone\.layers\.\d+$", name):
                self.layer_names.append(name)
            # Match pattern: backbone.layers.{i}.mixer
            if re.match(r"^backbone\.layers\.\d+\.mixer$", name):
                self.mixer_names.append(name)

        self.num_layers = len(self.layer_names)

        if self.num_layers == 0:
            # Try alternative naming convention
            for name, module in self.model.named_modules():
                if re.match(r"^layers\.\d+$", name):
                    self.layer_names.append(name)
                if re.match(r"^layers\.\d+\.mixer$", name):
                    self.mixer_names.append(name)

            self.num_layers = len(self.layer_names)

        if self.num_layers == 0:
            raise ValueError(
                "Could not identify Mamba layers. Model structure may be different than expected."
            )

        print(f"Identified {self.num_layers} Mamba layers")

    def get_layer_name(self, layer_num: int, component: Optional[str] = None) -> str:
        """
        Get the full module name for a specific layer and component.

        Args:
            layer_num: Layer index (0 to num_layers-1)
            component: Optional component name:
                - None: Full layer (e.g., 'backbone.layers.5')
                - 'mixer': SSM block (e.g., 'backbone.layers.5.mixer')
                - 'norm': Layer norm (e.g., 'backbone.layers.5.norm')
                - 'in_proj': Input projection
                - 'out_proj': Output projection
                - 'conv1d': Causal convolution

        Returns:
            Full module name as string
        """
        if layer_num < 0 or layer_num >= self.num_layers:
            raise ValueError(f"Layer {layer_num} out of range [0, {self.num_layers})")

        base_name = self.layer_names[layer_num]

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

    def get_embedding_layer_name(self) -> str:
        """Get the name of the embedding layer."""
        # Try common naming patterns
        for name in ["backbone.embedding", "embedding", "backbone.embed_tokens"]:
            try:
                # Check if this module exists
                parts = name.split(".")
                module = self.model
                for part in parts:
                    module = getattr(module, part)
                return name
            except AttributeError:
                continue

        raise ValueError("Could not find embedding layer")

    def get_lm_head_name(self) -> str:
        """Get the name of the language model head."""
        for name in ["lm_head", "head", "output"]:
            if hasattr(self.model, name):
                return name

        raise ValueError("Could not find LM head")

    def __repr__(self):
        return (
            f"MambaModelAndTokenizer(\n"
            f"  model: {self.model_name or type(self.model).__name__}\n"
            f"  layers: {self.num_layers}\n"
            f"  device: {self.device}\n"
            f"  tokenizer: {type(self.tokenizer).__name__}\n"
            f")"
        )


def load_mamba_model(
    model_name: str = "state-spaces/mamba-130m",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype: Optional[torch.dtype] = None,
) -> MambaModelAndTokenizer:
    """
    Convenience function to load a Mamba model.

    Args:
        model_name: HuggingFace model name
        device: Device to load on
        torch_dtype: Data type (None for default, torch.float16 for half precision)

    Returns:
        MambaModelAndTokenizer instance

    Example:
        >>> mt = load_mamba_model("state-spaces/mamba-130m")
        >>> print(mt.num_layers)
        24
    """
    return MambaModelAndTokenizer(
        model_name=model_name,
        device=device,
        torch_dtype=torch_dtype,
    )
