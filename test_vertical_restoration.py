"""Test vertical restoration functionality."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mamba_causal_analysis.mamba_models import load_mamba_model
from mamba_causal_analysis.mamba_causal_trace import calculate_hidden_flow

# Load model
print("Loading model...")
model_name = "state-spaces/mamba-130m"
device = "cuda" if torch.cuda.is_available() else "cpu"
mt = load_mamba_model(model_name, device=device)
print(f"✓ Model loaded with {mt.num_layers} layers")

# Test case
prompt = "The Eiffel Tower is located in"
subject = "Eiffel Tower"

print(f"\nTesting on: \"{prompt}\"")
print(f"Subject: \"{subject}\"")

# Run vertical restoration
print("\n" + "="*60)
print("VERTICAL RESTORATION TEST")
print("="*60)
print("This restores ALL layers at each position")

result_vertical = calculate_hidden_flow(
    mt,
    mt.tokenizer,
    prompt=prompt,
    subject=subject,
    samples=5,  # Use 5 samples for speed
    noise_level=3.0,
    mode="vertical",
)

print(f"\n✓ Complete!")
print(f"  Clean: {result_vertical['high_score']:.4f}")
print(f"  Corrupted: {result_vertical['low_score']:.4f}")
print(f"  Scores shape: {result_vertical['scores'].shape}")
print(f"  Expected shape: [seq_len] = [{len(result_vertical['input_tokens'])}]")

# Visualize
print("\nVertical Restoration Results:")
print("-" * 60)
tokens = result_vertical['input_tokens']
scores = result_vertical['scores']

for i, (token, score) in enumerate(zip(tokens, scores)):
    bar = "#" * int(score * 50)
    print(f"  {i:2d}. {token:15s} {score:.4f} {bar}")

# Find most important position
best_pos = scores.argmax()
print(f"\nMost important position: {best_pos}")
print(f"  Token: '{tokens[best_pos]}'")
print(f"  Score: {scores[best_pos]:.4f}")

# Plot
plt.figure(figsize=(12, 5))
plt.bar(range(len(scores)), scores, alpha=0.7)
plt.axhline(result_vertical['low_score'], color='red', linestyle='--', label='Corrupted', alpha=0.7)
plt.axhline(result_vertical['high_score'], color='green', linestyle='--', label='Clean', alpha=0.7)
plt.xlabel('Token Position', fontsize=12)
plt.ylabel('Probability (restoring all layers at this position)', fontsize=12)
plt.title(f'Vertical Restoration: "{prompt}"', fontsize=14)
plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('/Users/ajbailey4@ad.wisc.edu/mamba-causal-analysis/vertical_restoration_test.png', dpi=150)
print("\n✓ Plot saved to vertical_restoration_test.png")

print("\n" + "="*60)
print("COMPARISON: Horizontal vs Vertical")
print("="*60)

# Also run horizontal for comparison
result_horizontal = calculate_hidden_flow(
    mt,
    mt.tokenizer,
    prompt=prompt,
    subject=subject,
    samples=5,
    noise_level=3.0,
    mode="per_layer",
)

print(f"\nHorizontal (per-layer) best: Layer {result_horizontal['scores'].argmax()}")
print(f"  Score: {result_horizontal['scores'][result_horizontal['scores'].argmax()]:.4f}")

print(f"\nVertical (all layers at position) best: Position {best_pos}")
print(f"  Token: '{tokens[best_pos]}'")
print(f"  Score: {scores[best_pos]:.4f}")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
