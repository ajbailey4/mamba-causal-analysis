"""Test per-timestep SSM state restoration."""

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

# Run both modes
print("\n" + "="*60)
print("TESTING PER-TIMESTEP RESTORATION")
print("="*60)

# Mode 1: Mixer outputs (hidden states)
print("\n1. Per-timestep with MIXER OUTPUTS (fast)")
result_mixer = calculate_hidden_flow(
    mt,
    mt.tokenizer,
    prompt=prompt,
    subject=subject,
    samples=5,
    noise_level=3.0,
    mode="vertical",
)
print(f"✓ Complete! Shape: {result_mixer['scores'].shape}")

# Mode 2: SSM recurrent states
print("\n2. Per-timestep with SSM STATES (slow)")
result_ssm = calculate_hidden_flow(
    mt,
    mt.tokenizer,
    prompt=prompt,
    subject=subject,
    samples=5,
    noise_level=3.0,
    mode="per_timestep_ssm",
)
print(f"✓ Complete! Shape: {result_ssm['scores'].shape}")

# Compare results
print("\n" + "="*60)
print("COMPARISON")
print("="*60)

tokens = result_mixer['input_tokens']
scores_mixer = result_mixer['scores']
scores_ssm = result_ssm['scores']
subj_start, subj_end = result_mixer['subject_range']

print(f"\nSubject range: positions {subj_start} to {subj_end}")
print(f"Subject tokens: {tokens[subj_start:subj_end]}")

print("\nMixer Outputs:")
for i, (token, score) in enumerate(zip(tokens, scores_mixer)):
    marker = " <-- SUBJECT" if subj_start <= i < subj_end else ""
    print(f"  {i:2d}. {token:15s} {score:.4f}{marker}")

print("\nSSM States:")
for i, (token, score) in enumerate(zip(tokens, scores_ssm)):
    marker = " <-- SUBJECT" if subj_start <= i < subj_end else ""
    print(f"  {i:2d}. {token:15s} {score:.4f}{marker}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax1.bar(range(len(scores_mixer)), scores_mixer, alpha=0.7, color='lightcoral')
ax1.axhline(result_mixer['low_score'], color='red', linestyle='--', alpha=0.7)
ax1.axhline(result_mixer['high_score'], color='green', linestyle='--', alpha=0.7)
ax1.axvspan(subj_start - 0.5, subj_end - 0.5, alpha=0.2, color='yellow')
ax1.set_xlabel('Position')
ax1.set_ylabel('Probability')
ax1.set_title('Mixer Outputs')
ax1.set_xticks(range(len(tokens)))
ax1.set_xticklabels(tokens, rotation=45, ha='right')
ax1.grid(True, alpha=0.3, axis='y')

ax2 = axes[1]
ax2.bar(range(len(scores_ssm)), scores_ssm, alpha=0.7, color='mediumseagreen')
ax2.axhline(result_ssm['low_score'], color='red', linestyle='--', alpha=0.7)
ax2.axhline(result_ssm['high_score'], color='green', linestyle='--', alpha=0.7)
ax2.axvspan(subj_start - 0.5, subj_end - 0.5, alpha=0.2, color='yellow')
ax2.set_xlabel('Position')
ax2.set_ylabel('Probability')
ax2.set_title('SSM Recurrent States')
ax2.set_xticks(range(len(tokens)))
ax2.set_xticklabels(tokens, rotation=45, ha='right')
ax2.grid(True, alpha=0.3, axis='y')

plt.suptitle(f'Per-Timestep Restoration: "{prompt}"', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('per_timestep_ssm_comparison.png', dpi=150, bbox_inches='tight')
print("\n✓ Plot saved to per_timestep_ssm_comparison.png")

print("\n" + "="*60)
print("EXPECTED BEHAVIOR")
print("="*60)
print("SSM states should show MUCH higher restoration scores for")
print("positions AFTER the corrupted subject (positions >= 4),")
print("because the recurrent state carries information forward.")
print("\nMixer outputs only restore the final layer output at each")
print("position, so they don't benefit from recurrent propagation.")
