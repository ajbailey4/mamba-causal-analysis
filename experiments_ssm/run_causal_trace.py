"""
Run causal tracing experiments on Mamba models.

Command-line script to perform causal mediation analysis on factual prompts.
Generates heatmap data showing which (layer, position) pairs are critical
for factual recall.

Usage:
    python experiments_ssm/run_causal_trace.py \\
        --model state-spaces/mamba-130m \\
        --prompt "The Eiffel Tower is located in" \\
        --subject "Eiffel Tower" \\
        --samples 10 \\
        --output results/eiffel_tower_trace.npz

Example prompts:
    - "The Eiffel Tower is located in" / "Eiffel Tower" -> "Paris"
    - "The Space Needle is located in downtown" / "Space Needle" -> "Seattle"
    - "The mother tongue of Angela Merkel is" / "Angela Merkel" -> "German"
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from mamba_causal_analysis.mamba_models import load_mamba_model
from mamba_causal_analysis.mamba_causal_trace import calculate_hidden_flow


def main():
    parser = argparse.ArgumentParser(
        description="Run causal tracing on Mamba models"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="state-spaces/mamba-130m",
        help="Mamba model to use",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Factual prompt to trace (e.g., 'The Eiffel Tower is located in')",
    )
    parser.add_argument(
        "--subject",
        type=str,
        required=True,
        help="Subject phrase in the prompt (e.g., 'Eiffel Tower')",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of noise samples to average over (default: 10)",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=3.0,
        help="Noise level in standard deviations (default: 3.0)",
    )
    parser.add_argument(
        "--component",
        type=str,
        default="mixer",
        choices=["mixer", "norm", "in_proj", "out_proj"],
        help="Component to trace (default: mixer)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for results (.npz format)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Mamba Causal Tracing")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Prompt: \"{args.prompt}\"")
    print(f"Subject: \"{args.subject}\"")
    print(f"Samples: {args.samples}")
    print(f"Noise level: {args.noise} std")
    print(f"Component: {args.component}")
    print(f"Device: {args.device}")
    print()

    # Load model
    print("Loading model...")
    mt = load_mamba_model(args.model, device=args.device)
    print(f"Loaded {args.model} with {mt.num_layers} layers")
    print()

    # Run causal tracing
    print("Running causal tracing...")
    print(f"This will trace {mt.num_layers} layers across all token positions")
    print(f"Estimated time: ~{mt.num_layers * args.samples * 0.1:.1f} seconds")
    print()

    result = calculate_hidden_flow(
        mt,
        mt.tokenizer,
        prompt=args.prompt,
        subject=args.subject,
        samples=args.samples,
        noise_level=args.noise,
        component=args.component,
    )

    print()
    print("Results:")
    print(f"  Clean probability: {result['high_score']:.4f}")
    print(f"  Corrupted probability: {result['low_score']:.4f}")
    print(f"  Effect size: {result['high_score'] - result['low_score']:.4f}")
    print(f"  Target token: {result['target_token']}")
    print()

    # Find most important layer/position
    scores = result['scores']
    max_layer, max_pos = np.unravel_index(scores.argmax(), scores.shape)
    print(f"Most important restoration:")
    print(f"  Layer: {max_layer}")
    print(f"  Position: {max_pos} (token: '{result['input_tokens'][max_pos]}')")
    print(f"  Score: {scores[max_layer, max_pos]:.4f}")
    print()

    # Subject token restoration scores
    subj_start, subj_end = result['subject_range']
    subject_scores = scores[:, subj_start:subj_end+1]
    avg_subject_effect = subject_scores.mean()
    print(f"Average effect of restoring subject tokens: {avg_subject_effect:.4f}")
    print(f"  Subject tokens: {result['input_tokens'][subj_start:subj_end+1]}")
    print()

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            output_path,
            scores=result['scores'],
            low_score=result['low_score'],
            high_score=result['high_score'],
            input_tokens=result['input_tokens'],
            subject_range=result['subject_range'],
            target_token=result['target_token'],
            prompt=args.prompt,
            subject=args.subject,
            model=args.model,
            samples=args.samples,
            noise_level=args.noise,
            component=args.component,
        )

        print(f"Results saved to: {output_path}")
        print()
        print("To visualize:")
        print(f"  python experiments_ssm/visualize_trace.py {output_path}")
        print("Or load in Jupyter notebook:")
        print(f"  data = np.load('{output_path}')")
        print("  scores = data['scores']")
    else:
        print("(No output file specified, results not saved)")

    print()
    print("=" * 70)
    print("Causal tracing complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
