"""Evaluate model accuracy on Sudoku and ConceptARC.

Computes:
- Cell-level accuracy (fraction of correct cells)
- Puzzle-level (solved-board) accuracy (all cells correct)
- Pass@k: proportion of puzzles solved within k attempts
  (uses temperature sampling from output logits)

Usage:
  # Sudoku
  python scripts/evaluate_model.py --checkpoint path/to/checkpoint.pt \\
      --model-type original_trm --domain sudoku --num-test 1000 --pass-at-k 10

  # ConceptARC
  python scripts/evaluate_model.py --checkpoint path/to/checkpoint.pt \\
      --model-type arc_trm --domain arc --dataset-dir data/arc-concept-n1000 \\
      --num-test 500 --pass-at-k 10 --T 4 --L-cycles 4
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.model_loader import (
    load_model,
    get_arc_dataloader,
    get_test_dataloader,
)


@torch.no_grad()
def sudoku_accuracy(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    T: int = 32,
    L_cycles: int = 6,
    max_act_steps: int | None = None,
) -> tuple[float, float]:
    """Cell-level and puzzle-level accuracy for Sudoku."""
    model.eval()
    total_correct = 0
    total_cells = 0
    total_solved = 0
    total_puzzles = 0

    # Original TRM models are trained with the ACT recursion loop (carry recycled
    # across halt_max_steps steps); a single pass under-scores them. act_forward
    # runs the canonical ACT-loop eval and returns full-vocab (11-class) logits.
    use_act = hasattr(model, "act_forward")

    for x, labels in dataloader:
        x = x.to(device)
        labels = labels.to(device)
        if labels.dim() == 3:
            labels = labels.argmax(dim=-1)

        if use_act:
            logits = model.act_forward(x, max_steps=max_act_steps)
            # Training tokenizes labels as value + 1 (digit d -> d+1), so the
            # dataset's 0..8 classes map to 2..10 in the model's vocabulary.
            labels = labels + 2
        else:
            logits = model(x, T=T, L_cycles=L_cycles)
        preds = logits.argmax(dim=-1)

        correct = preds == labels
        total_correct += correct.sum().item()
        total_cells += labels.numel()
        total_solved += correct.all(dim=1).sum().item()
        total_puzzles += labels.size(0)

    cell_acc = total_correct / total_cells
    puzzle_acc = total_solved / total_puzzles
    return cell_acc, puzzle_acc


@torch.no_grad()
def arc_accuracy(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    T: int = 4,
    L_cycles: int = 4,
    max_act_steps: int | None = None,
) -> tuple[float, float]:
    """Cell-level and puzzle-level accuracy for ConceptARC."""
    model.eval()
    total_correct = 0
    total_cells = 0
    total_solved = 0
    total_puzzles = 0

    use_act = hasattr(model, "act_forward")

    for x, labels in dataloader:
        x = x.to(device)
        labels = labels.to(device)

        if use_act:
            # Full-vocab logits (B, 900, 12) from the ACT loop; ARC labels are
            # already in the training tokenization (0=PAD, 1=EOS, colors 2..11).
            logits = model.act_forward(x, max_steps=max_act_steps)
        else:
            # Bypass output_head's Sudoku-specific logit slice, capture full lm_head
            captured = [None]

            def _capture(m, _inp, out):
                captured[0] = out.detach()

            handle = model.inner.lm_head.register_forward_hook(_capture)
            _ = model(x, T=T, L_cycles=L_cycles)
            handle.remove()

            logits = captured[0][:, model.puzzle_emb_len:, :]
        preds = logits.argmax(dim=-1)

        correct = preds == labels
        total_correct += correct.sum().item()
        total_cells += labels.numel()
        total_solved += correct.all(dim=1).sum().item()
        total_puzzles += labels.size(0)

    cell_acc = total_correct / total_cells
    puzzle_acc = total_solved / total_puzzles
    return cell_acc, puzzle_acc


@torch.no_grad()
def sudoku_pass_at_k(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    k: int = 10,
    temperature: float = 1.0,
    T: int = 32,
    L_cycles: int = 6,
    max_act_steps: int | None = None,
) -> float:
    """Pass@k for Sudoku via temperature sampling from output logits."""
    model.eval()
    puzzles_solved = 0
    total = 0

    use_act = hasattr(model, "act_forward")

    for x, labels in dataloader:
        x = x.to(device)
        labels = labels.to(device)
        if labels.dim() == 3:
            labels = labels.argmax(dim=-1)

        B = x.size(0)
        x_k = x.repeat_interleave(k, dim=0)

        if use_act:
            logits = model.act_forward(x_k, max_steps=max_act_steps)
            # Sample over the digit classes only (vocab 2..10 <-> classes 0..8),
            # which matches the dataset's 0..8 labels.
            probs = torch.softmax(logits[..., 2:11] / temperature, dim=-1)
        else:
            logits = model(x_k, T=T, L_cycles=L_cycles)
            probs = torch.softmax(logits / temperature, dim=-1)

        samples = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(B * k, -1)

        solved = samples.view(B, k, -1).eq(labels.unsqueeze(1)).all(dim=-1).any(dim=1)
        puzzles_solved += solved.sum().item()
        total += B

    return puzzles_solved / total


@torch.no_grad()
def arc_pass_at_k(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    k: int = 10,
    temperature: float = 1.0,
    T: int = 4,
    L_cycles: int = 4,
    max_act_steps: int | None = None,
) -> float:
    """Pass@k for ConceptARC via temperature sampling."""
    model.eval()
    puzzles_solved = 0
    total = 0

    use_act = hasattr(model, "act_forward")

    for x, labels in dataloader:
        x = x.to(device)
        labels = labels.to(device)

        B = x.size(0)
        x_k = x.repeat_interleave(k, dim=0)

        if use_act:
            logits = model.act_forward(x_k, max_steps=max_act_steps)
        else:
            captured = [None]

            def _capture(m, _inp, out):
                captured[0] = out.detach()

            handle = model.inner.lm_head.register_forward_hook(_capture)
            _ = model(x_k, T=T, L_cycles=L_cycles)
            handle.remove()

            logits = captured[0][:, model.puzzle_emb_len:, :]
        probs = torch.softmax(logits / temperature, dim=-1)
        samples = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(B * k, -1)

        solved = samples.view(B, k, -1).eq(labels.unsqueeze(1)).all(dim=-1).any(dim=1)
        puzzles_solved += solved.sum().item()
        total += B

    return puzzles_solved / total


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model accuracy on Sudoku and ConceptARC"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--model-type", type=str, default="original_trm",
        choices=["trm_v2", "original_trm", "arc_trm", "transformer"],
    )
    parser.add_argument(
        "--domain", type=str, default="sudoku", choices=["sudoku", "arc"],
    )
    parser.add_argument(
        "--dataset-dir", type=str, default=None,
        help="ConceptARC dataset directory (required for --domain arc)",
    )
    parser.add_argument("--num-test", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--T", type=int, default=None,
                        help="H-cycles / outer recursion depth (default: 3 for both domains)")
    parser.add_argument("--L-cycles", type=int, default=None,
                        help="L-cycles (default: 6 for sudoku, 4 for arc)")
    parser.add_argument("--pass-at-k", type=int, default=0,
                        help="Compute Pass@k (k=0 to skip)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature for Pass@k")
    parser.add_argument("--max-act-steps", type=int, default=None,
                        help="ACT recursion steps for original_trm/arc_trm "
                             "(default: the model's halt_max_steps)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-json", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"Loading {args.model_type} from {args.checkpoint}...")
    model, config = load_model(args.checkpoint, model_type=args.model_type, device=device)
    model.eval()

    T = args.T or 3  # H_cycles from training config
    L_cycles = args.L_cycles or (4 if args.domain == "arc" else 6)

    results = {
        "checkpoint": str(args.checkpoint),
        "model_type": args.model_type,
        "domain": args.domain,
        "T": T,
        "L_cycles": L_cycles,
        "num_test": args.num_test,
    }

    if args.domain == "sudoku":
        dataloader = get_test_dataloader(
            num_samples=args.num_test, num_blanks=10,
            batch_size=args.batch_size,
        )
        cell_acc, puzzle_acc = sudoku_accuracy(
            model, dataloader, device, T=T, L_cycles=L_cycles,
            max_act_steps=args.max_act_steps,
        )
        results["cell_accuracy"] = cell_acc
        results["puzzle_accuracy"] = puzzle_acc
        print(f"Cell accuracy:   {cell_acc:.4f}")
        print(f"Puzzle accuracy: {puzzle_acc:.4f}")

        if args.pass_at_k > 0:
            pass_k = sudoku_pass_at_k(
                model, dataloader, device,
                k=args.pass_at_k, temperature=args.temperature,
                T=T, L_cycles=L_cycles, max_act_steps=args.max_act_steps,
            )
            results[f"pass@{args.pass_at_k}"] = pass_k
            print(f"Pass@{args.pass_at_k}:  {pass_k:.4f}")

    elif args.domain == "arc":
        if args.dataset_dir is None:
            print("error: --dataset-dir is required for --domain arc")
            sys.exit(1)
        dataloader = get_arc_dataloader(
            args.dataset_dir, num_samples=args.num_test,
            batch_size=args.batch_size, split="test",
        )
        cell_acc, puzzle_acc = arc_accuracy(
            model, dataloader, device, T=T, L_cycles=L_cycles,
            max_act_steps=args.max_act_steps,
        )
        results["cell_accuracy"] = cell_acc
        results["puzzle_accuracy"] = puzzle_acc
        print(f"Cell accuracy:   {cell_acc:.4f}")
        print(f"Puzzle accuracy: {puzzle_acc:.4f}")

        if args.pass_at_k > 0:
            pass_k = arc_pass_at_k(
                model, dataloader, device,
                k=args.pass_at_k, temperature=args.temperature,
                T=T, L_cycles=L_cycles, max_act_steps=args.max_act_steps,
            )
            results[f"pass@{args.pass_at_k}"] = pass_k
            print(f"Pass@{args.pass_at_k}:  {pass_k:.4f}")

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {args.save_json}")


if __name__ == "__main__":
    main()
