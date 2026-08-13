"""Unified model and data loading for MI experiments.

Resolves checkpoint → config → model constructor args automatically.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Architecture configs
# ---------------------------------------------------------------------------

# Sudoku MLP-T TRM (matches run_sudoku.sh)
# seq_len is dynamically determined from domain, not hardcoded here
SUDOKU_ARCH_CONFIG = dict(
    hidden_size=512,
    num_heads=8,
    expansion=4,
    H_cycles=3,
    L_cycles=6,
    H_layers=0,
    L_layers=2,
    pos_encodings="none",
    forward_dtype="bfloat16",
    mlp_t=True,
    puzzle_emb_ndim=512,
    puzzle_emb_len=16,
    halt_exploration_prob=0.1,
    halt_max_steps=16,
    no_ACT_continue=True,
    batch_size=64,
    vocab_size=11,       # PAD + digits 1-9 + blank
    num_cells=81,       # 9x9 grid
    num_digits=9,
    cell_dim=10,
    num_puzzle_identifiers=1,
    causal=False,
)

# ARC attention TRM (matches run_arc.sh)
# Variable grid sizes supported - seq_len determined at runtime from model
ARC_ARCH_CONFIG = dict(
    hidden_size=512,
    num_heads=8,
    expansion=4,
    H_cycles=3,
    L_cycles=4,
    H_layers=0,
    L_layers=2,
    pos_encodings="rope",
    forward_dtype="bfloat16",
    mlp_t=False,
    puzzle_emb_ndim=512,
    puzzle_emb_len=16,
    halt_exploration_prob=0.1,
    halt_max_steps=16,
    no_ACT_continue=True,
    batch_size=128,
    vocab_size=12,       # PAD + EOS + 10 colors
    num_puzzle_identifiers=481,
    causal=False,
)


def _extract_num_cells_from_state_dict(state_dict: dict, default: int = 81) -> int:
    """
    Infer num_cells from state dict keys.

    Looks for weight matrices whose shape reveals the grid size.
    """
    patterns = [
        ("embed.weight", 1),
        ("trm_net", 2),
        ("output_head.weight", 0),
        ("H_init", 1),
        ("L_init", 1),
    ]

    for key, dim in patterns:
        for sd_key, tensor in state_dict.items():
            if key in sd_key and hasattr(tensor, "shape") and len(tensor.shape) > dim:
                shape = tensor.shape[dim]
                if shape > 10:
                    return shape

    return default


def get_model_seq_len(model: nn.Module) -> int:
    """
    Get the sequence length (num_cells) from a loaded model.

    Args:
        model: A TRM or Transformer model.

    Returns:
        Number of spatial positions (cells) in the grid.
    """
    if hasattr(model, "num_cells"):
        return model.num_cells
    elif hasattr(model, "grid_size"):
        return model.grid_size
    elif hasattr(model, "inner"):
        if hasattr(model.inner, "num_cells"):
            return model.inner.num_cells
        if hasattr(model.inner, "config"):
            return model.inner.config.get("num_cells", 81)
    elif hasattr(model, "trm_net") and hasattr(model.trm_net, "num_cells"):
        return model.trm_net.num_cells

    logger.warning("Could not determine seq_len from model, defaulting to 81")
    return 81


def _resolve_config(checkpoint_path: Path) -> dict[str, Any]:
    """Resolve config.json from logs/ directory matching checkpoint name.

    Checkpoint layout: checkpoints/<run_id>/best.pt
    Log layout:        logs/<run_id>/config.json
    """
    run_id = checkpoint_path.parent.name
    project_root = checkpoint_path.parent.parent.parent
    config_path = project_root / "logs" / run_id / "config.json"

    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)

    # Fallback: try to infer from run_id
    logger.warning("Config not found at %s, inferring from run_id", config_path)
    return _infer_config_from_run_id(run_id)


def _infer_config_from_run_id(run_id: str) -> dict[str, Any]:
    """Infer model config from run_id naming convention."""
    config: dict[str, Any] = {}
    if "transformer" in run_id:
        config["model_type"] = "transformer"
    elif "trm_v2" in run_id:
        config["model_type"] = "trm_v2"

    # Parse dim from run_id: e.g., "dim288" or "dim630"
    for part in run_id.split("-"):
        if part.startswith("dim"):
            config["model_dim"] = int(part[3:])
    return config


def _strip_compile_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Strip `_orig_mod.` prefix inserted by torch.compile."""
    clean = {}
    for k, v in state_dict.items():
        new_k = k.replace("_orig_mod.", "") if k.startswith("_orig_mod.") else k
        clean[new_k] = v
    return clean


def load_trm(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a SudokuTRMv2 model from a checkpoint.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        device: Device to load the model to.

    Returns:
        Tuple of (model, config_dict) where config includes seq_len and num_cells.
    """
    from src.models.trm import SudokuTRMv2

    checkpoint_path = Path(checkpoint_path)
    device = torch.device(device) if isinstance(device, str) else device
    config = _resolve_config(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    num_cells = config.get("num_cells", 81)
    num_digits = config.get("num_digits", 9)
    cell_dim = config.get("cell_dim", num_digits + 1)

    model_kwargs: dict[str, Any] = {
        "hidden_size": config.get("model_dim", 630),
        "num_heads": config.get("n_heads", 9),
        "num_layers": 2,
        "cell_dim": cell_dim,
        "num_cells": num_cells,
        "num_digits": num_digits,
        "mlp_t": True,
    }

    model = SudokuTRMv2(**model_kwargs)
    state_dict = _strip_compile_prefix(checkpoint["model_state_dict"])
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    config["seq_len"] = num_cells
    config["num_cells"] = num_cells

    logger.info(
        "Loaded TRMv2: hidden=%d, num_cells=%d, params=%.1fM",
        model_kwargs["hidden_size"],
        num_cells,
        sum(p.numel() for p in model.parameters()) / 1e6,
    )
    return model, config


def load_transformer(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a SudokuTransformer model from a checkpoint.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        device: Device to load the model to.

    Returns:
        Tuple of (model, config_dict) where config includes seq_len and grid_size.
    """
    from src.models.transformer import SudokuTransformer

    checkpoint_path = Path(checkpoint_path)
    device = torch.device(device) if isinstance(device, str) else device
    config = _resolve_config(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    num_cells = config.get("num_cells", config.get("grid_size", 81))
    num_digits = config.get("num_digits", 9)
    cell_vocab_size = num_digits + 1

    model_kwargs: dict[str, Any] = {
        "d_model": config.get("model_dim", 288),
        "n_heads": config.get("n_heads", 4),
        "d_ff": 512,
        "depth": config.get("depth", 8),
        "cell_vocab_size": cell_vocab_size,
        "grid_size": num_cells,
        "num_digits": num_digits,
        "dropout": 0.0,  # No dropout at eval time
    }

    model = SudokuTransformer(**model_kwargs)
    state_dict = _strip_compile_prefix(checkpoint["model_state_dict"])
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    config["seq_len"] = num_cells
    config["grid_size"] = num_cells

    logger.info(
        "Loaded Transformer: d_model=%d, grid_size=%d, depth=%d, params=%.1fM",
        model_kwargs["d_model"],
        num_cells,
        model_kwargs["depth"],
        sum(p.numel() for p in model.parameters()) / 1e6,
    )
    return model, config


# ---------------------------------------------------------------------------
# Original TRM (TinyRecursiveModels) support
# ---------------------------------------------------------------------------

class _OriginalTRMNetAdapter(nn.Module):
    """Wraps Original TRM's L_level forward logic while exposing .layers"""
    def __init__(self, inner_model: nn.Module, get_cos_sin_fn):
        super().__init__()
        self.inner = inner_model
        self.get_cos_sin = get_cos_sin_fn

    @property
    def layers(self):
        # Allow exp7 / exp8 to iterate over the actual transformer blocks
        return self.inner.L_level.layers

    def forward(self, *args) -> torch.Tensor:
        cos_sin = self.get_cos_sin()
        if len(args) == 3:
            x_emb, z_H, z_L = args
            return self.inner.L_level(z_L, z_H + x_emb, cos_sin=cos_sin)
        else:
            z_H, z_L = args
            return self.inner.L_level(z_H, z_L, cos_sin=cos_sin)


class OriginalTRMAdapter(nn.Module):
    """Wraps an Original TRM (TinyRecursiveReasoningModel_ACTV1_Inner) to
    expose a TRMv2-compatible API: embed / init_state / trm_net / output_head.

    This lets all MI experiment scripts run without modification.
    """

    def __init__(
        self,
        inner_model: nn.Module,
        puzzle_emb_len: int = 16,
        act_model: nn.Module | None = None,
        token_shift: int = 0,
    ):
        super().__init__()
        self.inner = inner_model          # TinyRecursiveReasoningModel_ACTV1_Inner
        self.puzzle_emb_len = puzzle_emb_len
        self.act_model = act_model        # TinyRecursiveReasoningModel_ACTV1 (ACT-loop wrapper)
        self.token_shift = token_shift    # Training tokenization shift vs raw one-hot (Sudoku: +1)
        self._cos_sin = None              # Will be set on first forward if needed
        self.trm_net = _OriginalTRMNetAdapter(self.inner, self._get_cos_sin)

        # Alias layer names so MI experiments can introspect them correctly
        for layer in self.inner.L_level.layers:
            # MLP-T: alias mlp_t as token_mixer
            if getattr(layer, "mlp_t", None) not in (None, False):
                if not hasattr(layer, "token_mixer"):
                    layer.token_mixer = layer.mlp_t
            # Attention TRM: alias self_attn as token_mixer
            elif hasattr(layer, "self_attn") and not hasattr(layer, "token_mixer"):
                layer.token_mixer = layer.self_attn
            # Channel mixer
            if hasattr(layer, "mlp") and not hasattr(layer, "channel_mixer"):
                layer.channel_mixer = layer.mlp

    # -- public TRMv2-compatible API ------------------------------------------

    def embed(self, x: torch.Tensor, puzzle_identifiers: torch.Tensor | None = None) -> torch.Tensor:
        """Embed input (either one-hot or integer indices)."""
        if x.dtype in (torch.float32, torch.float64, torch.bfloat16, torch.float16):
            # One-hot from Sudoku (B, 81, 10). Convert to integer class indices.
            # Training tokenization stores values as value + 1 (blank=1, digit d -> d+1),
            # so match that encoding here.
            x_int = x.argmax(dim=-1) + self.token_shift
        else:
            # Integer indices from ARC (B, 900)
            x_int = x

        if puzzle_identifiers is None:
            puzzle_identifiers = torch.zeros(x.size(0), dtype=torch.int32, device=x.device)

        return self.inner._input_embeddings(x_int, puzzle_identifiers)

    def init_state(
        self, batch_size: int, seq_len: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create zero-initialised z_H, z_L.

        Returns shapes matching the original model's internal sequence length
        (81 + puzzle_emb_len).
        """
        full_seq = seq_len  # Already includes puzzle_emb_len from embed output
        H = self.inner.config.hidden_size
        dtype = getattr(torch, self.inner.config.forward_dtype)

        z_H = self.inner.H_init.expand(batch_size, full_seq, H).clone().to(device=device, dtype=dtype)
        z_L = self.inner.L_init.expand(batch_size, full_seq, H).clone().to(device=device, dtype=dtype)
        return z_H, z_L

    def forward(
        self, x: torch.Tensor, puzzle_identifiers: torch.Tensor | None = None, T: int = 1, L_cycles: int = 1
    ) -> torch.Tensor:
        """Run a full forward pass matching TRMv2's calling convention.

        Args:
            x:        Input (one-hot or integers).
            puzzle_identifiers: Optional dataset IDs.
            T:        Number of H-level steps (H_cycles).
            L_cycles: Number of L-level steps per H step.

        Returns:
            Logits (B, S, vocab_size).
        """
        x_emb = self.embed(x, puzzle_identifiers=puzzle_identifiers)  # (B, S+P, H)
        B, S, _ = x_emb.shape
        z_H, z_L = self.init_state(B, S, x.device)

        cos_sin = self._get_cos_sin()
        seq_info = dict(cos_sin=cos_sin)

        for _ in range(T):
            for _ in range(L_cycles):
                z_L = self.inner.L_level(z_L, z_H + x_emb, **seq_info)
            z_H = self.inner.L_level(z_H, z_L, **seq_info)

        return self.output_head(z_H)

    def output_head(self, z_H: torch.Tensor) -> torch.Tensor:
        """Project z_H to logits (B, 81, vocab_size).

        OriginalTRM outputs 11 logits (0=blank, 1-9=digits).
        TRMv2 interface expects 9 logits mapping digits to 0-8.
        We strip the puzzle-embedding prefix positions and slice logits [..., 1:10].
        """
        return self.inner.lm_head(z_H)[:, self.puzzle_emb_len:, 1:10]

    # -- faithful evaluation (ACT recursion loop) -------------------------------

    def act_forward(
        self,
        x: torch.Tensor,
        puzzle_identifiers: torch.Tensor | None = None,
        max_steps: int | None = None,
    ) -> torch.Tensor:
        """Run the ACT recursion loop used during training / canonical evaluation.

        Mirrors pretrain.py's ``evaluate()``: recycle the carry across
        ``halt_max_steps`` ACT steps (each step runs the full T x L inner
        recursion) and return the final step's full-vocabulary logits. The
        single-pass ``forward()`` (used by MI experiments) only runs ACT step 1,
        which under-scores trained models.

        Args:
            x: Input (one-hot Sudoku or integer ARC tokens).
            puzzle_identifiers: Optional dataset IDs.
            max_steps: Number of ACT steps (defaults to the model's halt_max_steps).

        Returns:
            Logits (B, S, vocab_size) with the full vocabulary, matching training.
        """
        if self.act_model is None:
            raise RuntimeError("act_forward requires the ACT wrapper (act_model)")

        if x.dtype in (torch.float32, torch.float64, torch.bfloat16, torch.float16):
            x_int = x.argmax(dim=-1) + self.token_shift
        else:
            x_int = x

        B = x.size(0)
        device = x.device
        if puzzle_identifiers is None:
            puzzle_identifiers = torch.zeros(B, dtype=torch.int32, device=device)

        batch = {
            "inputs": x_int.to(torch.int32),
            "puzzle_identifiers": puzzle_identifiers.to(torch.int32),
        }

        max_steps = max_steps or getattr(self.act_model.config, "halt_max_steps", 16)

        carry = self.act_model.initial_carry(batch)
        outputs = None
        for _ in range(max_steps):
            carry, outputs = self.act_model(carry, batch)

        return outputs["logits"]

    # -- helpers --------------------------------------------------------------

    def _get_cos_sin(self):
        """Lazily compute rotary embeddings if the model uses them."""
        if hasattr(self.inner, "rotary_emb"):
            if self._cos_sin is None:
                self._cos_sin = self.inner.rotary_emb()
            return self._cos_sin
        return None


def load_original_trm(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Load an Original TRM checkpoint and return a TRMv2-compatible adapter.

    Args:
        checkpoint_path: Path to the checkpoint file (e.g. step_296875.pt).
        device: Device to load the model to.

    Returns:
        Tuple of (OriginalTRMAdapter, config_dict).
    """
    import sys
    import glob
    trm_dir = Path(__file__).resolve().parent.parent
    if str(trm_dir) not in sys.path:
        sys.path.insert(0, str(trm_dir))
    # Also add the TinyRecursiveModels venv site-packages for pydantic etc.
    venv_sp = glob.glob(str(trm_dir / ".venv" / "lib" / "python*" / "site-packages"))
    for sp in venv_sp:
        if sp not in sys.path:
            sys.path.insert(0, sp)

    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1

    checkpoint_path = Path(checkpoint_path)
    device = torch.device(device) if isinstance(device, str) else device

    # Default architecture config (matches run_sudoku.sh and trm.yaml)
    arch_config = SUDOKU_ARCH_CONFIG.copy()

    # Load checkpoint first to determine num_cells
    raw_sd = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(raw_sd, dict) and "model_state_dict" in raw_sd:
        raw_sd = raw_sd["model_state_dict"]

    clean_sd: dict[str, Any] = {}
    for k, v in raw_sd.items():
        k = k.replace("_orig_mod.", "")
        if k.startswith("model."):
            k = k[len("model."):]
        clean_sd[k] = v

    # Extract num_cells from loaded model weights
    num_cells = _extract_num_cells_from_state_dict(clean_sd, default=81)
    arch_config["num_cells"] = num_cells
    arch_config["seq_len"] = num_cells
    arch_config["num_digits"] = 9
    arch_config["cell_dim"] = 10

    # Instantiate the full model (ACT wrapper included for state_dict compat)
    model = TinyRecursiveReasoningModel_ACTV1(arch_config)
    model.load_state_dict(clean_sd, strict=False)
    model.to(device)

    # Wrap inner model in adapter, keeping the ACT wrapper for faithful evaluation
    puzzle_emb_len = arch_config["puzzle_emb_len"]
    adapter = OriginalTRMAdapter(
        model.inner, puzzle_emb_len=puzzle_emb_len,
        act_model=model, token_shift=1,
    )
    adapter.to(device)
    adapter.eval()

    num_params = sum(p.numel() for p in adapter.parameters())
    logger.info(
        "Loaded Original TRM: hidden=%d, num_cells=%d, params=%.1fM",
        arch_config["hidden_size"],
        num_cells,
        num_params / 1e6,
    )

    return adapter, arch_config


def load_arc_trm(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Load an Original TRM checkpoint trained on ARC (attention variant).

    Uses ARC_ARCH_CONFIG: mlp_t=False, pos_encodings=rope, L_cycles=4,
    vocab_size=12 (PAD+EOS+10 colors), seq_len=900 (30x30 grid).

    Returns:
        Tuple of (OriginalTRMAdapter, config_dict).
    """
    import sys
    import glob
    trm_dir = Path(__file__).resolve().parent.parent
    if str(trm_dir) not in sys.path:
        sys.path.insert(0, str(trm_dir))
    venv_sp = glob.glob(str(trm_dir / ".venv" / "lib" / "python*" / "site-packages"))
    for sp in venv_sp:
        if sp not in sys.path:
            sys.path.insert(0, sp)

    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1

    checkpoint_path = Path(checkpoint_path)
    device = torch.device(device) if isinstance(device, str) else device

    arch_config = ARC_ARCH_CONFIG.copy()

    raw_sd = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(raw_sd, dict) and "model_state_dict" in raw_sd:
        raw_sd = raw_sd["model_state_dict"]

    clean_sd: dict[str, Any] = {}
    for k, v in raw_sd.items():
        k = k.replace("_orig_mod.", "")
        if k.startswith("model."):
            k = k[len("model."):]
        clean_sd[k] = v

    # Extract dynamic num_puzzle_identifiers
    for k, v in raw_sd.items():
        if k.endswith("inner.puzzle_emb.weights"):
            arch_config["num_puzzle_identifiers"] = v.shape[0]
            break

    # Extract num_cells from state dict
    num_cells = _extract_num_cells_from_state_dict(clean_sd, default=900)
    arch_config["num_cells"] = num_cells
    arch_config["seq_len"] = num_cells

    model = TinyRecursiveReasoningModel_ACTV1(arch_config)
    model.load_state_dict(clean_sd, strict=False)
    model.to(device)

    puzzle_emb_len = arch_config["puzzle_emb_len"]
    adapter = OriginalTRMAdapter(
        model.inner, puzzle_emb_len=puzzle_emb_len,
        act_model=model, token_shift=0,
    )
    adapter.to(device)
    adapter.eval()

    num_params = sum(p.numel() for p in adapter.parameters())
    logger.info(
        "Loaded ARC TRM (attention): hidden=%d, num_cells=%d, params=%.1fM",
        arch_config["hidden_size"],
        num_cells,
        num_params / 1e6,
    )
    return adapter, arch_config


def resolve_matched_checkpoint(run_dir: str | Path, matched_budget: int) -> Path:
    """Find the step checkpoint closest to matched_budget in run_dir.

    Args:
        run_dir: Path to the run directory (or a specific step checkpoint inside it)
        matched_budget: The target number of updates (e.g., 19000000)
    """
    path = Path(run_dir)
    # If a specific step file was passed, resolve to its parent run_dir
    if path.is_file() and path.name.startswith("step_"):
        path = path.parent
    elif path.is_dir() and path.name.startswith("step_"):
        path = path.parent

    step_paths = list(path.glob("step_*"))
    if not step_paths:
        raise ValueError(f"No step_* checkpoints found in {path}")

    best_path = None
    min_diff = float("inf")
    
    for sp in step_paths:
        try:
            # Parse step number from name "step_12345"
            step_num = int(sp.name.split("_")[1])
            diff = abs(step_num - matched_budget)
            if diff < min_diff:
                min_diff = diff
                best_path = sp
        except (ValueError, IndexError):
            continue

    if best_path is None:
        raise ValueError(f"Could not parse any valid step_* checkpoints in {path}")
        
    logger.info("Matched budget requested: %s. Closest step found: %s (diff: %d)",
                matched_budget, best_path.name, min_diff)
    return best_path


def load_model(
    checkpoint_path: str | Path,
    model_type: str = "trm_v2",
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Unified model loader that dispatches to the correct loader.

    Args:
        checkpoint_path: Path to the checkpoint file.
        model_type: 'trm_v2', 'original_trm', 'arc_trm', or 'transformer'.
        device: Device to load the model to.

    Returns:
        Tuple of (model, config_dict).
    """
    if model_type == "arc_trm":
        return load_arc_trm(checkpoint_path, device)
    elif model_type == "original_trm":
        return load_original_trm(checkpoint_path, device)
    elif model_type == "transformer":
        return load_transformer(checkpoint_path, device)
    else:
        return load_trm(checkpoint_path, device)


def get_arc_dataloader(
    dataset_dir: str | Path,
    num_samples: int = 500,
    batch_size: int = 32,
    split: str = "test",
    seed: int = 0,
) -> DataLoader:
    """Create a DataLoader from a built ARC PuzzleDataset directory.

    Args:
        dataset_dir: Path to dataset root (e.g. data/arc-concept-n1000-aug2).
        num_samples: Maximum number of examples to yield.
        batch_size: Batch size.
        split: 'train' or 'test'.
        seed: Random seed.

    Returns:
        DataLoader yielding (inputs, labels) tensors with shape (B, 900).
    """
    import sys
    trm_dir = Path(__file__).resolve().parent.parent
    if str(trm_dir) not in sys.path:
        sys.path.insert(0, str(trm_dir))

    from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig

    config = PuzzleDatasetConfig(
        seed=seed,
        dataset_paths=[str(dataset_dir)],
        global_batch_size=batch_size,
        test_set_mode=(split == "test"),
        epochs_per_iter=1,
        rank=0,
        num_replicas=1,
    )
    ds = PuzzleDataset(config, split=split)

    collected: list[tuple[torch.Tensor, torch.Tensor]] = []
    total = 0
    for _, batch, _ in ds:
        inp = batch["inputs"]   # (B, 900)
        lbl = batch["labels"]   # (B, 900)
        collected.append((inp, lbl))
        total += inp.size(0)
        if total >= num_samples:
            break

    if not collected:
        return DataLoader([], batch_size=batch_size)

    all_inp = torch.cat([x for x, _ in collected], dim=0)[:num_samples]
    all_lbl = torch.cat([y for _, y in collected], dim=0)[:num_samples]
    dataset = torch.utils.data.TensorDataset(all_inp, all_lbl)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# Cache for test datasets: avoids re-downloading HuggingFace on every checkpoint
_test_dataset_cache: dict[tuple, Any] = {}


def get_test_dataloader(
    num_samples: int = 500,
    num_blanks: int = 8,
    batch_size: int = 64,
    seed: int = 0,
    dataset: str = "extreme",
) -> DataLoader:
    """Create a Sudoku test DataLoader (cached — only downloads once)."""
    cache_key = (num_samples, num_blanks, batch_size, seed, dataset)
    if cache_key in _test_dataset_cache:
        return _test_dataset_cache[cache_key]

    if dataset == "extreme":
        from src.data.tasks.sudoku import SudokuExtremeTask, SudokuTaskConfig

        config = SudokuTaskConfig(
            test_samples=num_samples,
            train_samples=100,
        )
        task = SudokuExtremeTask(config)
        test_ds = task.get_test_dataset()
    else:
        from src.data.sudoku import SudokuDataset

        torch.manual_seed(seed)
        test_ds = SudokuDataset(
            num_samples=num_samples,
            grid_size=int(num_blanks**0.5) if num_blanks <= 16 else 9,
            num_blanks=num_blanks,
        )

    dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    _test_dataset_cache[cache_key] = dl
    return dl


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
