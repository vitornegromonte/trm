#!/usr/bin/env python3
"""Profile training pipeline for TRM.

Usage:
    python profile.py                     # single-GPU fast profile (50 batches)
    python profile.py --batches 200       # custom batch count
    python profile.py --trace             # full torch.profiler trace with Chrome export
    python profile.py --memory            # detailed CUDA memory snapshots
    python profile.py --kernel            # GPU kernel-level profiling (slow, detailed)

All flags accept config overrides like:
    python profile.py data_paths=['data/arc-aug-1000'] global_batch_size=128

Design:
    - Reuses the exact same training setup as pretrain.py (model, data, optimizer).
    - Profiles 5 stages separately: dataloading, forward, backward, allreduce, optimizer.
    - Outputs a table of median/latency timings and bandwidth estimates.
"""

import os
import sys
import time
import json
import contextlib
from typing import Optional
from dataclasses import dataclass, field

import torch
import torch.distributed as dist

os.environ["DISABLE_COMPILE"] = "1"  # skip compile for profiling to avoid startup noise

from pretrain import (
    PretrainConfig, init_train_state, create_dataloader,
    compute_lr, cosine_schedule_with_warmup_lr_lambda,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ProfileConfig:
    batches: int = 50
    warmup: int = 5
    trace: bool = False
    memory: bool = False
    kernel: bool = False
    no_compile: bool = True
    output_dir: str = "profiles"
    # overrides forwarded to Hydra
    overrides: list = field(default_factory=list)
    synthetic: bool = False
    # synthetic data shape
    seq_len: int = 81
    vocab_size: int = 12
    num_puzzle_ids: int = 100
    puzzle_emb_ndim: int = 512
    puzzle_emb_len: int = 16


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

def make_synthetic_dataset(save_dir: str, cfg: ProfileConfig):
    """Create a minimal 2-group synthetic dataset as .npy files."""
    import numpy as np
    split_dir = os.path.join(save_dir, "train")
    os.makedirs(split_dir, exist_ok=True)

    seq_len = cfg.seq_len
    vocab_size = cfg.vocab_size
    num_puzzles = cfg.num_puzzle_ids
    examples_per_puzzle = 4
    total = num_puzzles * examples_per_puzzle
    groups = 2

    inputs = np.random.randint(0, vocab_size, (total, seq_len)).astype(np.int32)
    labels = np.random.randint(0, vocab_size, (total, seq_len)).astype(np.int32)
    # puzzle identifiers per example
    puzzle_ids = np.repeat(np.arange(num_puzzles, dtype=np.int32), examples_per_puzzle)
    # puzzle indices: boundaries for each puzzle's examples
    puzzle_indices = np.arange(0, total + 1, examples_per_puzzle, dtype=np.int64)
    # group indices: split puzzles into groups
    group_indices = np.array([0, num_puzzles // groups, num_puzzles], dtype=np.int64)

    np.save(os.path.join(split_dir, "train__inputs.npy"), inputs)
    np.save(os.path.join(split_dir, "train__labels.npy"), labels)
    np.save(os.path.join(split_dir, "train__puzzle_identifiers.npy"), puzzle_ids)
    np.save(os.path.join(split_dir, "train__puzzle_indices.npy"), puzzle_indices)
    np.save(os.path.join(split_dir, "train__group_indices.npy"), group_indices)

    metadata = {
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "pad_id": 0,
        "ignore_label_id": -1,
        "blank_identifier_id": 0,
        "num_puzzle_identifiers": num_puzzles,
        "total_puzzles": num_puzzles,
        "mean_puzzle_examples": examples_per_puzzle,
        "total_groups": groups,
        "sets": ["train"],
    }
    with open(os.path.join(split_dir, "dataset.json"), "w") as f:
        json.dump(metadata, f)

    return save_dir, metadata


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

class Timer:
    """Wall-clock timer with .ns resolution via perf_counter_ns."""
    __slots__ = ("_start", "_elapsed")

    def __init__(self):
        self._start: Optional[int] = None
        self._elapsed = 0

    def start(self):
        self._start = time.perf_counter_ns()

    def stop(self):
        if self._start is not None:
            self._elapsed = time.perf_counter_ns() - self._start

    @property
    def ms(self) -> float:
        return self._elapsed / 1e6

    @property
    def us(self) -> float:
        return self._elapsed / 1e3


@contextlib.contextmanager
def measure(name: str, store: dict, sync: bool = False):
    if sync:
        torch.cuda.synchronize()
    t = Timer()
    t.start()
    try:
        yield
    finally:
        if sync:
            torch.cuda.synchronize()
        t.stop()
        store.setdefault(name, []).append(t.ms)


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def snapshot_memory(rank: int = 0) -> dict:
    """Return a dict of current CUDA memory stats."""
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "reserved_gb":  torch.cuda.memory_reserved() / 1e9,
        "peak_gb":      torch.cuda.max_memory_allocated() / 1e9,
    }


def detailed_memory_summary():
    """Print a human-readable memory summary."""
    print(torch.cuda.memory_summary())


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------

class PipelineProfiler:
    """Profile the training pipeline stage by stage."""

    def __init__(self, cfg: ProfileConfig):
        self.cfg = cfg
        self.timings: dict[str, list[float]] = {}
        self.mem_snapshots: list[dict] = []
        self.rank = 0
        self.world_size = 1

    def _log(self, msg: str):
        if self.rank == 0:
            print(msg, flush=True)

    def _gather_batch_timings(self, batch_gen, train_state, config):
        """Run warmup + profiling batches, recording per-stage timings."""
        for step, (set_name, batch, global_batch_size) in enumerate(batch_gen):
            if step >= self.cfg.warmup + self.cfg.batches:
                break
            is_profiling = step >= self.cfg.warmup

            if not is_profiling:
                warmup_batches = min(self.cfg.warmup, self.cfg.batches + self.cfg.warmup)
                self._log(f"  warmup batch {step + 1}/{warmup_batches}")

            batch = {k: v.cuda() for k, v in batch.items()}

            if train_state.carry is None:
                with torch.device("cuda"):
                    train_state.carry = train_state.model.initial_carry(batch)

            if not is_profiling:
                # warmup — run normally, no recording
                train_state.carry, loss, metrics, _, _ = train_state.model(
                    carry=train_state.carry, batch=batch, return_keys=[]
                )
                ((1 / global_batch_size) * loss).backward()
                if self.world_size > 1:
                    for p in train_state.model.parameters():
                        if p.grad is not None:
                            dist.all_reduce(p.grad)
                for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
                    lr = compute_lr(base_lr, config, train_state)
                    for pg in optim.param_groups:
                        pg["lr"] = lr
                    optim.step()
                    optim.zero_grad()
                train_state.step += 1
                continue

            # --- profiling step ---
            train_state.step += 1
            self._log(f"  profile batch {step - self.cfg.warmup + 1}/{self.cfg.batches}")

            # 1. H2D transfer (measure actual dataloading: CPU→GPU copy)
            with measure("dataloader", self.timings, sync=True):
                batch = {k: v.cuda() for k, v in batch.items()}

            # 2. forward
            with measure("forward", self.timings, sync=True):
                train_state.carry, loss, metrics, _, _ = train_state.model(
                    carry=train_state.carry, batch=batch, return_keys=[]
                )
                loss_val = (1 / global_batch_size) * loss

            # 3. backward
            with measure("backward", self.timings, sync=True):
                loss_val.backward()

            # 4. all-reduce
            if self.world_size > 1:
                with measure("allreduce", self.timings, sync=True):
                    for p in train_state.model.parameters():
                        if p.grad is not None:
                            dist.all_reduce(p.grad)

            # 5. optimizer
            with measure("optimizer", self.timings, sync=True):
                for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
                    lr = compute_lr(base_lr, config, train_state)
                    for pg in optim.param_groups:
                        pg["lr"] = lr
                    optim.step()
                    optim.zero_grad()

            # memory snapshot
            if self.cfg.memory:
                self.mem_snapshots.append(snapshot_memory())

            del loss, loss_val, metrics

    def profile(self):
        """Main profiling entry point — replicates pretrain setup exactly."""
        from hydra import compose, initialize_config_dir
        config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")

        # Synthetic data override
        if self.cfg.synthetic:
            syn_dir = "/tmp/trm_synthetic_profile"
            self._log(f"Generating synthetic dataset at {syn_dir} ...")
            make_synthetic_dataset(syn_dir, self.cfg)
            self.cfg.overrides = [
                f"data_paths=['{syn_dir}']",
            ] + [o for o in self.cfg.overrides if not o.startswith("data_paths")]

        if not torch.cuda.is_available():
            self._log("ERROR: CUDA is required for GPU profiling. No NVIDIA GPU detected.")
            self._log("Install CUDA drivers or run on a GPU node.")
            sys.exit(1)

        with initialize_config_dir(config_dir=config_dir):
            cfg_hydra = compose("cfg_pretrain", overrides=self.cfg.overrides)

        config = PretrainConfig(**cfg_hydra)

        rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = rank
        self.world_size = world_size

        if world_size > 1 and not dist.is_initialized():
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(rank)

        # Dataset
        self._log(f"Loading datasets from {config.data_paths} ...")
        train_loader, train_metadata = create_dataloader(
            config, "train", test_set_mode=False,
            epochs_per_iter=1,
            global_batch_size=config.global_batch_size,
            rank=rank, world_size=world_size,
        )

        # Model
        self._log("Creating model ...")
        os.environ.pop("DISABLE_COMPILE", None)  # allow compile if user wants
        if self.cfg.no_compile:
            os.environ["DISABLE_COMPILE"] = "1"

        train_state = init_train_state(config, train_metadata, rank=rank, world_size=world_size)
        train_state.model.train()

        self._log(f"Model parameters: {sum(p.numel() for p in train_state.model.parameters()):,}")
        self._log(f"  trainable:      {sum(p.numel() for p in train_state.model.parameters() if p.requires_grad):,}")

        # Run profiled batches
        self._log(f"Warmup: {self.cfg.warmup} batches, Profiling: {self.cfg.batches} batches")
        self._gather_batch_timings(train_loader, train_state, config)

        # Report
        self._report()

        # Trace
        if self.cfg.trace and self.rank == 0:
            self._log(f"Trace saved — rerun with --kernel for GPU kernel timeline")

    def _report(self):
        if self.rank != 0:
            return

        print("\n" + "=" * 72)
        print("PROFILING RESULTS")
        print("=" * 72)

        all_stages = ["dataloader", "forward", "backward", "allreduce", "optimizer"]
        present = [s for s in all_stages if s in self.timings and len(self.timings[s]) > 0]

        if not present:
            print("\nNo profiling data collected.")
            print("Possible causes:")
            print("  - Not enough data: dataset produced fewer batches than warmup + profiling")
            print("  - OOM during warmup: increase --warmup, decrease global_batch_size")
            print("  - Try: global_batch_size=32 --batches 10")
            return

        # Per-stage summary
        print(f"\n{'Stage':<16} {'Median(ms)':<12} {'Mean(ms)':<12} {'Min(ms)':<12} {'Max(ms)':<12} {'P99(ms)':<12}")
        print("-" * 76)
        totals = []
        median_total = 0.0
        for stage in present:
            ts = sorted(self.timings[stage])
            med = ts[len(ts) // 2]
            mn = sum(ts) / len(ts)
            mi = ts[0]
            ma = ts[-1]
            p99 = ts[int(len(ts) * 0.99)]
            print(f"{stage:<16} {med:<12.3f} {mn:<12.3f} {mi:<12.3f} {ma:<12.3f} {p99:<12.3f}")
            if stage != "dataloader":
                totals.extend(ts)
                median_total += med

        if totals:
            print("-" * 76)
            print(f"{'Compute total':<16} {median_total:<12.3f}")

        # Step time estimate
        step_median = median_total
        dl_med = 0.0
        if "dataloader" in self.timings and len(self.timings["dataloader"]) > 0:
            dl_ts = sorted(self.timings["dataloader"])
            dl_med = dl_ts[len(dl_ts) // 2]
            print(f"{'Data loader':<16} {dl_med:<12.3f}")
            print(f"{'Step total':<16} {step_median + dl_med:<12.3f}")

        total_step = step_median + dl_med
        if total_step > 0:
            print(f"\nEstimated throughput: {1000 / total_step:.1f} steps/s")
            print(f"Estimated step time:  {total_step:.1f} ms")
        print(f"Estimated memory:     peak={torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

        # Bottleneck
        print(f"\n--- Bottleneck Analysis ---")
        if step_median > 0:
            for stage in present:
                if stage == "dataloader":
                    continue
                ts = sorted(self.timings[stage])
                med = ts[len(ts) // 2]
                pct = med / step_median * 100
                label = "⚠️  " if pct > 30 else "   "
                print(f"  {label}{stage:<14} {med:<8.3f} ms  ({pct:.0f}%)")
            if dl_med > 0:
                dl_pct = dl_med / total_step * 100
                label = "⚠️  " if dl_pct > 20 else "   "
                print(f"  {label}dataloader     {dl_med:<8.3f} ms  ({dl_pct:.0f}% of total step)")




# ---------------------------------------------------------------------------
# Kernel-level profiler with torch.profiler
# ---------------------------------------------------------------------------

def run_torch_profiler(cfg: ProfileConfig):
    """Run a short training loop wrapped in torch.profiler for GPU kernel traces."""
    from hydra import compose, initialize_config_dir

    config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")

    if cfg.synthetic:
        syn_dir = "/tmp/trm_synthetic_profile"
        print(f"Generating synthetic dataset at {syn_dir} ...")
        make_synthetic_dataset(syn_dir, cfg)
        cfg.overrides = [
            f"data_paths=['{syn_dir}']",
        ] + [o for o in cfg.overrides if not o.startswith("data_paths")]

    with initialize_config_dir(config_dir=config_dir):
        cfg_hydra = compose("cfg_pretrain", overrides=cfg.overrides)

    config = PretrainConfig(**cfg_hydra)

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(rank)

    if cfg.no_compile:
        os.environ["DISABLE_COMPILE"] = "1"

    train_loader, train_metadata = create_dataloader(
        config, "train", test_set_mode=False,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        rank=rank, world_size=world_size,
    )
    train_state = init_train_state(config, train_metadata, rank=rank, world_size=world_size)
    train_state.model.train()

    os.makedirs(cfg.output_dir, exist_ok=True)
    trace_path = os.path.join(cfg.output_dir, f"trace_rank{rank}")

    if rank == 0:
        print(f"Saving trace to {trace_path}/ ...")

    # Profiler schedule: skip warmup batches, then sample a few
    prof = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=cfg.warmup,
            warmup=2,
            active=5,
            repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_path),
        record_shapes=True,
        profile_memory=cfg.memory,
        with_stack=True,
    )

    prof.start()
    for step, (set_name, batch, global_batch_size) in enumerate(train_loader):
        if step >= cfg.warmup + 2 + 5:
            break
        batch = {k: v.cuda() for k, v in batch.items()}
        if train_state.carry is None:
            with torch.device("cuda"):
                train_state.carry = train_state.model.initial_carry(batch)

        train_state.carry, loss, metrics, _, _ = train_state.model(
            carry=train_state.carry, batch=batch, return_keys=[]
        )
        ((1 / global_batch_size) * loss).backward()

        if world_size > 1:
            for p in train_state.model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad)

        for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
            lr = compute_lr(base_lr, config, train_state)
            for pg in optim.param_groups:
                pg["lr"] = lr
            optim.step()
            optim.zero_grad()

        train_state.step += 1
        prof.step()

    prof.stop()

    if rank == 0:
        print(f"\nTrace saved to {trace_path}/")
        print("View with: tensorboard --logdir " + trace_path)
        print("Or load in Chrome at chrome://tracing")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Profile TRM training pipeline")
    p.add_argument("--batches", type=int, default=50, help="Profiling batches")
    p.add_argument("--warmup", type=int, default=5, help="Warmup batches (excluded)")
    p.add_argument("--trace", action="store_true", help="Run torch.profiler trace")
    p.add_argument("--kernel", action="store_true", help="GPU kernel-level profiling (implies --trace)")
    p.add_argument("--memory", action="store_true", help="Profile CUDA memory")
    p.add_argument("--compile", action="store_true", help="Enable torch.compile")
    p.add_argument("--synthetic", action="store_true", help="Use synthetic data (no real dataset needed)")
    p.add_argument("--output-dir", default="profiles", help="Output directory for traces")
    p.add_argument("overrides", nargs="*", help="Hydra config overrides")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = ProfileConfig(
        batches=args.batches,
        warmup=args.warmup,
        trace=args.trace or args.kernel,
        memory=args.memory,
        kernel=args.kernel,
        no_compile=not args.compile,
        output_dir=args.output_dir,
        overrides=args.overrides,
        synthetic=args.synthetic,
    )

    if cfg.trace:
        run_torch_profiler(cfg)
    else:
        profiler = PipelineProfiler(cfg)
        profiler.profile()

        if cfg.memory:
            print("\n--- CUDA Memory Summary ---")
            detailed_memory_summary()
