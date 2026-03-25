#!/usr/bin/env python3
"""
Isolate: is it the DataLoader->CPU->GPU path itself that causes variance?

Tests:
  1. Pure GPU bs=96 (baseline, no variance)
  2. DataLoader bs=96 with FIXED shape .pt tensors (no augmentation, no gzip)
  3. DataLoader bs=32 aug (the problematic case, for reference)
  4. DataLoader bs=32 aug, num_workers=0 (no worker processes)

Usage:
  torchrun --nproc_per_node=8 diag_dataloader_path.py
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

import sys
sys.path.insert(0, "/home/ubuntu/lotus/pluto")

import benchmark_augmentation as ba


def setup_distributed():
    if "RANK" not in os.environ:
        return False
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    return True

def get_rank():
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

def get_world_size():
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

def log(msg):
    if get_rank() == 0:
        print(msg, flush=True)


class FixedShapeCPUDataset(Dataset):
    """Returns pre-generated fixed-shape CPU tensors (like pure GPU but via DataLoader)."""
    def __init__(self, num_samples=5000, num_agents=20, num_polygons=150,
                 num_points=20, num_static=10, num_ref=6, num_ref_pts=100,
                 history_steps=21, future_steps=80):
        self.num_samples = num_samples
        self.T = history_steps + future_steps
        self.A = num_agents
        self.M = num_polygons
        self.P = num_points
        self.Ns = num_static
        self.R = num_ref
        self.Pr = num_ref_pts
        self.F = future_steps

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """Generate a fixed-shape sample as CPU tensors."""
        A, T, M, P, Ns, R, Pr, F = self.A, self.T, self.M, self.P, self.Ns, self.R, self.Pr, self.F
        data = {
            "current_state": torch.randn(6),
            "agent": {
                "position": torch.randn(A, T, 2),
                "heading": torch.randn(A, T),
                "velocity": torch.randn(A, T, 2),
                "shape": torch.rand(A, T, 2) + 0.5,
                "category": torch.randint(0, 4, (A,)),
                "valid_mask": torch.ones(A, T, dtype=torch.bool),
                "target": torch.randn(A, F, 3),
            },
            "map": {
                "polygon_center": torch.randn(M, 3),
                "polygon_type": torch.randint(0, 3, (M,)),
                "polygon_on_route": torch.randint(0, 2, (M,), dtype=torch.bool),
                "polygon_tl_status": torch.randint(0, 4, (M,)),
                "polygon_has_speed_limit": torch.randint(0, 2, (M,), dtype=torch.bool),
                "polygon_speed_limit": torch.rand(M) * 30,
                "polygon_position": torch.randn(M, 2),
                "polygon_orientation": torch.randn(M),
                "point_position": torch.randn(M, 3, P, 2),
                "point_vector": torch.randn(M, 3, P, 2),
                "point_orientation": torch.randn(M, 3, P),
                "point_side": torch.arange(3).unsqueeze(0).expand(M, -1),
                "valid_mask": torch.ones(M, P, dtype=torch.bool),
            },
            "static_objects": {
                "position": torch.randn(Ns, 2),
                "heading": torch.randn(Ns),
                "shape": torch.rand(Ns, 2) + 0.5,
                "category": torch.randint(0, 4, (Ns,)),
                "valid_mask": torch.ones(Ns, dtype=torch.bool),
            },
            "reference_line": {
                "position": torch.randn(R, Pr, 2),
                "vector": torch.randn(R, Pr, 2),
                "orientation": torch.randn(R, Pr),
                "valid_mask": torch.ones(R, Pr, dtype=torch.bool),
                "future_projection": torch.randn(R, 8, 2),
            },
            "cost_maps": torch.randn(500, 500, 1),
        }
        return data


def collate_fixed(batch):
    """Simple stack collate for fixed-shape data (no pad_sequence needed)."""
    out = {}
    for key in batch[0]:
        if isinstance(batch[0][key], dict):
            out[key] = {}
            for k in batch[0][key]:
                if isinstance(batch[0][key][k], torch.Tensor):
                    out[key][k] = torch.stack([b[key][k] for b in batch])
        elif isinstance(batch[0][key], torch.Tensor):
            out[key] = torch.stack([b[key] for b in batch])
    return out


def generate_gpu_batch(bs, device):
    from diag_backward_variance import generate_gpu_batch as gen
    return gen(bs, device)


def run_test(name, model, raw, col, device, data_src, n=50, warmup=10, loader_ref=None):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    model.train()
    timings = {"fwd": [], "bwd": [], "step": []}
    data_iter = data_src

    for step in range(n + warmup):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        if callable(data_src):
            data = data_src()
        else:
            try:
                batch_cpu = next(data_iter)
            except StopIteration:
                data_iter = iter(loader_ref) if loader_ref else data_src
                batch_cpu = next(data_iter)
            data = ba.batch_to_device(batch_cpu, device)
        torch.cuda.synchronize(device)
        t_ready = time.perf_counter()

        res = model(data)
        loss = ba.compute_loss(res, data, col, raw.num_modes, raw.radius, 21)
        torch.cuda.synchronize(device)
        t_fwd = time.perf_counter()

        loss.backward()
        torch.cuda.synchronize(device)
        t_bwd = time.perf_counter()

        opt.step(); opt.zero_grad()
        torch.cuda.synchronize(device)
        t_end = time.perf_counter()

        if step >= warmup:
            timings["fwd"].append(t_fwd - t_ready)
            timings["bwd"].append(t_bwd - t_fwd)
            timings["step"].append(t_end - t0)
    return timings


def print_stats(name, timings):
    log(f"\n  [{name}] ({len(timings['step'])} steps)")
    log(f"  {'Phase':8s} {'median':>8s} {'mean':>8s} {'std':>8s} {'min':>8s} {'max':>8s} {'max/min':>8s}")
    log(f"  {'-'*56}")
    for key in ["fwd", "bwd", "step"]:
        v = np.array(timings[key]) * 1000
        r = v.max() / v.min() if v.min() > 0 else 0
        log(f"  {key:8s} {np.median(v):7.1f}ms {v.mean():7.1f}ms {v.std():7.1f}ms "
            f"{v.min():7.1f}ms {v.max():7.1f}ms {r:7.1f}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="/tmp/pluto_augment_cache_20k")
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    ddp = setup_distributed()
    rank, ws = get_rank(), get_world_size()
    lr = int(os.environ.get("LOCAL_RANK", 0)) if ddp else 0
    dev = torch.device(f"cuda:{lr}")

    log("=" * 70)
    log("  DATALOADER PATH DIAGNOSTIC")
    log("=" * 70)

    def make_model():
        m = ba.PlanningModel(dim=128).to(dev)
        if ddp: m = DDP(m, device_ids=[lr], output_device=lr)
        raw = m.module if ddp else m
        col = ba.ESDFCollisionLoss().to(dev)
        return m, raw, col

    # Test 1: Pure GPU bs=96
    log("\n>>> Test 1: Pure GPU bs=96 (baseline)")
    model, raw, col = make_model()
    gpu_data = generate_gpu_batch(96, dev)
    t = run_test("pure_gpu", model, raw, col, dev, lambda: gpu_data,
                 args.num_steps, args.warmup)
    if rank == 0: print_stats("Pure GPU bs=96 (reuse)", t)
    del model, raw, col, gpu_data; torch.cuda.empty_cache()
    if ddp: dist.barrier()

    # Test 2: DataLoader bs=96 fixed shape (CPU->GPU, 8 workers, stack collate)
    log("\n>>> Test 2: DataLoader bs=96, fixed shape CPU tensors, 8 workers")
    model, raw, col = make_model()
    ds = FixedShapeCPUDataset(num_samples=20000)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank) if ddp else None
    loader = DataLoader(ds, batch_size=96, num_workers=8, pin_memory=True,
                        collate_fn=collate_fixed, sampler=sampler, drop_last=True,
                        persistent_workers=True, prefetch_factor=2,
                        shuffle=(sampler is None))
    t = run_test("dl_fixed_96", model, raw, col, dev, iter(loader),
                 args.num_steps, args.warmup, loader_ref=loader)
    if rank == 0: print_stats("DataLoader bs=96 fixed (8 workers)", t)
    del model, raw, col, loader, ds; torch.cuda.empty_cache()
    if ddp: dist.barrier()

    # Test 3: DataLoader bs=96 fixed shape, 0 workers (main process)
    log("\n>>> Test 3: DataLoader bs=96, fixed shape, 0 workers (main process)")
    model, raw, col = make_model()
    ds = FixedShapeCPUDataset(num_samples=20000)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank) if ddp else None
    loader = DataLoader(ds, batch_size=96, num_workers=0, pin_memory=True,
                        collate_fn=collate_fixed, sampler=sampler, drop_last=True,
                        shuffle=(sampler is None))
    t = run_test("dl_fixed_0w", model, raw, col, dev, iter(loader),
                 args.num_steps, args.warmup, loader_ref=loader)
    if rank == 0: print_stats("DataLoader bs=96 fixed (0 workers)", t)
    del model, raw, col, loader, ds; torch.cuda.empty_cache()
    if ddp: dist.barrier()

    # Test 4: DataLoader bs=32 aug with pad_sequence (the problematic case)
    log("\n>>> Test 4: DataLoader bs=32 aug (gzip + pad_sequence, 8 workers)")
    model, raw, col = make_model()
    ds = ba.AugmentedCacheDataset(args.cache_dir, augment=True)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank) if ddp else None
    loader = DataLoader(ds, batch_size=32, num_workers=8, pin_memory=True,
                        collate_fn=ba.collate_fn_augmented, sampler=sampler,
                        drop_last=True, persistent_workers=True, prefetch_factor=2,
                        shuffle=(sampler is None))
    t = run_test("dl_aug_32", model, raw, col, dev, iter(loader),
                 args.num_steps, args.warmup, loader_ref=loader)
    if rank == 0: print_stats("DataLoader bs=32 aug (8 workers)", t)
    del model, raw, col, loader, ds; torch.cuda.empty_cache()
    if ddp: dist.barrier()

    # Test 5: DataLoader bs=32 aug, 0 workers
    log("\n>>> Test 5: DataLoader bs=32 aug, 0 workers (main process)")
    model, raw, col = make_model()
    ds = ba.AugmentedCacheDataset(args.cache_dir, augment=True)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank) if ddp else None
    loader = DataLoader(ds, batch_size=32, num_workers=0, pin_memory=False,
                        collate_fn=ba.collate_fn_augmented, sampler=sampler,
                        drop_last=True, shuffle=(sampler is None))
    t = run_test("dl_aug_0w", model, raw, col, dev, iter(loader),
                 args.num_steps, args.warmup, loader_ref=loader)
    if rank == 0: print_stats("DataLoader bs=32 aug (0 workers)", t)
    del model, raw, col, loader, ds; torch.cuda.empty_cache()
    if ddp: dist.barrier()

    log("\n" + "=" * 70)
    log("  DONE")
    log("=" * 70)
    if ddp: dist.destroy_process_group()


if __name__ == "__main__":
    main()
