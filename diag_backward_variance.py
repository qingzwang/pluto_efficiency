#!/usr/bin/env python3
"""
Diagnose backward variance in augmented training.

Tests:
1. Pure GPU bs=96 (baseline, no DataLoader)
2. DataLoader bs=96 no-aug (same GPU batch, different data path)
3. DataLoader bs=32 aug (the problematic case)
4. DataLoader bs=32 aug + gc.disable()
5. DataLoader bs=32 aug + PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

Usage:
  torchrun --nproc_per_node=8 diag_backward_variance.py --cache-dir /tmp/pluto_augment_cache_20k
"""

import argparse
import gc
import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler

import sys
sys.path.insert(0, "/home/ubuntu/lotus/pluto")

from src.models.pluto.layers.fourier_embedding import FourierEmbedding
from src.models.pluto.layers.transformer import TransformerEncoderLayer
from src.models.pluto.modules.agent_encoder import AgentEncoder
from src.models.pluto.modules.agent_predictor import AgentPredictor
from src.models.pluto.modules.map_encoder import MapEncoder
from src.models.pluto.modules.static_objects_encoder import StaticObjectsEncoder
from src.models.pluto.modules.planning_decoder import PlanningDecoder
from src.models.pluto.loss.esdf_collision_loss import ESDFCollisionLoss

# Import from benchmark_augmentation
from benchmark_augmentation import (
    PlanningModel, AugmentedCacheDataset, collate_fn_augmented,
    batch_to_device, compute_loss,
)


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


def generate_gpu_batch(bs, device, num_agents=20, num_polygons=150, num_points=20,
                       num_static=10, num_ref=6, num_ref_pts=100,
                       history_steps=21, future_steps=80):
    """Generate a batch directly on GPU (same as DDP benchmark)."""
    T = history_steps + future_steps
    data = {
        "current_state": torch.randn(bs, 6, device=device),
        "agent": {
            "position": torch.randn(bs, num_agents, T, 2, device=device),
            "heading": torch.randn(bs, num_agents, T, device=device),
            "velocity": torch.randn(bs, num_agents, T, 2, device=device),
            "shape": torch.rand(bs, num_agents, T, 2, device=device) + 0.5,
            "category": torch.randint(0, 4, (bs, num_agents), device=device),
            "valid_mask": torch.ones(bs, num_agents, T, dtype=torch.bool, device=device),
            "target": torch.randn(bs, num_agents, future_steps, 3, device=device),
        },
        "map": {
            "polygon_center": torch.randn(bs, num_polygons, 3, device=device),
            "polygon_type": torch.randint(0, 3, (bs, num_polygons), device=device),
            "polygon_on_route": torch.randint(0, 2, (bs, num_polygons), dtype=torch.bool, device=device),
            "polygon_tl_status": torch.randint(0, 4, (bs, num_polygons), device=device),
            "polygon_has_speed_limit": torch.randint(0, 2, (bs, num_polygons), dtype=torch.bool, device=device),
            "polygon_speed_limit": torch.rand(bs, num_polygons, device=device) * 30,
            "polygon_position": torch.randn(bs, num_polygons, 2, device=device),
            "polygon_orientation": torch.randn(bs, num_polygons, device=device),
            "point_position": torch.randn(bs, num_polygons, 3, num_points, 2, device=device),
            "point_vector": torch.randn(bs, num_polygons, 3, num_points, 2, device=device),
            "point_orientation": torch.randn(bs, num_polygons, 3, num_points, device=device),
            "point_side": torch.arange(3, device=device).unsqueeze(0).expand(num_polygons, -1).unsqueeze(0).expand(bs, -1, -1),
            "valid_mask": torch.ones(bs, num_polygons, num_points, dtype=torch.bool, device=device),
        },
        "static_objects": {
            "position": torch.randn(bs, num_static, 2, device=device),
            "heading": torch.randn(bs, num_static, device=device),
            "shape": torch.rand(bs, num_static, 2, device=device) + 0.5,
            "category": torch.randint(0, 4, (bs, num_static), device=device),
            "valid_mask": torch.ones(bs, num_static, dtype=torch.bool, device=device),
        },
        "reference_line": {
            "position": torch.randn(bs, num_ref, num_ref_pts, 2, device=device),
            "vector": torch.randn(bs, num_ref, num_ref_pts, 2, device=device),
            "orientation": torch.randn(bs, num_ref, num_ref_pts, device=device),
            "valid_mask": torch.ones(bs, num_ref, num_ref_pts, dtype=torch.bool, device=device),
            "future_projection": torch.randn(bs, num_ref, 8, 2, device=device),
        },
        "cost_maps": torch.randn(bs, 500, 500, 1, device=device),
    }
    return data


def run_test(name, model, raw_model, collision_loss_fn, device, data_source,
             num_steps=50, warmup=10):
    """Run a test and collect per-step timing."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    model.train()

    timings = {"fwd": [], "bwd": [], "opt": [], "h2d": [], "step": []}

    for step in range(num_steps + warmup):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        if callable(data_source):
            data = data_source()
            torch.cuda.synchronize(device)
            t_h2d = time.perf_counter()
            h2d_time = t_h2d - t0
        else:
            batch_cpu = next(data_source)
            t_h2d_start = time.perf_counter()
            data = batch_to_device(batch_cpu, device)
            torch.cuda.synchronize(device)
            t_h2d = time.perf_counter()
            h2d_time = t_h2d - t_h2d_start

        res = model(data)
        loss = compute_loss(res, data, collision_loss_fn,
                            raw_model.num_modes, raw_model.radius, 21)
        torch.cuda.synchronize(device)
        t_fwd = time.perf_counter()

        loss.backward()
        torch.cuda.synchronize(device)
        t_bwd = time.perf_counter()

        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize(device)
        t_opt = time.perf_counter()

        if step >= warmup:
            timings["h2d"].append(h2d_time)
            timings["fwd"].append(t_fwd - t_h2d)
            timings["bwd"].append(t_bwd - t_fwd)
            timings["opt"].append(t_opt - t_bwd)
            timings["step"].append(t_opt - t0)

    return timings


def print_stats(name, timings):
    log(f"\n  [{name}] ({len(timings['step'])} steps)")
    log(f"  {'Phase':12s} {'median':>8s} {'mean':>8s} {'std':>8s} {'min':>8s} {'max':>8s} {'max/min':>8s}")
    log(f"  {'-'*62}")
    for key in ["h2d", "fwd", "bwd", "opt", "step"]:
        v = np.array(timings[key]) * 1000
        ratio = v.max() / v.min() if v.min() > 0 else float('inf')
        log(f"  {key:12s} {np.median(v):7.1f}ms {v.mean():7.1f}ms {v.std():7.1f}ms "
            f"{v.min():7.1f}ms {v.max():7.1f}ms {ratio:7.1f}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="/tmp/pluto_augment_cache_20k")
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    ddp_enabled = setup_distributed()
    rank = get_rank()
    world_size = get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0)) if ddp_enabled else 0
    device = torch.device(f"cuda:{local_rank}")

    log("=" * 70)
    log("  BACKWARD VARIANCE DIAGNOSTIC")
    log("=" * 70)

    # Build model
    def make_model():
        m = PlanningModel(dim=128).to(device)
        if ddp_enabled:
            m = DDP(m, device_ids=[local_rank], output_device=local_rank)
        raw = m.module if ddp_enabled else m
        col = ESDFCollisionLoss().to(device)
        return m, raw, col

    # ================================================================
    # Test 1: Pure GPU bs=96 (baseline)
    # ================================================================
    log("\n>>> Test 1: Pure GPU data, bs=96 (no DataLoader, no H2D)")
    model, raw_model, col_fn = make_model()
    gpu_data = generate_gpu_batch(96, device)
    timings = run_test("pure_gpu_96", model, raw_model, col_fn, device,
                       lambda: gpu_data, args.num_steps, args.warmup)
    if rank == 0:
        print_stats("Pure GPU bs=96", timings)
    del model, raw_model, col_fn, gpu_data
    torch.cuda.empty_cache()
    if ddp_enabled: dist.barrier()

    # ================================================================
    # Test 2: DataLoader bs=32 no-aug (small GPU batch baseline)
    # ================================================================
    log("\n>>> Test 2: DataLoader bs=32, no augmentation")
    model, raw_model, col_fn = make_model()
    ds = AugmentedCacheDataset(args.cache_dir, augment=False)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank) if ddp_enabled else None
    loader = DataLoader(ds, batch_size=32, num_workers=8, pin_memory=True,
                        collate_fn=collate_fn_augmented, sampler=sampler, drop_last=True,
                        persistent_workers=True, prefetch_factor=2,
                        shuffle=(sampler is None))
    data_iter = iter(loader)
    timings = run_test("dl_noaug_32", model, raw_model, col_fn, device,
                       data_iter, args.num_steps, args.warmup)
    if rank == 0:
        print_stats("DataLoader bs=32 no-aug", timings)
    del model, raw_model, col_fn, loader, ds
    torch.cuda.empty_cache()
    if ddp_enabled: dist.barrier()

    # ================================================================
    # Test 3: DataLoader bs=32 aug (the problematic case)
    # ================================================================
    log("\n>>> Test 3: DataLoader bs=32, WITH augmentation (default)")
    model, raw_model, col_fn = make_model()
    ds = AugmentedCacheDataset(args.cache_dir, augment=True)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank) if ddp_enabled else None
    loader = DataLoader(ds, batch_size=32, num_workers=8, pin_memory=True,
                        collate_fn=collate_fn_augmented, sampler=sampler, drop_last=True,
                        persistent_workers=True, prefetch_factor=2,
                        shuffle=(sampler is None))
    data_iter = iter(loader)
    timings = run_test("dl_aug_32", model, raw_model, col_fn, device,
                       data_iter, args.num_steps, args.warmup)
    if rank == 0:
        print_stats("DataLoader bs=32 aug (default)", timings)
    del model, raw_model, col_fn, loader, ds
    torch.cuda.empty_cache()
    if ddp_enabled: dist.barrier()

    # ================================================================
    # Test 4: DataLoader bs=32 aug + gc.disable()
    # ================================================================
    log("\n>>> Test 4: DataLoader bs=32 aug + gc.disable()")
    gc.disable()
    model, raw_model, col_fn = make_model()
    ds = AugmentedCacheDataset(args.cache_dir, augment=True)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank) if ddp_enabled else None
    loader = DataLoader(ds, batch_size=32, num_workers=8, pin_memory=True,
                        collate_fn=collate_fn_augmented, sampler=sampler, drop_last=True,
                        persistent_workers=True, prefetch_factor=2,
                        shuffle=(sampler is None))
    data_iter = iter(loader)
    timings = run_test("dl_aug_32_nogc", model, raw_model, col_fn, device,
                       data_iter, args.num_steps, args.warmup)
    if rank == 0:
        print_stats("DataLoader bs=32 aug + gc.disable()", timings)
    gc.enable()
    del model, raw_model, col_fn, loader, ds
    torch.cuda.empty_cache()
    if ddp_enabled: dist.barrier()

    # ================================================================
    # Test 5: Pure GPU bs=96 with NEW allocation each step
    #         (simulates DataLoader's fresh tensor allocation pattern)
    # ================================================================
    log("\n>>> Test 5: Pure GPU bs=96, fresh allocation each step")
    model, raw_model, col_fn = make_model()

    def gen_fresh():
        return generate_gpu_batch(96, device)

    timings = run_test("gpu_96_fresh", model, raw_model, col_fn, device,
                       gen_fresh, args.num_steps, args.warmup)
    if rank == 0:
        print_stats("Pure GPU bs=96 fresh alloc", timings)
    del model, raw_model, col_fn
    torch.cuda.empty_cache()
    if ddp_enabled: dist.barrier()

    # ================================================================
    # Summary
    # ================================================================
    log("\n" + "=" * 70)
    log("  DIAGNOSTIC COMPLETE")
    log("=" * 70)

    if ddp_enabled:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
