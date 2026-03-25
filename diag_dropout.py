#!/usr/bin/env python3
"""
Verify: is agent dropout the cause of backward variance?

Compares 3 cases:
  1. Augmentation with dropout (default) - high variance expected
  2. Augmentation WITHOUT dropout - if variance disappears, dropout is the cause
  3. Pure GPU bs=96 fixed shape - baseline (no variance)

Usage:
  torchrun --nproc_per_node=8 diag_dropout.py --cache-dir /tmp/pluto_augment_cache_20k
"""
import argparse
import os
import time
import unittest.mock as mock

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

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


# Patched version: skip the agent dropout block
def generate_positive_sample_no_dropout(data, history_steps=21):
    """Same as original but WITHOUT agent dropout."""
    from copy import deepcopy
    new_data = deepcopy(data)

    noise = np.random.uniform(
        [0.0, -1.5, -0.35, -1, -0.5, -0.2, -0.2],
        [2.0, 1.5, 0.35, 1, 0.5, 0.2, 0.2],
    )

    current_state = data["current_state"]
    agents_position = data["agent"]["position"][1:11, history_steps - 1]
    agents_shape = data["agent"]["shape"][1:11, history_steps - 1]
    agents_heading = data["agent"]["heading"][1:11, history_steps - 1]

    num_tries, scale = 0, 1.0
    while num_tries < 5:
        new_noise = noise * scale
        new_state = current_state + new_noise
        new_state[3] = max(0.0, new_state[3])
        if ba.safety_check(
            ego_position=new_state[:2], ego_heading=new_state[2],
            agents_position=agents_position, agents_heading=agents_heading,
            agents_shape=agents_shape,
        ):
            break
        num_tries += 1
        scale *= 0.5

    new_data["current_state"] = new_state
    new_data["agent"]["position"][0, history_steps - 1] = new_state[:2]
    new_data["agent"]["heading"][0, history_steps - 1] = new_state[2]

    if "cost_maps" in data:
        new_data["cost_maps"] = ba.crop_img_from_center(
            ba.shift_and_rotate_img(
                img=new_data["cost_maps"].astype(np.float32),
                shift=np.array([new_noise[1], -new_noise[0], 0]),
                angle=-new_noise[2], resolution=0.2, cval=-200,
            ), (500, 500),
        )

    # NO agent dropout here!

    new_data = ba.normalize_data(new_data, history_steps)
    return new_data


# Also patch negative sample to skip dropout/insertion (keep agent count fixed)
def generate_negative_sample_no_change(data, history_steps=21):
    """Negative sample WITHOUT agent dropout or insertion (keep agent count fixed)."""
    from copy import deepcopy
    new_data = deepcopy(data)
    # Just return a copy without modifying agent count
    return new_data, {"valid_mask": True, "type": 1}


def run_test(name, model, raw_model, col_fn, device, data_iter, num_steps=50, warmup=10):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    model.train()
    timings = {"fwd": [], "bwd": [], "step": []}

    for step in range(num_steps + warmup):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        if callable(data_iter):
            data = data_iter()
            torch.cuda.synchronize(device)
            t_ready = time.perf_counter()
        else:
            batch_cpu = next(data_iter)
            data = ba.batch_to_device(batch_cpu, device)
            torch.cuda.synchronize(device)
            t_ready = time.perf_counter()

        res = model(data)
        loss = ba.compute_loss(res, data, col_fn, raw_model.num_modes, raw_model.radius, 21)
        torch.cuda.synchronize(device)
        t_fwd = time.perf_counter()

        loss.backward()
        torch.cuda.synchronize(device)
        t_bwd = time.perf_counter()

        optimizer.step()
        optimizer.zero_grad()
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
        ratio = v.max() / v.min() if v.min() > 0 else 0
        log(f"  {key:8s} {np.median(v):7.1f}ms {v.mean():7.1f}ms {v.std():7.1f}ms "
            f"{v.min():7.1f}ms {v.max():7.1f}ms {ratio:7.1f}x")


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
    log("  DROPOUT IMPACT DIAGNOSTIC")
    log("=" * 70)

    def make_model():
        m = ba.PlanningModel(dim=128).to(dev)
        if ddp:
            m = DDP(m, device_ids=[lr], output_device=lr)
        raw = m.module if ddp else m
        col = ba.ESDFCollisionLoss().to(dev)
        return m, raw, col

    def make_loader(augment=True):
        ds = ba.AugmentedCacheDataset(args.cache_dir, augment=augment)
        sampler = DistributedSampler(ds, num_replicas=ws, rank=rank) if ddp else None
        return DataLoader(ds, batch_size=32, num_workers=8, pin_memory=True,
                          collate_fn=ba.collate_fn_augmented, sampler=sampler,
                          drop_last=True, persistent_workers=True, prefetch_factor=2,
                          shuffle=(sampler is None))

    # ================================================================
    # Test 1: Augmentation WITH dropout (default) - expect high variance
    # ================================================================
    log("\n>>> Test 1: Augmentation WITH agent dropout (default)")
    model, raw, col = make_model()
    loader = make_loader(augment=True)
    t = run_test("aug_dropout", model, raw, col, dev, iter(loader),
                 args.num_steps, args.warmup)
    if rank == 0: print_stats("Aug WITH dropout", t)
    del model, raw, col, loader
    torch.cuda.empty_cache()
    if ddp: dist.barrier()

    # ================================================================
    # Test 2: Augmentation WITHOUT dropout - expect low variance
    # ================================================================
    log("\n>>> Test 2: Augmentation WITHOUT agent dropout (patched)")
    # Monkey-patch to disable dropout
    orig_pos = ba.generate_positive_sample
    orig_neg = ba.generate_negative_sample
    ba.generate_positive_sample = generate_positive_sample_no_dropout
    ba.generate_negative_sample = generate_negative_sample_no_change

    model, raw, col = make_model()
    loader = make_loader(augment=True)
    t = run_test("aug_no_dropout", model, raw, col, dev, iter(loader),
                 args.num_steps, args.warmup)
    if rank == 0: print_stats("Aug WITHOUT dropout", t)

    # Restore
    ba.generate_positive_sample = orig_pos
    ba.generate_negative_sample = orig_neg
    del model, raw, col, loader
    torch.cuda.empty_cache()
    if ddp: dist.barrier()

    # ================================================================
    # Test 3: Pure GPU bs=96 baseline
    # ================================================================
    log("\n>>> Test 3: Pure GPU bs=96 (baseline)")
    model, raw, col = make_model()
    from diag_backward_variance import generate_gpu_batch
    gpu_data = generate_gpu_batch(96, dev)
    t = run_test("pure_gpu", model, raw, col, dev,
                 lambda: gpu_data, args.num_steps, args.warmup)
    if rank == 0: print_stats("Pure GPU bs=96", t)
    del model, raw, col, gpu_data
    torch.cuda.empty_cache()
    if ddp: dist.barrier()

    log("\n" + "=" * 70)
    log("  DONE")
    log("=" * 70)
    if ddp: dist.destroy_process_group()


if __name__ == "__main__":
    main()
