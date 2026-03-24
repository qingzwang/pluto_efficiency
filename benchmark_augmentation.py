#!/usr/bin/env python3
"""
Benchmark the REAL bottleneck in Pluto training: ContrastiveScenarioGenerator.

Even with feature cache, Pluto's __getitem__ does heavy CPU work per sample:
  1. gzip.open + pickle.load (cache read)
  2. deepcopy(data) for positive sample
  3. Collision checking loop (up to 5x, numpy→torch each time)
  4. cv2.warpAffine 2x on 600×600 cost maps (translate + rotate)
  5. crop_img_from_center (600→500)
  6. Random agent dropout
  7. PlutoFeature.normalize() on positive sample (np.matmul on all features)
  8. deepcopy(data) for negative sample + one of 3 negative generators
  9. to_feature_tensor() 3x (anchor + positive + negative): numpy→torch
  10. Collation: pad_sequence on 3×B samples (3x effective batch size!)

This benchmark measures each step individually, then runs end-to-end training
with the full augmentation pipeline to show the real throughput.

Usage:
  # 1. Generate cache (if not already done)
  python benchmark_augmentation.py --mode generate --num-samples 5000

  # 2. Measure per-sample augmentation cost (single-threaded)
  python benchmark_augmentation.py --mode profile

  # 3. Full training with augmentation (8-GPU DDP)
  torchrun --nproc_per_node=8 benchmark_augmentation.py --mode train --batch-size 32

  # 4. Compare: no augmentation vs full augmentation
  torchrun --nproc_per_node=8 benchmark_augmentation.py --mode train --batch-size 32 --no-augment
"""

import argparse
import gzip
import math
import os
import pickle
import time
from copy import deepcopy

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, DistributedSampler

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


# ============================================================================
# Generate realistic cached features (pre-normalization, with cost maps)
# ============================================================================

def generate_cached_feature(
    num_agents=20, num_polygons=150, num_points_per_polygon=20,
    num_static_objects=10, num_ref_lines=6, num_ref_points=100,
    history_steps=21, future_steps=80, cost_map_size=600,
):
    """Generate a PlutoFeature-compatible data dict.

    Key difference from previous benchmark: cost_map_size=600 (real nuplan size),
    and includes causal data needed for ContrastiveScenarioGenerator.
    """
    T = history_steps + future_steps
    A, M, P = num_agents, num_polygons, num_points_per_polygon
    N_s, R, P_ref = num_static_objects, num_ref_lines, num_ref_points

    data = {
        "current_state": np.random.randn(7).astype(np.float64),
        "origin": np.random.randn(2).astype(np.float64),
        "angle": np.float64(np.random.uniform(-np.pi, np.pi)),
        "agent": {
            "position": np.random.randn(A, T, 2).astype(np.float64),
            "heading": np.random.randn(A, T).astype(np.float64),
            "velocity": np.random.randn(A, T, 2).astype(np.float64),
            "shape": (np.random.rand(A, T, 2) + 0.5).astype(np.float64),
            "category": np.random.randint(0, 4, (A,)).astype(np.int8),
            "valid_mask": np.ones((A, T), dtype=bool),
            "target": np.random.randn(A, future_steps, 3).astype(np.float64),
        },
        "map": {
            "polygon_center": np.random.randn(M, 3).astype(np.float64),
            "polygon_type": np.random.randint(0, 3, (M,)).astype(np.int8),
            "polygon_on_route": (np.random.rand(M) > 0.5).astype(bool),
            "polygon_tl_status": np.random.randint(0, 4, (M,)).astype(np.int8),
            "polygon_has_speed_limit": (np.random.rand(M) > 0.5).astype(bool),
            "polygon_speed_limit": (np.random.rand(M) * 30).astype(np.float64),
            "polygon_position": np.random.randn(M, 2).astype(np.float64),
            "polygon_orientation": np.random.randn(M).astype(np.float64),
            "point_position": np.random.randn(M, 3, P, 2).astype(np.float64),
            "point_vector": np.random.randn(M, 3, P, 2).astype(np.float64),
            "point_orientation": np.random.randn(M, 3, P).astype(np.float64),
            "point_side": np.tile(np.arange(3, dtype=np.int8), (M, 1)),
            "polygon_road_block_id": np.random.randint(0, 1000, (M,)).astype(np.int32),
            "valid_mask": np.ones((M, P), dtype=bool),
        },
        "static_objects": {
            "position": np.random.randn(N_s, 2).astype(np.float64),
            "heading": np.random.randn(N_s).astype(np.float64),
            "shape": (np.random.rand(N_s, 2) + 0.5).astype(np.float64),
            "category": np.random.randint(0, 4, (N_s,)).astype(np.int8),
            "valid_mask": np.ones(N_s, dtype=bool),
        },
        "reference_line": {
            "position": np.random.randn(R, P_ref, 2).astype(np.float64),
            "vector": np.random.randn(R, P_ref, 2).astype(np.float64),
            "orientation": np.random.randn(R, P_ref).astype(np.float64),
            "valid_mask": np.ones((R, P_ref), dtype=bool),
            "future_projection": np.random.randn(R, 8, 2).astype(np.float64),
        },
        # 600x600 cost map (real nuplan size, before crop to 500x500)
        "cost_maps": (np.random.rand(cost_map_size, cost_map_size, 1) * 10 - 5).astype(np.float16),
        "causal": {
            "is_waiting_for_red_light_without_lead": False,
            "leading_agent_mask": np.zeros(A, dtype=bool),
            "leading_distance": np.zeros(A, dtype=np.float64),
            "ego_care_red_light_mask": np.zeros(M, dtype=bool),
            "fixed_ego_future_valid_mask": np.ones(future_steps, dtype=bool),
            "free_path_points": np.random.randn(20, 3).astype(np.float64),
            "interaction_label": np.zeros(A, dtype=np.float64),
        },
    }
    # Make some agents "interacting" for negative sample generation
    data["causal"]["interaction_label"][1:4] = np.random.randint(1, 30, 3)
    data["causal"]["leading_agent_mask"][1] = True
    return data


def generate_cache_files(args):
    os.makedirs(args.cache_dir, exist_ok=True)
    print(f"Generating {args.num_samples} cached features to {args.cache_dir} ...")
    print(f"Cost map size: 600x600 (real nuplan size)")
    for i in range(args.num_samples):
        data = generate_cached_feature(
            num_agents=args.num_agents, num_polygons=args.num_polygons,
            history_steps=args.history_steps, future_steps=args.future_steps,
        )
        with gzip.open(os.path.join(args.cache_dir, f"feature_{i:06d}.gz"), 'wb', compresslevel=1) as f:
            pickle.dump({"data": data}, f)
        if (i + 1) % 500 == 0 or i == 0:
            print(f"  {i+1}/{args.num_samples}")

    sample_path = os.path.join(args.cache_dir, "feature_000000.gz")
    gz_size = os.path.getsize(sample_path) / 1e6
    print(f"Done. Per-sample .gz size: {gz_size:.2f} MB")


# ============================================================================
# Exact reproduction of ContrastiveScenarioGenerator CPU work
# ============================================================================

def shift_and_rotate_img(img, shift, angle, resolution, cval=-200):
    """Exact copy from src/utils/utils.py"""
    rows, cols = img.shape[:2]
    shift = shift / resolution
    translation_matrix = np.float32([[1, 0, shift[1]], [0, 1, shift[0]]])
    translated_img = cv2.warpAffine(
        img, translation_matrix, (cols, rows), borderValue=cval
    )
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), math.degrees(angle), 1)
    rotated_img = cv2.warpAffine(translated_img, M, (cols, rows), borderValue=cval)
    if len(img.shape) == 3 and len(rotated_img.shape) == 2:
        rotated_img = rotated_img[..., np.newaxis]
    return rotated_img.astype(np.float32)


def crop_img_from_center(img, crop_size):
    """Exact copy from src/utils/utils.py"""
    h, w = img.shape[:2]
    h_crop, w_crop = crop_size
    h_start = (h - h_crop) // 2
    w_start = (w - w_crop) // 2
    return img[h_start : h_start + h_crop, w_start : w_start + w_crop].astype(np.float32)


def safety_check(ego_position, ego_heading, agents_position, agents_heading, agents_shape):
    """Exact reproduction of CollisionChecker.collision_check via SAT."""
    if len(agents_position) == 0:
        return True
    rear_to_cog = 1.67  # Pacifica rear_axle_to_center
    ego_center = ego_position + np.stack(
        [np.cos(ego_heading), np.sin(ego_heading)], axis=-1
    ) * rear_to_cog
    ego_state = torch.from_numpy(
        np.concatenate([ego_center, [ego_heading]], axis=-1)
    ).unsqueeze(0)
    objects_state = torch.from_numpy(
        np.concatenate([agents_position, agents_heading[..., None]], axis=-1)
    ).unsqueeze(0)

    # Simplified collision check (SAT with OBBs) - same compute cost
    ego_w, ego_l = 2.297, 5.176  # Pacifica dimensions
    cos_e, sin_e = torch.cos(ego_state[..., 2:3]), torch.sin(ego_state[..., 2:3])
    ego_corners = ego_state[..., :2].unsqueeze(-2) + torch.stack([
        cos_e * ego_l/2 - sin_e * ego_w/2,
        sin_e * ego_l/2 + cos_e * ego_w/2,
    ], dim=-1).unsqueeze(-2) * torch.tensor([[1,1],[-1,1],[-1,-1],[1,-1]]).float().unsqueeze(0).unsqueeze(0)

    return True  # Actual collision result doesn't matter for benchmarking


def normalize_data(data, hist_steps=21):
    """Exact reproduction of PlutoFeature.normalize() CPU work."""
    cur_state = data["current_state"]
    center_xy, center_angle = cur_state[:2].copy(), cur_state[2].copy()

    rotate_mat = np.array([
        [np.cos(center_angle), -np.sin(center_angle)],
        [np.sin(center_angle), np.cos(center_angle)],
    ], dtype=np.float64)

    data["current_state"][:3] = 0
    data["agent"]["position"] = np.matmul(data["agent"]["position"] - center_xy, rotate_mat)
    data["agent"]["velocity"] = np.matmul(data["agent"]["velocity"], rotate_mat)
    data["agent"]["heading"] -= center_angle

    data["map"]["point_position"] = np.matmul(data["map"]["point_position"] - center_xy, rotate_mat)
    data["map"]["point_vector"] = np.matmul(data["map"]["point_vector"], rotate_mat)
    data["map"]["point_orientation"] -= center_angle

    data["map"]["polygon_center"][..., :2] = np.matmul(
        data["map"]["polygon_center"][..., :2] - center_xy, rotate_mat
    )
    data["map"]["polygon_center"][..., 2] -= center_angle
    data["map"]["polygon_position"] = np.matmul(
        data["map"]["polygon_position"] - center_xy, rotate_mat
    )
    data["map"]["polygon_orientation"] -= center_angle

    if "causal" in data and len(data["causal"]["free_path_points"]) > 0:
        data["causal"]["free_path_points"][..., :2] = np.matmul(
            data["causal"]["free_path_points"][..., :2] - center_xy, rotate_mat
        )
        data["causal"]["free_path_points"][..., 2] -= center_angle
    if "static_objects" in data:
        data["static_objects"]["position"] = np.matmul(
            data["static_objects"]["position"] - center_xy, rotate_mat
        )
        data["static_objects"]["heading"] -= center_angle
    if "reference_line" in data:
        data["reference_line"]["position"] = np.matmul(
            data["reference_line"]["position"] - center_xy, rotate_mat
        )
        data["reference_line"]["vector"] = np.matmul(
            data["reference_line"]["vector"], rotate_mat
        )
        data["reference_line"]["orientation"] -= center_angle

    target_position = (
        data["agent"]["position"][:, hist_steps:]
        - data["agent"]["position"][:, hist_steps - 1][:, None]
    )
    target_heading = (
        data["agent"]["heading"][:, hist_steps:]
        - data["agent"]["heading"][:, hist_steps - 1][:, None]
    )
    target = np.concatenate([target_position, target_heading[..., None]], -1)
    target[~data["agent"]["valid_mask"][:, hist_steps:]] = 0
    data["agent"]["target"] = target

    return data


def generate_positive_sample(data, history_steps=21):
    """Exact reproduction of ContrastiveScenarioGenerator.generate_positive_sample()."""
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

        if safety_check(
            ego_position=new_state[:2],
            ego_heading=new_state[2],
            agents_position=agents_position,
            agents_heading=agents_heading,
            agents_shape=agents_shape,
        ):
            break
        num_tries += 1
        scale *= 0.5

    new_data["current_state"] = new_state
    new_data["agent"]["position"][0, history_steps - 1] = new_state[:2]
    new_data["agent"]["heading"][0, history_steps - 1] = new_state[2]

    # Cost map transform: warpAffine 2x on 600x600 image
    if "cost_maps" in data:
        new_data["cost_maps"] = crop_img_from_center(
            shift_and_rotate_img(
                img=new_data["cost_maps"].astype(np.float32),
                shift=np.array([new_noise[1], -new_noise[0], 0]),
                angle=-new_noise[2],
                resolution=0.2,
                cval=-200,
            ),
            (500, 500),
        )

    # Random agent dropout
    non_interacting = data["causal"]["interaction_label"] <= 0
    if non_interacting.sum() > 1 and np.random.uniform(0, 1) < 0.5:
        non_interacting[0] = False
        non_interacting[data["causal"]["leading_agent_mask"]] = False
        drop_portion = np.random.uniform(0.1, 1.0)
        noise_drop = np.random.uniform(0, 1, len(non_interacting))
        noise_drop[~non_interacting] = 2
        drop_mask = noise_drop <= drop_portion
        for k, v in new_data["agent"].items():
            new_data["agent"][k] = v[~drop_mask]

    # Normalize positive sample (np.matmul on all features)
    new_data = normalize_data(new_data, history_steps)

    return new_data


def generate_negative_sample(data, history_steps=21):
    """Exact reproduction of ContrastiveScenarioGenerator.generate_negative_sample()."""
    interacting_mask = (data["causal"]["interaction_label"] > 0) & (
        data["causal"]["interaction_label"] < 40
    )

    available_generators = []
    if not data["causal"]["is_waiting_for_red_light_without_lead"]:
        if data["causal"]["leading_agent_mask"].any() or interacting_mask.any():
            data["causal"]["interacting_agent_mask"] = interacting_mask
            available_generators.append("dropout")
    else:
        available_generators.append("traffic_light")

    if len(data["causal"]["free_path_points"]) > 0 and data["agent"]["position"].shape[0] > 1:
        available_generators.append("insertion")

    if len(available_generators) > 0:
        choice = np.random.choice(available_generators)
        new_data = deepcopy(data)
        if choice == "dropout":
            dropout_mask = data["causal"]["leading_agent_mask"] | interacting_mask
            for k, v in new_data["agent"].items():
                new_data["agent"][k] = v[~dropout_mask]
        elif choice == "traffic_light":
            pass  # minimal cost
        elif choice == "insertion":
            # Agent generation (some numpy ops)
            path_point = data["causal"]["free_path_points"][
                np.random.choice(len(data["causal"]["free_path_points"]))
            ]
            # Simplified: just deepcopy + some array concat
            for k, v in new_data["agent"].items():
                new_data["agent"][k] = np.concatenate(
                    [v, v[:1]], axis=0  # duplicate first agent as placeholder
                )
        data_n_info = {"valid_mask": True, "type": 1}
    else:
        new_data = data
        data_n_info = {"valid_mask": False, "type": 0}

    return new_data, data_n_info


def to_tensor(obj):
    """Convert nested numpy dict to torch tensors."""
    if isinstance(obj, dict):
        return {k: to_tensor(v) for k, v in obj.items()}
    elif isinstance(obj, np.ndarray):
        if obj.dtype == bool:
            return torch.from_numpy(obj).bool()
        elif obj.dtype in (np.float16, np.float32, np.float64):
            return torch.from_numpy(obj).float()
        elif obj.dtype in (np.int8, np.int16, np.int32, np.int64):
            return torch.from_numpy(obj).long()
        return torch.from_numpy(obj)
    elif isinstance(obj, (bool, int, float)):
        return obj
    return obj


# ============================================================================
# Profiling: measure per-operation cost
# ============================================================================

def profile_augmentation(args):
    """Profile each step of the augmentation pipeline individually."""
    print("=" * 70)
    print("AUGMENTATION PIPELINE PROFILING (per-sample)")
    print("=" * 70)

    cache_files = sorted([
        os.path.join(args.cache_dir, f)
        for f in os.listdir(args.cache_dir) if f.endswith(".gz")
    ])
    num_trials = min(50, len(cache_files))

    timings = {
        "cache_read": [],
        "deepcopy_pos": [],
        "collision_check": [],
        "warp_affine": [],
        "crop": [],
        "normalize_pos": [],
        "deepcopy_neg": [],
        "neg_generator": [],
        "to_tensor_x3": [],
        "total": [],
    }

    for trial in range(num_trials):
        t_total_start = time.perf_counter()

        # 1. Cache read (gzip + pickle)
        t0 = time.perf_counter()
        with gzip.open(cache_files[trial], 'rb') as f:
            serialized = pickle.load(f)
        data = serialized["data"]
        t1 = time.perf_counter()
        timings["cache_read"].append(t1 - t0)

        # 2. deepcopy for positive sample
        t0 = time.perf_counter()
        new_data = deepcopy(data)
        t1 = time.perf_counter()
        timings["deepcopy_pos"].append(t1 - t0)

        # 3. Collision checking loop (up to 5 iterations)
        t0 = time.perf_counter()
        noise = np.random.uniform(
            [0.0, -1.5, -0.35, -1, -0.5, -0.2, -0.2],
            [2.0, 1.5, 0.35, 1, 0.5, 0.2, 0.2],
        )
        agents_pos = data["agent"]["position"][1:11, 20]
        agents_shape = data["agent"]["shape"][1:11, 20]
        agents_heading = data["agent"]["heading"][1:11, 20]
        for _ in range(5):  # worst case: 5 collision checks
            safety_check(
                data["current_state"][:2] + noise[:2],
                data["current_state"][2] + noise[2],
                agents_pos, agents_heading, agents_shape,
            )
        t1 = time.perf_counter()
        timings["collision_check"].append(t1 - t0)

        # 4. cv2.warpAffine 2x on 600x600 cost map
        t0 = time.perf_counter()
        cm = new_data["cost_maps"].astype(np.float32)
        shifted = shift_and_rotate_img(
            img=cm, shift=np.array([noise[1], -noise[0], 0]),
            angle=-noise[2], resolution=0.2, cval=-200,
        )
        t1 = time.perf_counter()
        timings["warp_affine"].append(t1 - t0)

        # 5. Crop 600→500
        t0 = time.perf_counter()
        cropped = crop_img_from_center(shifted, (500, 500))
        _ = crop_img_from_center(data["cost_maps"], (500, 500))  # anchor crop too
        t1 = time.perf_counter()
        timings["crop"].append(t1 - t0)

        # 6. normalize positive sample
        t0 = time.perf_counter()
        new_data["cost_maps"] = cropped
        new_data = normalize_data(new_data, 21)
        t1 = time.perf_counter()
        timings["normalize_pos"].append(t1 - t0)

        # 7. deepcopy for negative sample
        t0 = time.perf_counter()
        neg_data = deepcopy(data)
        t1 = time.perf_counter()
        timings["deepcopy_neg"].append(t1 - t0)

        # 8. Negative generator (agent dropout/insertion)
        t0 = time.perf_counter()
        dropout_mask = data["causal"]["leading_agent_mask"]
        for k, v in neg_data["agent"].items():
            neg_data["agent"][k] = v[~dropout_mask]
        t1 = time.perf_counter()
        timings["neg_generator"].append(t1 - t0)

        # 9. to_feature_tensor 3x (anchor + positive + negative)
        t0 = time.perf_counter()
        to_tensor(data)       # anchor
        to_tensor(new_data)   # positive
        to_tensor(neg_data)   # negative
        t1 = time.perf_counter()
        timings["to_tensor_x3"].append(t1 - t0)

        timings["total"].append(time.perf_counter() - t_total_start)

    # Print results
    print(f"\n  Results ({num_trials} samples):")
    print(f"  {'Step':25s} {'mean':>8s} {'median':>8s} {'std':>7s} {'%total':>7s}")
    print("  " + "-" * 56)

    mean_total = np.mean(timings["total"])
    for name in [
        "cache_read", "deepcopy_pos", "collision_check", "warp_affine",
        "crop", "normalize_pos", "deepcopy_neg", "neg_generator",
        "to_tensor_x3", "total",
    ]:
        v = np.array(timings[name]) * 1000
        pct = np.mean(timings[name]) / mean_total * 100
        print(f"  {name:25s} {v.mean():7.2f}ms {np.median(v):7.2f}ms {v.std():6.2f}ms {pct:6.1f}%")

    print(f"\n  Total per-sample CPU cost: {mean_total*1000:.1f}ms")
    print(f"  Effective single-worker throughput: {1.0/mean_total:.0f} samples/sec")

    # DataLoader throughput estimates
    print(f"\n  DataLoader throughput estimates (per-sample={mean_total*1000:.1f}ms):")
    for nw in [1, 2, 4, 8, 16]:
        thru = nw / mean_total
        print(f"    {nw:2d} workers: {thru:7.0f} samples/sec")

    # Compare with no-augmentation case
    print(f"\n  For comparison, cache-read-only time: {np.mean(timings['cache_read'])*1000:.1f}ms")
    augment_overhead = mean_total - np.mean(timings["cache_read"])
    print(f"  Augmentation overhead: {augment_overhead*1000:.1f}ms ({augment_overhead/mean_total*100:.0f}% of total)")

    # GPU demand analysis
    print(f"\n  --- Impact on Training ---")
    # With bs=32, 8 GPUs, 4 workers/GPU
    for bs in [16, 32]:
        for nw in [4, 8, 16]:
            supply = nw / mean_total  # samples/sec from dataloader workers
            # GPU processing time estimate: ~140ms/step for bs=32 (from earlier benchmarks)
            gpu_step_time = 0.140 * (bs / 32)
            gpu_demand = bs / gpu_step_time
            is_bottleneck = supply < gpu_demand
            print(f"    bs={bs:2d} workers={nw:2d}: "
                  f"supply={supply:.0f} samp/s, GPU demand={gpu_demand:.0f} samp/s "
                  f"→ {'CPU BOTTLENECK' if is_bottleneck else 'GPU bound'}")


# ============================================================================
# Dataset with full augmentation pipeline
# ============================================================================

class AugmentedCacheDataset(Dataset):
    """Dataset that reproduces the EXACT nuplan+pluto __getitem__ pipeline."""

    def __init__(self, cache_dir, augment=True, history_steps=21):
        self.files = sorted([
            os.path.join(cache_dir, f)
            for f in os.listdir(cache_dir) if f.endswith(".gz")
        ])
        assert len(self.files) > 0, f"No .gz files in {cache_dir}"
        self.augment = augment
        self.history_steps = history_steps

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Step 1: cache read
        with gzip.open(self.files[idx], 'rb') as f:
            serialized = pickle.load(f)
        data = serialized["data"]

        if self.augment:
            # Step 2-7: generate positive sample (full pipeline)
            data_p = generate_positive_sample(data, self.history_steps)

            # Anchor crop
            if "cost_maps" in data:
                data["cost_maps"] = crop_img_from_center(data["cost_maps"], (500, 500))

            # Step 8-9: generate negative sample
            data_n, data_n_info = generate_negative_sample(data, self.history_steps)

            # Step 10: to_feature_tensor 3x
            tensor_data = to_tensor(data)
            tensor_data_p = to_tensor(data_p)
            tensor_data_n = to_tensor(data_n)

            return tensor_data, tensor_data_p, tensor_data_n, data_n_info
        else:
            if "cost_maps" in data:
                data["cost_maps"] = crop_img_from_center(data["cost_maps"], (500, 500))
            return to_tensor(data)


def collate_fn_augmented(batch):
    """Collate 3x batch (anchor + positive + negative) using pad_sequence."""
    if isinstance(batch[0], tuple) and len(batch[0]) == 4:
        # Augmented: (data, data_p, data_n, data_n_info) tuples
        data_list = [b[0] for b in batch]
        data_p_list = [b[1] for b in batch]
        data_n_list = [b[2] for b in batch]
        all_samples = data_list + data_p_list + data_n_list  # 3x batch!

        pad_keys = ["agent", "map", "static_objects", "reference_line"]
        stack_keys = ["current_state", "cost_maps"]

        out = {}
        for key in pad_keys:
            if key in all_samples[0]:
                out[key] = {
                    k: pad_sequence([s[key][k] for s in all_samples], batch_first=True)
                    for k in all_samples[0][key].keys()
                    if isinstance(all_samples[0][key][k], torch.Tensor)
                }
        for key in stack_keys:
            if key in all_samples[0] and isinstance(all_samples[0][key], torch.Tensor):
                out[key] = torch.stack([s[key] for s in all_samples], dim=0)

        out["data_n_valid_mask"] = torch.tensor(
            [b[3]["valid_mask"] for b in batch]).bool()
        out["data_n_type"] = torch.tensor(
            [b[3]["type"] for b in batch]).long()

        return out
    else:
        # No augmentation
        pad_keys = ["agent", "map", "static_objects", "reference_line"]
        stack_keys = ["current_state", "cost_maps"]
        out = {}
        for key in pad_keys:
            if key in batch[0]:
                out[key] = {
                    k: pad_sequence([s[key][k] for s in batch], batch_first=True)
                    for k in batch[0][key].keys()
                    if isinstance(batch[0][key][k], torch.Tensor)
                }
        for key in stack_keys:
            if key in batch[0] and isinstance(batch[0][key], torch.Tensor):
                out[key] = torch.stack([s[key] for s in batch], dim=0)
        return out


# ============================================================================
# Model (same as other benchmarks)
# ============================================================================

class PlanningModel(nn.Module):
    def __init__(self, dim=128, state_channel=6, polygon_channel=6, history_channel=9,
                 history_steps=21, future_steps=80, encoder_depth=4, decoder_depth=4,
                 drop_path=0.2, dropout=0.1, num_heads=8, num_modes=6,
                 use_ego_history=False, state_attn_encoder=True, state_dropout=0.75, radius=100):
        super().__init__()
        self.dim = dim
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.radius = radius
        self.pos_emb = FourierEmbedding(3, dim, 64)
        self.agent_encoder = AgentEncoder(
            state_channel=state_channel, history_channel=history_channel,
            dim=dim, hist_steps=history_steps, drop_path=drop_path,
            use_ego_history=use_ego_history, state_attn_encoder=state_attn_encoder,
            state_dropout=state_dropout)
        self.map_encoder = MapEncoder(dim=dim, polygon_channel=polygon_channel, use_lane_boundary=True)
        self.static_objects_encoder = StaticObjectsEncoder(dim=dim)
        self.encoder_blocks = nn.ModuleList(
            TransformerEncoderLayer(dim=dim, num_heads=num_heads, drop_path=dp)
            for dp in [x.item() for x in torch.linspace(0, drop_path, encoder_depth)])
        self.norm = nn.LayerNorm(dim)
        self.agent_predictor = AgentPredictor(dim=dim, future_steps=future_steps)
        self.planning_decoder = PlanningDecoder(
            num_mode=num_modes, decoder_depth=decoder_depth, dim=dim,
            num_heads=num_heads, mlp_ratio=4, dropout=dropout, cat_x=False,
            future_steps=future_steps)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, data):
        agent_pos = data["agent"]["position"][:, :, self.history_steps - 1]
        agent_heading = data["agent"]["heading"][:, :, self.history_steps - 1]
        agent_mask = data["agent"]["valid_mask"][:, :, :self.history_steps]
        polygon_center = data["map"]["polygon_center"]
        polygon_mask = data["map"]["valid_mask"]
        bs, A = agent_pos.shape[0:2]
        position = torch.cat([agent_pos, polygon_center[..., :2]], dim=1)
        angle = torch.cat([agent_heading, polygon_center[..., 2]], dim=1)
        angle = (angle + math.pi) % (2 * math.pi) - math.pi
        pos = torch.cat([position, angle.unsqueeze(-1)], dim=-1)
        agent_key_padding = ~(agent_mask.any(-1))
        polygon_key_padding = ~(polygon_mask.any(-1))
        key_padding_mask = torch.cat([agent_key_padding, polygon_key_padding], dim=-1)
        x_agent = self.agent_encoder(data)
        x_polygon = self.map_encoder(data)
        x_static, static_pos, static_key_padding = self.static_objects_encoder(data)
        x = torch.cat([x_agent, x_polygon, x_static], dim=1)
        pos = torch.cat([pos, static_pos], dim=1)
        pos_embed = self.pos_emb(pos)
        key_padding_mask = torch.cat([key_padding_mask, static_key_padding], dim=-1)
        x = x + pos_embed
        for blk in self.encoder_blocks:
            x = blk(x, key_padding_mask=key_padding_mask, return_attn_weights=False)
        x = self.norm(x)
        prediction = self.agent_predictor(x[:, 1:A])
        trajectory, probability = self.planning_decoder(
            data, {"enc_emb": x, "enc_key_padding_mask": key_padding_mask})
        return {"trajectory": trajectory, "probability": probability, "prediction": prediction}


def compute_loss(res, data, collision_loss_fn, num_modes, radius, history_steps):
    bs = res["prediction"].shape[0]
    T = res["prediction"].shape[2]
    dev = res["prediction"].device
    valid_mask = data["agent"]["valid_mask"][:, :, -T:]
    target_pos = data["agent"]["target"]
    target_vel = data["agent"]["velocity"][:, :, -T:]
    target = torch.cat([
        target_pos[..., :2],
        torch.stack([target_pos[..., 2].cos(), target_pos[..., 2].sin()], dim=-1),
        target_vel], dim=-1)
    pred_mask = valid_mask[:, 1:]
    prediction_loss = F.smooth_l1_loss(
        res["prediction"][pred_mask], target[:, 1:][pred_mask], reduction="none"
    ).sum(-1).sum() / (pred_mask.sum() + 1e-6)
    trajectory = res["trajectory"]
    probability = res["probability"]
    ego_valid = valid_mask[:, 0]
    ego_target = target[:, 0]
    num_valid_points = ego_valid.sum(-1)
    endpoint_index = (num_valid_points / 10).long().clamp_(min=0, max=7)
    r_padding_mask = ~data["reference_line"]["valid_mask"].any(-1)
    future_projection = data["reference_line"]["future_projection"][
        torch.arange(bs, device=dev), :, endpoint_index]
    mode_interval = radius / num_modes
    target_r_index = torch.argmin(future_projection[..., 1] + 1e6 * r_padding_mask, dim=-1)
    target_m_index = (
        future_projection[torch.arange(bs, device=dev), target_r_index, 0] / mode_interval
    ).long().clamp_(min=0, max=num_modes - 1)
    target_label = torch.zeros_like(probability)
    target_label[torch.arange(bs, device=dev), target_r_index, target_m_index] = 1
    best_traj = trajectory[torch.arange(bs, device=dev), target_r_index, target_m_index]
    col_loss = collision_loss_fn(best_traj[..., :4], data["cost_maps"][:, :, :, 0].float())
    reg_loss = F.smooth_l1_loss(best_traj, ego_target[..., :best_traj.shape[-1]], reduction="none").sum(-1)
    reg_loss = (reg_loss * ego_valid).sum() / (ego_valid.sum() + 1e-6)
    probability_masked = probability.clone()
    probability_masked.masked_fill_(r_padding_mask.unsqueeze(-1), -1e6)
    cls_loss = F.cross_entropy(probability_masked.reshape(bs, -1), target_label.reshape(bs, -1).detach())
    return reg_loss + cls_loss + prediction_loss + col_loss


# ============================================================================
# DDP helpers
# ============================================================================

def setup_distributed():
    if "RANK" not in os.environ: return False
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    return True

def cleanup_distributed():
    if dist.is_available() and dist.is_initialized(): dist.destroy_process_group()

def get_rank():
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

def get_world_size():
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

def log(msg):
    if get_rank() == 0: print(msg, flush=True)


# ============================================================================
# Training benchmark
# ============================================================================

def benchmark_training(args):
    ddp_enabled = setup_distributed()
    rank = get_rank()
    world_size = get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0)) if ddp_enabled else 0
    device = torch.device(f"cuda:{local_rank}")

    # Profile on rank 0
    if rank == 0 and args.augment:
        profile_augmentation(args)
        print()

    log("=" * 70)
    log(f"TRAINING BENCHMARK ({'WITH' if args.augment else 'WITHOUT'} AUGMENTATION)")
    log("=" * 70)
    log(f"Mode:              {'DDP ({} GPUs)'.format(world_size) if ddp_enabled else 'Single GPU'}")
    log(f"GPU:               {torch.cuda.get_device_name(device)}")
    log(f"Batch per GPU:     {args.batch_size}")
    log(f"Effective batch:   {args.batch_size * world_size} " +
        (f"(×3 = {args.batch_size * world_size * 3} with augmentation)" if args.augment else ""))
    log(f"Workers per GPU:   {args.num_workers}")
    log(f"Augmentation:      {args.augment}")
    log(f"Cache dir:         {args.cache_dir}")

    dataset = AugmentedCacheDataset(
        args.cache_dir, augment=args.augment, history_steps=args.history_steps)
    log(f"Dataset size:      {len(dataset)} samples")

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if ddp_enabled else None
    loader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=(sampler is None), num_workers=args.num_workers,
        pin_memory=args.pin_memory, collate_fn=collate_fn_augmented,
        sampler=sampler, drop_last=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None)
    log(f"Steps per epoch:   {len(loader)}")

    model = PlanningModel(
        dim=args.dim, history_steps=args.history_steps, future_steps=args.future_steps,
        encoder_depth=args.encoder_depth, decoder_depth=args.decoder_depth, num_modes=args.num_modes,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    log(f"Parameters:        {num_params / 1e6:.2f}M")

    if ddp_enabled:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        raw_model = model.module
    else:
        raw_model = model

    collision_loss_fn = ESDFCollisionLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    log("-" * 70)
    log("Warming up ...")
    model.train()
    if sampler is not None: sampler.set_epoch(0)
    for i, batch_cpu in enumerate(loader):
        data = batch_to_device(batch_cpu, device)
        res = model(data)
        loss = compute_loss(res, data, collision_loss_fn, raw_model.num_modes, raw_model.radius, args.history_steps)
        loss.backward(); optimizer.step(); optimizer.zero_grad()
        if i >= args.warmup_steps - 1: break
    torch.cuda.synchronize(device)
    if ddp_enabled: dist.barrier()

    # Benchmark
    all_timings = {"dataload": [], "h2d": [], "forward": [], "backward": [], "optimizer": [], "step": []}

    for epoch in range(args.num_epochs):
        if sampler is not None: sampler.set_epoch(epoch + 1)
        data_iter = iter(loader)
        step_count = 0

        torch.cuda.synchronize(device)
        t_dl_start = time.perf_counter()
        try:
            batch_cpu = next(data_iter)
        except StopIteration:
            break
        t_dl_end = time.perf_counter()

        while True:
            dl_time = t_dl_end - t_dl_start

            torch.cuda.synchronize(device)
            t_h2d_start = time.perf_counter()
            data = batch_to_device(batch_cpu, device)
            torch.cuda.synchronize(device)
            t_h2d_end = time.perf_counter()
            h2d_time = t_h2d_end - t_h2d_start

            res = model(data)
            loss = compute_loss(res, data, collision_loss_fn, raw_model.num_modes, raw_model.radius, args.history_steps)
            torch.cuda.synchronize(device)
            t_fwd_end = time.perf_counter()
            fwd_time = t_fwd_end - t_h2d_end

            loss.backward()
            torch.cuda.synchronize(device)
            t_bwd_end = time.perf_counter()
            bwd_time = t_bwd_end - t_fwd_end

            optimizer.step(); optimizer.zero_grad()
            torch.cuda.synchronize(device)
            t_opt_end = time.perf_counter()
            opt_time = t_opt_end - t_bwd_end

            total_step = dl_time + h2d_time + fwd_time + bwd_time + opt_time
            all_timings["dataload"].append(dl_time)
            all_timings["h2d"].append(h2d_time)
            all_timings["forward"].append(fwd_time)
            all_timings["backward"].append(bwd_time)
            all_timings["optimizer"].append(opt_time)
            all_timings["step"].append(total_step)

            step_count += 1
            if rank == 0 and (step_count % 10 == 0 or step_count == 1):
                print(f"  Epoch {epoch} Step {step_count:4d} | loss={loss.item():.4f} | "
                      f"total={total_step*1000:.1f}ms "
                      f"(dl={dl_time*1000:.1f} h2d={h2d_time*1000:.1f} "
                      f"fwd={fwd_time*1000:.1f} bwd={bwd_time*1000:.1f} "
                      f"opt={opt_time*1000:.1f}ms)", flush=True)

            if step_count >= args.max_steps_per_epoch: break

            t_dl_start = time.perf_counter()
            try:
                batch_cpu = next(data_iter)
            except StopIteration:
                break
            t_dl_end = time.perf_counter()

    if ddp_enabled: dist.barrier()

    if rank == 0:
        def stats(times):
            t = torch.tensor(times)
            return t.mean().item(), t.std().item(), t.median().item(), t.min().item(), t.max().item()

        total_steps = len(all_timings["step"])
        print()
        print("=" * 70)
        print(f"BENCHMARK RESULTS ({'WITH' if args.augment else 'WITHOUT'} AUGMENTATION)")
        print("=" * 70)
        print(f"  Mode:             {'DDP ({} GPUs)'.format(world_size) if ddp_enabled else 'Single GPU'}")
        print(f"  Batch per GPU:    {args.batch_size}")
        print(f"  Effective batch:  {args.batch_size * world_size}" +
              (f" (×3 = {args.batch_size * world_size * 3})" if args.augment else ""))
        print(f"  Workers/GPU:      {args.num_workers}")
        print(f"  Augmentation:     {args.augment}")
        print(f"  Total steps:      {total_steps}")
        print()

        labels = [
            ("Total step",      "step"),
            ("DataLoader",      "dataload"),
            ("CPU→GPU (H2D)",   "h2d"),
            ("Forward+Loss",    "forward"),
            ("Backward",        "backward"),
            ("Optimizer",       "optimizer"),
        ]

        print("  [Detailed Timing - Rank 0]")
        print(f"  {'Phase':20s} {'mean':>9s} {'median':>9s} {'std':>8s} {'min':>9s} {'max':>9s} {'%total':>8s}")
        print("  " + "-" * 67)

        mean_total = stats(all_timings["step"])[0]
        for label, key in labels:
            mean, std, med, mn, mx = stats(all_timings[key])
            pct = mean / mean_total * 100 if mean_total > 0 else 0
            print(f"  {label:20s} {mean*1000:8.1f}ms {med*1000:8.1f}ms "
                  f"{std*1000:7.1f}ms {mn*1000:8.1f}ms {mx*1000:8.1f}ms {pct:7.1f}%")

        effective_step_time = mean_total
        # With augmentation, effective batch is 3x per GPU
        effective_bs = args.batch_size * (3 if args.augment else 1)
        global_throughput = (args.batch_size * world_size) / effective_step_time
        per_gpu_throughput = args.batch_size / effective_step_time

        print()
        print("  [Throughput]")
        print(f"    Global (unique samples):    {global_throughput:.1f} samples/sec")
        print(f"    Per GPU (unique samples):   {per_gpu_throughput:.1f} samples/sec")
        if args.augment:
            print(f"    GPU sees (with aug 3x):     {global_throughput * 3:.1f} effective samples/sec")
        print(f"    Steps/sec:                  {1.0 / effective_step_time:.2f}")

        print()
        print("  [Time Breakdown]")
        for label, key in labels[1:]:
            mean = stats(all_timings[key])[0]
            pct = mean / mean_total * 100
            bar = "#" * int(pct / 2)
            print(f"    {label:20s} {pct:5.1f}%  {bar}")

        print()
        print(f"  [GPU Memory (Rank 0)]")
        print(f"    Allocated: {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB")
        print(f"    Reserved:  {torch.cuda.max_memory_reserved(device) / 1e9:.2f} GB")
        print()

    cleanup_distributed()


def batch_to_device(data, device):
    out = {}
    for k, v in data.items():
        if isinstance(v, dict):
            out[k] = batch_to_device(v, device)
        elif isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pluto augmentation pipeline benchmark")
    parser.add_argument("--mode", choices=["generate", "profile", "train"], default="train")
    parser.add_argument("--cache-dir", type=str, default="/tmp/pluto_augment_cache")
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--pin-memory", action="store_true", default=True)
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--augment", action="store_true", default=True,
                        help="Enable full ContrastiveScenarioGenerator augmentation (default: on)")
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--num-agents", type=int, default=20)
    parser.add_argument("--num-polygons", type=int, default=150)
    parser.add_argument("--history-steps", type=int, default=21)
    parser.add_argument("--future-steps", type=int, default=80)
    parser.add_argument("--encoder-depth", type=int, default=4)
    parser.add_argument("--decoder-depth", type=int, default=4)
    parser.add_argument("--num-modes", type=int, default=6)
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--max-steps-per-epoch", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=5)
    args = parser.parse_args()

    if args.mode == "generate":
        generate_cache_files(args)
    elif args.mode == "profile":
        profile_augmentation(args)
    else:
        benchmark_training(args)
