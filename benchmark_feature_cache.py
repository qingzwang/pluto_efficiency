#!/usr/bin/env python3
"""
Benchmark Pluto training with simulated feature cache (gzip pickle).

This simulates the real nuplan training pipeline when using
`cache.use_cache_without_dataset=true`:
  1. gzip.open + pickle.load to read cached PlutoFeature
  2. PlutoFeature.deserialize → to_feature_tensor (numpy→torch)
  3. Data augmentation (contrastive scenario generator) [optional]
  4. PlutoFeature.collate (pad_sequence for variable-length features)
  5. CPU→GPU transfer
  6. Forward + Loss + Backward + Optimizer

Usage:
  # 1. Generate cache files
  python benchmark_feature_cache.py --mode generate --num-samples 5000 \
    --cache-dir /tmp/pluto_feature_cache

  # 2. Single GPU benchmark
  python benchmark_feature_cache.py --mode benchmark --cache-dir /tmp/pluto_feature_cache

  # 3. 8-GPU DDP benchmark
  torchrun --nproc_per_node=8 benchmark_feature_cache.py --mode benchmark \
    --cache-dir /tmp/pluto_feature_cache --batch-size 32

  # 4. Compare: with vs without augmentation
  torchrun --nproc_per_node=8 benchmark_feature_cache.py --mode benchmark \
    --cache-dir /tmp/pluto_feature_cache --batch-size 32 --augment
"""

import argparse
import gzip
import math
import os
import pickle
import time

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
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
# Generate cached features in nuplan's exact format (gzip pickle)
# ============================================================================

def generate_cached_feature(
    num_agents=20, num_polygons=150, num_points_per_polygon=20,
    num_static_objects=10, num_ref_lines=6, num_ref_points=100,
    history_steps=21, future_steps=80, cost_map_size=200,
):
    """Generate a PlutoFeature-compatible data dict (numpy arrays, post-normalization)."""
    T = history_steps + future_steps
    A, M, P = num_agents, num_polygons, num_points_per_polygon
    N_s, R, P_ref = num_static_objects, num_ref_lines, num_ref_points

    data = {
        "current_state": np.zeros(7, dtype=np.float64),  # normalized: ego at origin
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
    return data


def generate_cache_files(args):
    """Generate cache files in nuplan's exact format: gzip-compressed pickle."""
    os.makedirs(args.cache_dir, exist_ok=True)

    print(f"Generating {args.num_samples} cached features to {args.cache_dir} ...")
    print(f"Format: gzip-compressed pickle (same as nuplan FeatureCachePickle)")

    for i in range(args.num_samples):
        data = generate_cached_feature(
            num_agents=args.num_agents,
            num_polygons=args.num_polygons,
            history_steps=args.history_steps,
            future_steps=args.future_steps,
        )
        # Exact same format as nuplan: PlutoFeature.serialize() returns {"data": self.data}
        serializable_dict = {"data": data}

        path = os.path.join(args.cache_dir, f"feature_{i:06d}.gz")
        with gzip.open(path, 'wb', compresslevel=1) as f:
            pickle.dump(serializable_dict, f)

        if (i + 1) % 500 == 0 or i == 0:
            print(f"  {i+1}/{args.num_samples}")

    # Check sizes
    sample_path = os.path.join(args.cache_dir, "feature_000000.gz")
    gz_size = os.path.getsize(sample_path) / 1e6
    total_gb = gz_size * args.num_samples / 1e3
    print(f"Done. Per-sample .gz size: {gz_size:.2f} MB, Total: {total_gb:.2f} GB")

    # Also measure uncompressed size
    with gzip.open(sample_path, 'rb') as f:
        raw = f.read()
    print(f"Uncompressed pickle size: {len(raw)/1e6:.2f} MB")
    print(f"Compression ratio: {len(raw)/os.path.getsize(sample_path):.1f}x")


# ============================================================================
# Dataset that reads from gzip pickle cache
# ============================================================================

def to_tensor(obj):
    """Convert nested numpy dict to torch tensors (like PlutoFeature.to_feature_tensor)."""
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


def simulate_augmentation(data, history_steps=21):
    """Simulate ContrastiveScenarioGenerator: generate positive sample with noise.
    Real augmentation adds noise to ego state and checks for collisions.
    """
    import copy
    data_p = copy.deepcopy(data)

    # Add noise to ego current state (similar to ContrastiveScenarioGenerator)
    noise = np.random.uniform(
        [-1.0, -0.75, -0.35, -1, -0.5, -0.2, -0.1],
        [1.0, 0.75, 0.35, 1, 0.5, 0.2, 0.1],
    )
    data_p["current_state"][:7] += noise

    # Shift ego agent position slightly
    data_p["agent"]["position"][0] += noise[:2]
    data_p["agent"]["heading"][0] += noise[2]

    return data_p


class CachedFeatureDataset(Dataset):
    """Dataset that reads from gzip pickle cache (same as nuplan FeatureCachePickle)."""

    def __init__(self, cache_dir, augment=False, history_steps=21):
        self.files = sorted([
            os.path.join(cache_dir, f)
            for f in os.listdir(cache_dir) if f.endswith(".gz")
        ])
        assert len(self.files) > 0, f"No .gz files found in {cache_dir}"
        self.augment = augment
        self.history_steps = history_steps

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Step 1: gzip decompress + pickle load (same as FeatureCachePickle)
        with gzip.open(self.files[idx], 'rb') as f:
            serialized = pickle.load(f)

        # Step 2: deserialize (same as PlutoFeature.deserialize)
        data = serialized["data"]

        # Step 3: augmentation (if enabled)
        data_p = None
        if self.augment:
            data_p = simulate_augmentation(data, self.history_steps)

        # Step 4: to_feature_tensor (numpy → torch)
        tensor_data = to_tensor(data)

        if data_p is not None:
            tensor_data_p = to_tensor(data_p)
            # Return both for collation
            return tensor_data, tensor_data_p

        return tensor_data


class PtFileDataset(Dataset):
    """Baseline: read pre-saved torch .pt files (our existing benchmark format)."""
    def __init__(self, data_dir):
        self.files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir) if f.endswith(".pt")
        ])
        assert len(self.files) > 0

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(self.files[idx], map_location="cpu")


def collate_fn(batch):
    """Collate function for cached features."""
    # Handle augmented vs non-augmented batches
    if isinstance(batch[0], tuple):
        # Augmented: (data, data_p) pairs
        data_list = [b[0] for b in batch]
        data_p_list = [b[1] for b in batch]
        all_samples = data_list + data_p_list
    else:
        all_samples = batch

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
        if key in all_samples[0]:
            out[key] = torch.stack([s[key] for s in all_samples], dim=0)
    return out


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
# IO Benchmark: measure per-sample cache read time
# ============================================================================

def benchmark_io(args):
    """Measure per-sample cache IO time for different formats."""
    log("=" * 70)
    log("CACHE IO BENCHMARK")
    log("=" * 70)

    cache_files = sorted([
        os.path.join(args.cache_dir, f)
        for f in os.listdir(args.cache_dir) if f.endswith(".gz")
    ])
    num_trials = min(100, len(cache_files))

    # 1. gzip pickle read (nuplan cache format)
    gz_times = []
    deser_times = []
    tensor_times = []
    aug_times = []
    total_times = []

    for i in range(num_trials):
        t0 = time.perf_counter()

        # gzip + pickle
        with gzip.open(cache_files[i], 'rb') as f:
            serialized = pickle.load(f)
        t1 = time.perf_counter()
        gz_times.append(t1 - t0)

        # deserialize
        data = serialized["data"]
        t2 = time.perf_counter()
        deser_times.append(t2 - t1)

        # to_feature_tensor
        tensor_data = to_tensor(data)
        t3 = time.perf_counter()
        tensor_times.append(t3 - t2)

        # augmentation
        if args.augment:
            data_p = simulate_augmentation(data)
            to_tensor(data_p)
        t4 = time.perf_counter()
        aug_times.append(t4 - t3)

        total_times.append(t4 - t0)

    log(f"\n  Per-sample cache read time ({num_trials} trials):")
    log(f"  {'Phase':25s} {'mean':>8s} {'median':>8s} {'std':>7s} {'%total':>7s}")
    log("  " + "-" * 56)
    mean_total = np.mean(total_times)
    for name, vals in [
        ("gzip+pickle", gz_times),
        ("deserialize", deser_times),
        ("numpy→torch", tensor_times),
        ("augmentation", aug_times),
        ("total", total_times),
    ]:
        v = np.array(vals) * 1000
        pct = np.mean(vals) / mean_total * 100
        log(f"  {name:25s} {v.mean():7.1f}ms {np.median(v):7.1f}ms {v.std():6.1f}ms {pct:6.1f}%")

    # File size info
    gz_size = os.path.getsize(cache_files[0]) / 1e6
    log(f"\n  Cache file size: {gz_size:.2f} MB/sample (.gz)")

    # 2. Compare with torch.load (.pt) if available
    if args.pt_dir and os.path.exists(args.pt_dir):
        pt_files = sorted([
            os.path.join(args.pt_dir, f)
            for f in os.listdir(args.pt_dir) if f.endswith(".pt")
        ])
        pt_times = []
        for i in range(min(num_trials, len(pt_files))):
            t0 = time.perf_counter()
            torch.load(pt_files[i], map_location="cpu")
            pt_times.append(time.perf_counter() - t0)

        pt_v = np.array(pt_times) * 1000
        pt_size = os.path.getsize(pt_files[0]) / 1e6
        log(f"\n  Comparison: torch.load (.pt)")
        log(f"  {'torch.load':25s} {pt_v.mean():7.1f}ms {np.median(pt_v):7.1f}ms {pt_v.std():6.1f}ms")
        log(f"  .pt file size: {pt_size:.2f} MB/sample")
        log(f"  Speedup vs cache: {np.mean(total_times)/np.mean(pt_times):.1f}x slower")

    # Worker throughput estimates
    log(f"\n  Effective worker throughput ({mean_total*1000:.0f}ms/sample):")
    for nw in [1, 2, 4, 8, 16]:
        thru = nw / mean_total
        log(f"    {nw:2d} workers: {thru:7.1f} samples/sec")

    return mean_total


# ============================================================================
# Training benchmark
# ============================================================================

def benchmark_training(args):
    ddp_enabled = setup_distributed()
    rank = get_rank()
    world_size = get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0)) if ddp_enabled else 0
    device = torch.device(f"cuda:{local_rank}")

    # IO benchmark (rank 0 only)
    if rank == 0:
        benchmark_io(args)

    log("")
    log("=" * 70)
    log("TRAINING BENCHMARK (feature cache → GPU)")
    log("=" * 70)
    log(f"Mode:              {'DDP ({} GPUs)'.format(world_size) if ddp_enabled else 'Single GPU'}")
    log(f"GPU:               {torch.cuda.get_device_name(device)}")
    log(f"Batch per GPU:     {args.batch_size}")
    log(f"Global batch:      {args.batch_size * world_size}")
    log(f"Workers per GPU:   {args.num_workers}")
    log(f"Augmentation:      {args.augment}")
    log(f"Pin memory:        {args.pin_memory}")
    log(f"Cache dir:         {args.cache_dir}")

    dataset = CachedFeatureDataset(
        args.cache_dir, augment=args.augment, history_steps=args.history_steps)
    log(f"Dataset size:      {len(dataset)} samples")

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if ddp_enabled else None
    loader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=(sampler is None), num_workers=args.num_workers,
        pin_memory=args.pin_memory, collate_fn=collate_fn,
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

    # Warmup
    log("Warming up ...")
    model.train()
    if sampler is not None: sampler.set_epoch(0)
    for i, batch_cpu in enumerate(loader):
        if isinstance(batch_cpu, tuple):
            batch_cpu = batch_cpu[0]  # handle augmented case
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

    # Results
    if rank == 0:
        def stats(times):
            t = torch.tensor(times)
            return t.mean().item(), t.std().item(), t.median().item(), t.min().item(), t.max().item()

        total_steps = len(all_timings["step"])
        print()
        print("=" * 70)
        print("BENCHMARK RESULTS")
        print("=" * 70)
        print(f"  Mode:             {'DDP ({} GPUs)'.format(world_size) if ddp_enabled else 'Single GPU'}")
        print(f"  Batch per GPU:    {args.batch_size}")
        print(f"  Global batch:     {args.batch_size * world_size}")
        print(f"  Workers/GPU:      {args.num_workers}")
        print(f"  Augmentation:     {args.augment}")
        print(f"  Pin memory:       {args.pin_memory}")
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
        global_throughput = (args.batch_size * world_size) / effective_step_time
        per_gpu_throughput = args.batch_size / effective_step_time

        print()
        print("  [Throughput]")
        print(f"    Global:    {global_throughput:.1f} samples/sec")
        print(f"    Per GPU:   {per_gpu_throughput:.1f} samples/sec")
        print(f"    Steps/sec: {1.0 / effective_step_time:.2f}")

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


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pluto feature cache training benchmark")
    parser.add_argument("--mode", choices=["generate", "benchmark", "io"], default="benchmark")
    parser.add_argument("--cache-dir", type=str, default="/tmp/pluto_feature_cache")
    parser.add_argument("--pt-dir", type=str, default="/tmp/pluto_bench_data_20k",
                        help="Pre-generated .pt dir for comparison")
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--pin-memory", action="store_true", default=True)
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--augment", action="store_true", default=False,
                        help="Simulate contrastive data augmentation")
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
    elif args.mode == "io":
        benchmark_io(args)
    else:
        benchmark_training(args)
