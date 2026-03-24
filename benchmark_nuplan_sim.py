#!/usr/bin/env python3
"""
Simulate nuplan-like training pipeline to benchmark realistic CPU preprocessing overhead.

The real nuplan pipeline does heavy CPU work in __getitem__:
  1. Map API queries + polyline interpolation (~100-500ms)
  2. Cost map generation: cv2.fillPoly + scipy distance_transform_edt (~100-300ms)
  3. Coordinate normalization: numpy matmul on all features (~10-50ms)
  4. Causal reasoning: shapely geometry operations (~20-100ms)
  5. Agent tracking across timesteps (~20-50ms)

This script simulates these CPU operations without requiring nuplan data, then
runs the full DDP training loop to measure end-to-end throughput.

Usage:
  # Single GPU
  python benchmark_nuplan_sim.py --batch-size 32

  # 8-GPU DDP
  torchrun --nproc_per_node=8 benchmark_nuplan_sim.py --batch-size 32

  # Skip CPU simulation (pure .pt loading baseline)
  torchrun --nproc_per_node=8 benchmark_nuplan_sim.py --batch-size 32 --no-simulate-cpu
"""

import argparse
import math
import os
import time
from typing import Dict

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
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
# Model (same as benchmark_training_e2e.py)
# ============================================================================

class PlanningModel(nn.Module):
    def __init__(
        self, dim=128, state_channel=6, polygon_channel=6, history_channel=9,
        history_steps=21, future_steps=80, encoder_depth=4, decoder_depth=4,
        drop_path=0.2, dropout=0.1, num_heads=8, num_modes=6,
        use_ego_history=False, state_attn_encoder=True, state_dropout=0.75, radius=100,
    ):
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
            state_dropout=state_dropout,
        )
        self.map_encoder = MapEncoder(dim=dim, polygon_channel=polygon_channel, use_lane_boundary=True)
        self.static_objects_encoder = StaticObjectsEncoder(dim=dim)
        self.encoder_blocks = nn.ModuleList(
            TransformerEncoderLayer(dim=dim, num_heads=num_heads, drop_path=dp)
            for dp in [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.agent_predictor = AgentPredictor(dim=dim, future_steps=future_steps)
        self.planning_decoder = PlanningDecoder(
            num_mode=num_modes, decoder_depth=decoder_depth, dim=dim,
            num_heads=num_heads, mlp_ratio=4, dropout=dropout, cat_x=False,
            future_steps=future_steps,
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
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
            data, {"enc_emb": x, "enc_key_padding_mask": key_padding_mask}
        )
        return {"trajectory": trajectory, "probability": probability, "prediction": prediction}


# ============================================================================
# Simulated CPU preprocessing (mimics PlutoFeatureBuilder)
# ============================================================================

def simulate_map_query_and_interpolation(num_polygons=150, sample_points=20):
    """Simulate map API queries + polyline interpolation.
    Real pipeline: query proximal map objects, sample discrete paths, compute vectors.
    """
    # Simulate querying ~150 lane/connector/crosswalk objects
    # Each has centerline, left_boundary, right_boundary with ~50-200 raw points
    # Then interpolated to sample_points+1 points
    M, P = num_polygons, sample_points
    point_position = np.zeros((M, 3, P, 2), dtype=np.float64)
    point_vector = np.zeros((M, 3, P, 2), dtype=np.float64)
    point_orientation = np.zeros((M, 3, P), dtype=np.float64)
    polygon_center = np.zeros((M, 3), dtype=np.float64)

    for i in range(M):
        # Simulate 3 polylines per polygon (center, left, right boundary)
        for edge in range(3):
            # Random raw path with ~80 points (typical lane length)
            raw_path = np.cumsum(np.random.randn(80, 2) * 0.5, axis=0)
            # Interpolate to sample_points+1
            t_raw = np.linspace(0, 1, len(raw_path))
            t_new = np.linspace(0, 1, P + 1)
            interp_x = np.interp(t_new, t_raw, raw_path[:, 0])
            interp_y = np.interp(t_new, t_raw, raw_path[:, 1])
            edges = np.stack([interp_x, interp_y], axis=-1)

            point_position[i, edge] = edges[:-1]
            point_vector[i, edge] = edges[1:] - edges[:-1]
            point_orientation[i, edge] = np.arctan2(
                point_vector[i, edge, :, 1], point_vector[i, edge, :, 0]
            )

        polygon_center[i, :2] = point_position[i, 0, P // 2]
        polygon_center[i, 2] = point_orientation[i, 0, P // 2]

    return point_position, point_vector, point_orientation, polygon_center


def simulate_cost_map_generation(height=500, width=500, resolution=0.2, num_polygons=80):
    """Simulate CostMapManager.build_cost_maps().
    Real pipeline: cv2.fillPoly for drivable area, scipy distance_transform_edt for SDF.
    """
    drivable_area_mask = np.zeros((height, width), dtype=np.uint8)

    # Fill ~80 random polygons (simulating lane polygons)
    for _ in range(num_polygons):
        n_vertices = np.random.randint(4, 8)
        center = np.random.rand(2) * np.array([height, width])
        angles = np.sort(np.random.uniform(0, 2 * np.pi, n_vertices))
        radii = np.random.uniform(10, 40, n_vertices)
        pts = center + np.stack([radii * np.cos(angles), radii * np.sin(angles)], -1)
        pts = np.clip(pts, 0, min(height, width) - 1).astype(np.int32)
        cv2.fillPoly(drivable_area_mask, [pts], 1)

    # SDF computation (the main bottleneck)
    distance = ndimage.distance_transform_edt(drivable_area_mask)
    inv_distance = ndimage.distance_transform_edt(1 - drivable_area_mask)
    sdf = (distance - inv_distance) * resolution

    return sdf[:, :, None].astype(np.float16)


def simulate_causal_reasoning(num_agents=20, num_polygons=150):
    """Simulate shapely geometry operations for causal reasoning.
    Real pipeline: path buffering, intersection tests, distance computations.
    """
    # Build ego path as LineString
    ego_path_pts = np.cumsum(np.random.randn(50, 2) * 2, axis=0)
    ego_path = LineString(ego_path_pts)
    ego_buffered = ego_path.buffer(1.0)

    # Create agent polygons and test intersections
    leading_agent_mask = np.zeros(num_agents, dtype=bool)
    leading_distance = np.zeros(num_agents, dtype=np.float64)
    ego_polygon = Polygon(ego_path_pts[:4])

    for i in range(min(num_agents, 20)):
        center = np.random.randn(2) * 20
        w, l = np.random.uniform(1.5, 2.5), np.random.uniform(3.5, 5.5)
        angle = np.random.uniform(-np.pi, np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        corners = np.array([[-l/2, -w/2], [l/2, -w/2], [l/2, w/2], [-l/2, w/2]])
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        corners = corners @ rot.T + center
        agent_poly = Polygon(corners)

        if ego_buffered.intersects(agent_poly):
            leading_agent_mask[i] = True
            leading_distance[i] = ego_polygon.distance(agent_poly)

    # Simulate red light polygon intersection
    for _ in range(5):
        rl_center = np.random.randn(2) * 30
        rl_poly = Point(rl_center).buffer(3.0)
        ego_buffered.intersects(rl_poly)

    return leading_agent_mask, leading_distance


def simulate_coordinate_normalization(data, hist_steps=21):
    """Simulate PlutoFeature.normalize() - ego-centric coordinate transforms."""
    center_xy = data["agent"]["position"][0, hist_steps - 1].copy()
    center_angle = data["agent"]["heading"][0, hist_steps - 1].copy()

    rotate_mat = np.array([
        [np.cos(center_angle), -np.sin(center_angle)],
        [np.sin(center_angle), np.cos(center_angle)],
    ], dtype=np.float64)

    # Transform agents
    data["agent"]["position"] = np.matmul(data["agent"]["position"] - center_xy, rotate_mat)
    data["agent"]["velocity"] = np.matmul(data["agent"]["velocity"], rotate_mat)
    data["agent"]["heading"] -= center_angle

    # Transform map
    data["map"]["point_position"] = np.matmul(
        data["map"]["point_position"] - center_xy, rotate_mat
    )
    data["map"]["point_vector"] = np.matmul(data["map"]["point_vector"], rotate_mat)
    data["map"]["polygon_center"][..., :2] = np.matmul(
        data["map"]["polygon_center"][..., :2] - center_xy, rotate_mat
    )

    # Transform reference lines
    data["reference_line"]["position"] = np.matmul(
        data["reference_line"]["position"] - center_xy, rotate_mat
    )
    data["reference_line"]["vector"] = np.matmul(
        data["reference_line"]["vector"], rotate_mat
    )

    # Transform static objects
    data["static_objects"]["position"] = np.matmul(
        data["static_objects"]["position"] - center_xy, rotate_mat
    )

    # Compute targets
    target_position = (
        data["agent"]["position"][:, hist_steps:]
        - data["agent"]["position"][:, hist_steps - 1][:, None]
    )
    target_heading = (
        data["agent"]["heading"][:, hist_steps:]
        - data["agent"]["heading"][:, hist_steps - 1][:, None]
    )
    target = np.concatenate([target_position, target_heading[..., None]], -1)
    data["agent"]["target"] = target


def simulate_agent_tracking(num_agents=20, T=101):
    """Simulate agent detection + cross-timestep tracking.
    Real pipeline: iterate through T timesteps of TrackedObjects, match by track_token.
    """
    position = np.random.randn(num_agents, T, 2).astype(np.float64)
    # Simulate cumulative motion
    position = np.cumsum(position * 0.1, axis=1)
    heading = np.random.randn(num_agents, T).astype(np.float64) * 0.1
    heading = np.cumsum(heading, axis=1)
    velocity = np.diff(position, axis=1, prepend=position[:, :1]) / 0.1
    shape = np.random.uniform(1.5, 5.0, (num_agents, T, 2)).astype(np.float64)
    category = np.random.randint(0, 4, (num_agents,)).astype(np.int8)
    valid_mask = np.ones((num_agents, T), dtype=bool)

    # Simulate distance-based sorting (select closest agents)
    ego_pos = position[0, T // 2]
    distances = np.linalg.norm(position[:, T // 2] - ego_pos, axis=-1)
    sorted_idx = np.argsort(distances)[:num_agents]
    position = position[sorted_idx]
    heading = heading[sorted_idx]
    velocity = velocity[sorted_idx]
    shape = shape[sorted_idx]
    category = category[sorted_idx]
    valid_mask = valid_mask[sorted_idx]

    return {
        "position": position,
        "heading": heading,
        "velocity": velocity,
        "shape": shape,
        "category": category,
        "valid_mask": valid_mask,
    }


def simulate_reference_lines(num_lines=6, num_points=100):
    """Simulate reference line computation with shapely projections."""
    position = np.zeros((num_lines, num_points, 2), dtype=np.float64)
    vector = np.zeros((num_lines, num_points, 2), dtype=np.float64)
    orientation = np.zeros((num_lines, num_points), dtype=np.float64)
    valid_mask = np.zeros((num_lines, num_points), dtype=bool)
    future_projection = np.zeros((num_lines, 8, 2), dtype=np.float64)

    for i in range(num_lines):
        # Random path
        raw = np.cumsum(np.random.randn(400, 2) * 0.5, axis=0)
        subsample = raw[::4][:num_points + 1]
        n_valid = min(len(subsample) - 1, num_points)

        position[i, :n_valid] = subsample[:n_valid, :2]
        vector[i, :n_valid] = np.diff(subsample[:n_valid + 1, :2], axis=0)[:n_valid]
        orientation[i, :n_valid] = np.arctan2(vector[i, :n_valid, 1], vector[i, :n_valid, 0])
        valid_mask[i, :n_valid] = True

        # Shapely projections (simulates future_projection computation)
        ls = LineString(subsample)
        future_pts = [Point(np.random.randn(2) * 20) for _ in range(8)]
        for j, pt in enumerate(future_pts):
            future_projection[i, j, 0] = ls.project(pt)
            future_projection[i, j, 1] = ls.distance(pt)

    return {
        "position": position,
        "vector": vector,
        "orientation": orientation,
        "valid_mask": valid_mask,
        "future_projection": future_projection,
    }


def generate_sample_with_cpu_simulation(
    num_agents=20, num_polygons=150, history_steps=21, future_steps=80,
    cost_map_height=500, cost_map_width=500,
):
    """Generate one sample while running CPU-heavy operations that mimic the real pipeline."""
    T = history_steps + future_steps

    # 1. Agent tracking (~20-50ms)
    agent_data = simulate_agent_tracking(num_agents + 1, T)  # +1 for ego

    # 2. Map queries + interpolation (~100-500ms)
    point_position, point_vector, point_orientation, polygon_center = \
        simulate_map_query_and_interpolation(num_polygons)

    # 3. Cost map generation (~100-300ms)
    cost_maps = simulate_cost_map_generation(cost_map_height, cost_map_width)

    # 4. Causal reasoning (~20-100ms)
    leading_mask, leading_dist = simulate_causal_reasoning(num_agents, num_polygons)

    # 5. Reference lines (~20-50ms)
    ref_line_data = simulate_reference_lines()

    # 6. Static objects
    N_s = 10
    static_objects = {
        "position": np.random.randn(N_s, 2).astype(np.float64),
        "heading": np.random.randn(N_s).astype(np.float64),
        "shape": np.random.uniform(0.5, 2.0, (N_s, 2)).astype(np.float64),
        "category": np.random.randint(0, 4, (N_s,)).astype(np.int8),
        "valid_mask": np.ones(N_s, dtype=bool),
    }

    # Assemble data dict
    data = {
        "agent": agent_data,
        "current_state": np.random.randn(7).astype(np.float64),
        "map": {
            "polygon_center": polygon_center,
            "polygon_type": np.random.randint(0, 3, (num_polygons,)).astype(np.int8),
            "polygon_on_route": (np.random.rand(num_polygons) > 0.5),
            "polygon_tl_status": np.random.randint(0, 4, (num_polygons,)).astype(np.int8),
            "polygon_has_speed_limit": (np.random.rand(num_polygons) > 0.5),
            "polygon_speed_limit": np.random.rand(num_polygons).astype(np.float64) * 30,
            "point_position": point_position,
            "point_vector": point_vector,
            "point_orientation": point_orientation,
            "valid_mask": np.ones((num_polygons, 20), dtype=bool),
        },
        "static_objects": static_objects,
        "reference_line": ref_line_data,
        "cost_maps": cost_maps,
    }

    # 7. Coordinate normalization (~10-50ms)
    simulate_coordinate_normalization(data, history_steps)

    # Convert to tensors
    def to_tensor(obj):
        if isinstance(obj, dict):
            return {k: to_tensor(v) for k, v in obj.items()}
        elif isinstance(obj, np.ndarray):
            if obj.dtype == bool:
                return torch.from_numpy(obj.copy()).bool()
            return torch.from_numpy(obj.copy()).float()
        elif isinstance(obj, (int, float, np.floating, np.integer)):
            return torch.tensor(float(obj))
        return obj

    return to_tensor(data)


# ============================================================================
# Dataset with simulated CPU preprocessing
# ============================================================================

class NuplanSimDataset(Dataset):
    """Dataset that simulates nuplan CPU preprocessing on every __getitem__ call."""

    def __init__(self, num_samples, simulate_cpu=True, data_dir=None,
                 num_agents=20, num_polygons=150, history_steps=21, future_steps=80):
        self.num_samples = num_samples
        self.simulate_cpu = simulate_cpu
        self.data_dir = data_dir
        self.num_agents = num_agents
        self.num_polygons = num_polygons
        self.history_steps = history_steps
        self.future_steps = future_steps

        # If data_dir provided and exists, load pre-generated .pt files as baseline
        self.files = None
        if data_dir and os.path.exists(data_dir):
            self.files = sorted([
                os.path.join(data_dir, f)
                for f in os.listdir(data_dir) if f.endswith(".pt")
            ])
            self.num_samples = len(self.files)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.simulate_cpu:
            # Run CPU-heavy operations to simulate real feature extraction
            return generate_sample_with_cpu_simulation(
                num_agents=self.num_agents,
                num_polygons=self.num_polygons,
                history_steps=self.history_steps,
                future_steps=self.future_steps,
            )
        elif self.files:
            return torch.load(self.files[idx % len(self.files)], map_location="cpu")
        else:
            # Fast path: just generate random tensors (no CPU simulation)
            return generate_sample_with_cpu_simulation(
                num_agents=self.num_agents,
                num_polygons=self.num_polygons,
                history_steps=self.history_steps,
                future_steps=self.future_steps,
            )


def collate_fn(batch):
    """Stack/pad a list of samples into a batched dict (mimics PlutoFeature.collate)."""
    pad_keys = ["agent", "map", "static_objects", "reference_line"]
    stack_keys = ["current_state", "cost_maps"]

    out = {}
    for key in pad_keys:
        out[key] = {
            k: pad_sequence([sample[key][k] for sample in batch], batch_first=True)
            for k in batch[0][key].keys()
        }
    for key in stack_keys:
        out[key] = torch.stack([sample[key] for sample in batch], dim=0)
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
# Loss (same as benchmark_training_e2e.py)
# ============================================================================

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
        target_vel,
    ], dim=-1)
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
        torch.arange(bs, device=dev), :, endpoint_index
    ]
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
    if "RANK" not in os.environ:
        return False
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    return True

def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def get_rank():
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

def get_world_size():
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

def log(msg):
    if get_rank() == 0:
        print(msg, flush=True)


# ============================================================================
# Benchmark: measure per-sample CPU preprocessing time
# ============================================================================

def benchmark_cpu_preprocessing(args):
    """Measure per-sample CPU feature extraction time."""
    log("=" * 70)
    log("CPU PREPROCESSING BENCHMARK (simulating nuplan feature extraction)")
    log("=" * 70)

    timings = {
        "agent_tracking": [],
        "map_interpolation": [],
        "cost_map": [],
        "causal_reasoning": [],
        "reference_lines": [],
        "normalization": [],
        "total": [],
    }

    num_trials = 50

    for i in range(num_trials):
        t0 = time.perf_counter()

        t_start = time.perf_counter()
        agent_data = simulate_agent_tracking(args.num_agents + 1, args.history_steps + args.future_steps)
        timings["agent_tracking"].append(time.perf_counter() - t_start)

        t_start = time.perf_counter()
        pp, pv, po, pc = simulate_map_query_and_interpolation(args.num_polygons)
        timings["map_interpolation"].append(time.perf_counter() - t_start)

        t_start = time.perf_counter()
        cost_maps = simulate_cost_map_generation()
        timings["cost_map"].append(time.perf_counter() - t_start)

        t_start = time.perf_counter()
        simulate_causal_reasoning(args.num_agents, args.num_polygons)
        timings["causal_reasoning"].append(time.perf_counter() - t_start)

        t_start = time.perf_counter()
        simulate_reference_lines()
        timings["reference_lines"].append(time.perf_counter() - t_start)

        # Build data dict for normalization
        N_s = 10
        data = {
            "agent": agent_data,
            "current_state": np.random.randn(7).astype(np.float64),
            "map": {
                "polygon_center": pc,
                "point_position": pp,
                "point_vector": pv,
                "point_orientation": po,
                "valid_mask": np.ones((args.num_polygons, 20), dtype=bool),
            },
            "static_objects": {
                "position": np.random.randn(N_s, 2),
                "heading": np.random.randn(N_s),
            },
            "reference_line": {
                "position": np.random.randn(6, 100, 2),
                "vector": np.random.randn(6, 100, 2),
                "orientation": np.random.randn(6, 100),
            },
        }

        t_start = time.perf_counter()
        simulate_coordinate_normalization(data, args.history_steps)
        timings["normalization"].append(time.perf_counter() - t_start)

        timings["total"].append(time.perf_counter() - t0)

    log(f"\n  Per-sample CPU preprocessing time ({num_trials} trials):")
    log(f"  {'Phase':25s} {'mean':>9s} {'median':>9s} {'std':>8s} {'%total':>8s}")
    log("  " + "-" * 60)

    mean_total = np.mean(timings["total"])
    for key in ["agent_tracking", "map_interpolation", "cost_map",
                "causal_reasoning", "reference_lines", "normalization", "total"]:
        vals = np.array(timings[key]) * 1000  # ms
        pct = np.mean(timings[key]) / mean_total * 100
        log(f"  {key:25s} {vals.mean():8.1f}ms {np.median(vals):8.1f}ms "
            f"{vals.std():7.1f}ms {pct:7.1f}%")

    log(f"\n  Effective worker throughput at {mean_total*1000:.0f}ms/sample:")
    for nw in [1, 2, 4, 8]:
        # With nw workers, throughput ≈ nw / per_sample_time
        thru = nw / mean_total
        log(f"    {nw} workers: {thru:.1f} samples/sec")

    return mean_total


# ============================================================================
# Main benchmark
# ============================================================================

def benchmark(args):
    ddp_enabled = setup_distributed()
    rank = get_rank()
    world_size = get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0)) if ddp_enabled else 0
    device = torch.device(f"cuda:{local_rank}")

    # Phase 1: CPU preprocessing benchmark (rank 0 only)
    if rank == 0:
        cpu_time_per_sample = benchmark_cpu_preprocessing(args)

    log("")
    log("=" * 70)
    log("END-TO-END TRAINING BENCHMARK (simulated nuplan pipeline)")
    log("=" * 70)
    log(f"Mode:              {'DDP ({} GPUs)'.format(world_size) if ddp_enabled else 'Single GPU'}")
    log(f"GPU:               {torch.cuda.get_device_name(device)}")
    log(f"Batch per GPU:     {args.batch_size}")
    log(f"Global batch:      {args.batch_size * world_size}")
    log(f"Workers per GPU:   {args.num_workers}")
    log(f"Simulate CPU:      {args.simulate_cpu}")
    log(f"Pin memory:        {args.pin_memory}")

    # Dataset
    dataset = NuplanSimDataset(
        num_samples=args.num_samples,
        simulate_cpu=args.simulate_cpu,
        data_dir=args.data_dir,
        num_agents=args.num_agents,
        num_polygons=args.num_polygons,
        history_steps=args.history_steps,
        future_steps=args.future_steps,
    )
    log(f"Dataset size:      {len(dataset)} samples")

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if ddp_enabled else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=collate_fn,
        sampler=sampler,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    log(f"Steps per epoch:   {len(loader)}")

    # Model
    model = PlanningModel(
        dim=args.dim,
        history_steps=args.history_steps,
        future_steps=args.future_steps,
        encoder_depth=args.encoder_depth,
        decoder_depth=args.decoder_depth,
        num_modes=args.num_modes,
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
    if sampler is not None:
        sampler.set_epoch(0)
    for i, batch_cpu in enumerate(loader):
        data = batch_to_device(batch_cpu, device)
        res = model(data)
        loss = compute_loss(res, data, collision_loss_fn, raw_model.num_modes, raw_model.radius, args.history_steps)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i >= args.warmup_steps - 1:
            break
    torch.cuda.synchronize(device)
    if ddp_enabled:
        dist.barrier()

    # Benchmark
    all_timings = {
        "dataload": [], "h2d": [], "forward": [],
        "backward": [], "optimizer": [], "step": [],
    }

    for epoch in range(args.num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch + 1)

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

            optimizer.step()
            optimizer.zero_grad()
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
                print(
                    f"  Epoch {epoch} Step {step_count:4d} | loss={loss.item():.4f} | "
                    f"total={total_step*1000:.1f}ms "
                    f"(dl={dl_time*1000:.1f} h2d={h2d_time*1000:.1f} "
                    f"fwd={fwd_time*1000:.1f} bwd={bwd_time*1000:.1f} "
                    f"opt={opt_time*1000:.1f}ms)", flush=True
                )

            if step_count >= args.max_steps_per_epoch:
                break

            t_dl_start = time.perf_counter()
            try:
                batch_cpu = next(data_iter)
            except StopIteration:
                break
            t_dl_end = time.perf_counter()

    if ddp_enabled:
        dist.barrier()

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
        print(f"  Simulate CPU:     {args.simulate_cpu}")
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
            print(
                f"  {label:20s} {mean*1000:8.1f}ms {med*1000:8.1f}ms "
                f"{std*1000:7.1f}ms {mn*1000:8.1f}ms {mx*1000:8.1f}ms {pct:7.1f}%"
            )

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

        # Comparison with baseline
        print()
        print("  [Analysis: CPU Preprocessing Impact]")
        dl_mean = stats(all_timings["dataload"])[0] * 1000
        compute_mean = (stats(all_timings["forward"])[0] + stats(all_timings["backward"])[0] +
                       stats(all_timings["optimizer"])[0]) * 1000
        print(f"    DataLoader wait (incl. CPU preproc):  {dl_mean:.1f} ms")
        print(f"    GPU compute (fwd+bwd+opt):            {compute_mean:.1f} ms")
        if dl_mean > compute_mean * 0.1:
            print(f"    ⚠ CPU preprocessing is a bottleneck ({dl_mean/(dl_mean+compute_mean)*100:.0f}% of pipeline)")
            print(f"    → Consider: more workers, feature caching, or reducing num_polygons")
        else:
            print(f"    ✓ CPU preprocessing is NOT a bottleneck (< 10% of pipeline)")

        print()

    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pluto nuplan-simulated training benchmark")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers per GPU")
    parser.add_argument("--num-samples", type=int, default=10000, help="Dataset size")
    parser.add_argument("--pin-memory", action="store_true", default=True)
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--simulate-cpu", action="store_true", default=True,
                        help="Simulate CPU preprocessing (map queries, cost maps, etc.)")
    parser.add_argument("--no-simulate-cpu", dest="simulate_cpu", action="store_false",
                        help="Disable CPU simulation (baseline comparison)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Pre-generated .pt data dir (used with --no-simulate-cpu)")
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
    benchmark(args)
