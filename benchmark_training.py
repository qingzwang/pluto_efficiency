#!/usr/bin/env python3
"""
Pluto model training speed benchmark with simulated data.
No nuplan dependency required - standalone model + fake data.
"""

import math
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import model sub-modules (they don't depend on nuplan)
import sys
sys.path.insert(0, "/home/ubuntu/lotus/pluto")

from src.models.pluto.layers.fourier_embedding import FourierEmbedding
from src.models.pluto.layers.transformer import TransformerEncoderLayer
from src.models.pluto.layers.mlp_layer import MLPLayer
from src.models.pluto.modules.agent_encoder import AgentEncoder
from src.models.pluto.modules.agent_predictor import AgentPredictor
from src.models.pluto.modules.map_encoder import MapEncoder
from src.models.pluto.modules.static_objects_encoder import StaticObjectsEncoder
from src.models.pluto.modules.planning_decoder import PlanningDecoder
from src.models.pluto.loss.esdf_collision_loss import ESDFCollisionLoss


# ---- Standalone PlanningModel (no nuplan dependency) ----

class PlanningModel(nn.Module):
    def __init__(
        self,
        dim=128,
        state_channel=6,
        polygon_channel=6,
        history_channel=9,
        history_steps=21,
        future_steps=80,
        encoder_depth=4,
        decoder_depth=4,
        drop_path=0.2,
        dropout=0.1,
        num_heads=8,
        num_modes=6,
        use_ego_history=False,
        state_attn_encoder=True,
        state_dropout=0.75,
        radius=100,
    ):
        super().__init__()
        self.dim = dim
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.radius = radius

        self.pos_emb = FourierEmbedding(3, dim, 64)

        self.agent_encoder = AgentEncoder(
            state_channel=state_channel,
            history_channel=history_channel,
            dim=dim,
            hist_steps=history_steps,
            drop_path=drop_path,
            use_ego_history=use_ego_history,
            state_attn_encoder=state_attn_encoder,
            state_dropout=state_dropout,
        )

        self.map_encoder = MapEncoder(
            dim=dim, polygon_channel=polygon_channel, use_lane_boundary=True
        )

        self.static_objects_encoder = StaticObjectsEncoder(dim=dim)

        self.encoder_blocks = nn.ModuleList(
            TransformerEncoderLayer(dim=dim, num_heads=num_heads, drop_path=dp)
            for dp in [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]
        )
        self.norm = nn.LayerNorm(dim)

        self.agent_predictor = AgentPredictor(dim=dim, future_steps=future_steps)
        self.planning_decoder = PlanningDecoder(
            num_mode=num_modes,
            decoder_depth=decoder_depth,
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=4,
            dropout=dropout,
            cat_x=False,
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
        agent_mask = data["agent"]["valid_mask"][:, :, : self.history_steps]
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

        return {
            "trajectory": trajectory,
            "probability": probability,
            "prediction": prediction,
        }


# ---- Fake data generator ----

def generate_fake_batch(
    batch_size,
    num_agents=20,
    num_polygons=150,
    num_points_per_polygon=20,
    num_static_objects=10,
    num_ref_lines=6,
    num_ref_points=100,
    history_steps=21,
    future_steps=80,
    cost_map_size=200,
    device="cuda",
):
    T = history_steps + future_steps
    bs = batch_size
    A = num_agents
    M = num_polygons
    P = num_points_per_polygon
    N_s = num_static_objects
    R = num_ref_lines
    P_ref = num_ref_points

    data = {}

    # Agent data
    data["agent"] = {
        "position": torch.randn(bs, A, T, 2, device=device),
        "heading": torch.randn(bs, A, T, device=device),
        "velocity": torch.randn(bs, A, T, 2, device=device),
        "shape": torch.rand(bs, A, T, 2, device=device) + 0.5,
        "category": torch.randint(0, 4, (bs, A), device=device),
        "valid_mask": torch.ones(bs, A, T, dtype=torch.bool, device=device),
        "target": torch.randn(bs, A, future_steps, 3, device=device),
    }

    # Current ego state
    data["current_state"] = torch.randn(bs, 6, device=device)

    # Map data
    data["map"] = {
        "polygon_center": torch.randn(bs, M, 3, device=device),
        "polygon_type": torch.randint(0, 3, (bs, M), device=device),
        "polygon_on_route": torch.randint(0, 2, (bs, M), device=device),
        "polygon_tl_status": torch.randint(0, 4, (bs, M), device=device),
        "polygon_has_speed_limit": torch.rand(bs, M, device=device) > 0.5,
        "polygon_speed_limit": torch.rand(bs, M, device=device) * 30,
        "point_position": torch.randn(bs, M, 3, P, 2, device=device),
        "point_vector": torch.randn(bs, M, 3, P, 2, device=device),
        "point_orientation": torch.randn(bs, M, 3, P, device=device),
        "valid_mask": torch.ones(bs, M, P, dtype=torch.bool, device=device),
    }

    # Static objects
    data["static_objects"] = {
        "position": torch.randn(bs, N_s, 2, device=device),
        "heading": torch.randn(bs, N_s, device=device),
        "shape": torch.rand(bs, N_s, 2, device=device) + 0.5,
        "category": torch.randint(0, 4, (bs, N_s), device=device),
        "valid_mask": torch.ones(bs, N_s, dtype=torch.bool, device=device),
    }

    # Reference lines
    data["reference_line"] = {
        "position": torch.randn(bs, R, P_ref, 2, device=device),
        "vector": torch.randn(bs, R, P_ref, 2, device=device),
        "orientation": torch.randn(bs, R, P_ref, device=device),
        "valid_mask": torch.ones(bs, R, P_ref, dtype=torch.bool, device=device),
        "future_projection": torch.randn(bs, R, 8, 2, device=device),
    }

    # Cost maps for collision loss
    data["cost_maps"] = torch.rand(bs, cost_map_size, cost_map_size, 1, device=device)

    return data


# ---- Simplified loss (mirrors pluto_trainer) ----

def compute_loss(res, data, collision_loss_fn, num_modes, radius, history_steps):
    bs = res["prediction"].shape[0]
    T = res["prediction"].shape[2]
    valid_mask = data["agent"]["valid_mask"][:, :, -T:]

    # --- prediction loss ---
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

    # --- planning loss ---
    trajectory = res["trajectory"]  # (bs, R, M, T, 4+2)
    probability = res["probability"]  # (bs, R, M)

    ego_valid = valid_mask[:, 0]
    ego_target = target[:, 0]

    num_valid_points = ego_valid.sum(-1)
    endpoint_index = (num_valid_points / 10).long().clamp_(min=0, max=7)

    r_padding_mask = ~data["reference_line"]["valid_mask"].any(-1)
    future_projection = data["reference_line"]["future_projection"][
        torch.arange(bs), :, endpoint_index
    ]

    mode_interval = radius / num_modes
    target_r_index = torch.argmin(
        future_projection[..., 1] + 1e6 * r_padding_mask, dim=-1
    )
    target_m_index = (
        future_projection[torch.arange(bs), target_r_index, 0] / mode_interval
    ).long().clamp_(min=0, max=num_modes - 1)

    target_label = torch.zeros_like(probability)
    target_label[torch.arange(bs), target_r_index, target_m_index] = 1

    best_traj = trajectory[torch.arange(bs), target_r_index, target_m_index]

    # collision loss
    col_loss = collision_loss_fn(best_traj[..., :4], data["cost_maps"][:, :, :, 0].float())

    # regression loss
    reg_loss = F.smooth_l1_loss(best_traj, ego_target[..., :best_traj.shape[-1]], reduction="none").sum(-1)
    reg_loss = (reg_loss * ego_valid).sum() / (ego_valid.sum() + 1e-6)

    # classification loss
    probability_masked = probability.clone()
    probability_masked.masked_fill_(r_padding_mask.unsqueeze(-1), -1e6)
    cls_loss = F.cross_entropy(
        probability_masked.reshape(bs, -1), target_label.reshape(bs, -1).detach()
    )

    total = reg_loss + cls_loss + prediction_loss + col_loss
    return total, {
        "reg": reg_loss.item(),
        "cls": cls_loss.item(),
        "pred": prediction_loss.item(),
        "col": col_loss.item(),
    }


# ---- Benchmark ----

def benchmark(args):
    device = torch.device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")

    model = PlanningModel(
        dim=args.dim,
        history_steps=args.history_steps,
        future_steps=args.future_steps,
        encoder_depth=args.encoder_depth,
        decoder_depth=args.decoder_depth,
        num_modes=args.num_modes,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {num_params / 1e6:.2f}M (trainable: {trainable_params / 1e6:.2f}M)")

    collision_loss_fn = ESDFCollisionLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    use_amp = args.amp and device.type == "cuda"
    if use_amp:
        print("WARNING: AMP may fail due to natten (NeighborhoodAttention1D) not supporting fp16.")
        print("         If it crashes, run without --amp.")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    print(f"AMP (mixed precision): {use_amp}")
    print(f"Batch size: {args.batch_size}, Agents: {args.num_agents}, Polygons: {args.num_polygons}")
    print(f"History steps: {args.history_steps}, Future steps: {args.future_steps}")
    print("-" * 60)

    # Warmup
    print("Warming up...")
    model.train()
    for _ in range(args.warmup_steps):
        data = generate_fake_batch(
            batch_size=args.batch_size,
            num_agents=args.num_agents,
            num_polygons=args.num_polygons,
            history_steps=args.history_steps,
            future_steps=args.future_steps,
            device=device,
        )
        with torch.cuda.amp.autocast(enabled=use_amp):
            res = model(data)
            loss, _ = compute_loss(res, data, collision_loss_fn, args.num_modes, model.radius, args.history_steps)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # Benchmark
    print(f"Benchmarking {args.num_steps} steps...")
    step_times = []
    fwd_times = []
    bwd_times = []
    data_times = []

    for step in range(args.num_steps):
        # Data generation timing
        t_data_start = time.perf_counter()
        data = generate_fake_batch(
            batch_size=args.batch_size,
            num_agents=args.num_agents,
            num_polygons=args.num_polygons,
            history_steps=args.history_steps,
            future_steps=args.future_steps,
            device=device,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_data_end = time.perf_counter()
        data_times.append(t_data_end - t_data_start)

        # Forward
        t_fwd_start = time.perf_counter()
        with torch.cuda.amp.autocast(enabled=use_amp):
            res = model(data)
            loss, loss_dict = compute_loss(res, data, collision_loss_fn, args.num_modes, model.radius, args.history_steps)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_fwd_end = time.perf_counter()
        fwd_times.append(t_fwd_end - t_fwd_start)

        # Backward + optimizer
        t_bwd_start = time.perf_counter()
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_bwd_end = time.perf_counter()
        bwd_times.append(t_bwd_end - t_bwd_start)

        step_times.append(data_times[-1] + fwd_times[-1] + bwd_times[-1])

        if (step + 1) % 10 == 0 or step == 0:
            print(
                f"  Step {step+1:4d} | loss={loss.item():.4f} | "
                f"total={step_times[-1]*1000:.1f}ms "
                f"(data={data_times[-1]*1000:.1f} fwd={fwd_times[-1]*1000:.1f} bwd={bwd_times[-1]*1000:.1f}ms)"
            )

    # Summary
    print("=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    def stats(times):
        t = torch.tensor(times)
        return t.mean().item(), t.std().item(), t.median().item(), t.min().item(), t.max().item()

    for name, times in [("Total step", step_times), ("Data gen", data_times), ("Forward+Loss", fwd_times), ("Backward+Optim", bwd_times)]:
        mean, std, med, mn, mx = stats(times)
        print(f"  {name:18s}: mean={mean*1000:7.1f}ms | median={med*1000:7.1f}ms | "
              f"std={std*1000:5.1f}ms | min={mn*1000:7.1f}ms | max={mx*1000:7.1f}ms")

    throughput = args.batch_size / (sum(step_times) / len(step_times))
    print(f"\n  Throughput: {throughput:.1f} samples/sec")
    print(f"  Steps/sec:  {1.0 / (sum(step_times) / len(step_times)):.2f}")

    if device.type == "cuda":
        print(f"\n  GPU memory allocated: {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB")
        print(f"  GPU memory reserved:  {torch.cuda.max_memory_reserved(device) / 1e9:.2f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pluto training speed benchmark")
    parser.add_argument("--device", default="cuda:0", help="Device (cuda:0, cpu, etc.)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--num-agents", type=int, default=20)
    parser.add_argument("--num-polygons", type=int, default=150)
    parser.add_argument("--history-steps", type=int, default=21)
    parser.add_argument("--future-steps", type=int, default=80)
    parser.add_argument("--encoder-depth", type=int, default=4)
    parser.add_argument("--decoder-depth", type=int, default=4)
    parser.add_argument("--num-modes", type=int, default=6)
    parser.add_argument("--num-steps", type=int, default=50, help="Benchmark steps")
    parser.add_argument("--warmup-steps", type=int, default=5, help="Warmup steps")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision (AMP)")
    args = parser.parse_args()
    benchmark(args)
