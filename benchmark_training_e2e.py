#!/usr/bin/env python3
"""
Pluto end-to-end training benchmark: Disk I/O → CPU→GPU → Forward → Backward → Optimizer.

Simulates a realistic training pipeline with DataLoader reading .pt files from disk.

Usage:
  # 1. First generate data
  python benchmark_generate_data.py --num-samples 2000 --output-dir /tmp/pluto_bench_data

  # 2. Single GPU
  python benchmark_training_e2e.py --data-dir /tmp/pluto_bench_data --batch-size 32

  # 3. Multi-GPU DDP
  torchrun --nproc_per_node=8 benchmark_training_e2e.py --data-dir /tmp/pluto_bench_data --batch-size 32

Note: --batch-size is per-GPU.
"""

import math
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

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


# ---- Standalone PlanningModel ----

class PlanningModel(nn.Module):
    def __init__(
        self,
        dim=128, state_channel=6, polygon_channel=6, history_channel=9,
        history_steps=21, future_steps=80, encoder_depth=4, decoder_depth=4,
        drop_path=0.2, dropout=0.1, num_heads=8, num_modes=6,
        use_ego_history=False, state_attn_encoder=True, state_dropout=0.75,
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


# ---- Dataset ----

class PlutoBenchDataset(Dataset):
    def __init__(self, data_dir):
        self.files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir) if f.endswith(".pt")
        ])
        assert len(self.files) > 0, f"No .pt files found in {data_dir}"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(self.files[idx], map_location="cpu")


def collate_fn(batch):
    """Stack a list of single-sample dicts into a batched dict."""
    keys_nested = ["agent", "map", "static_objects", "reference_line"]
    keys_tensor = ["current_state", "cost_maps"]

    out = {}
    for key in keys_nested:
        out[key] = {
            k: torch.stack([sample[key][k] for sample in batch], dim=0)
            for k in batch[0][key].keys()
        }
    for key in keys_tensor:
        out[key] = torch.stack([sample[key] for sample in batch], dim=0)
    return out


def batch_to_device(data, device):
    """Recursively move all tensors in nested dict to device, returns new dict."""
    out = {}
    for k, v in data.items():
        if isinstance(v, dict):
            out[k] = batch_to_device(v, device)
        elif isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


# ---- Loss ----

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
    total = reg_loss + cls_loss + prediction_loss + col_loss
    return total


# ---- DDP helpers ----

def setup_distributed():
    if "RANK" not in os.environ:
        return False
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    return True

def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def is_distributed():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_distributed() else 0

def get_world_size():
    return dist.get_world_size() if is_distributed() else 1

def log(msg):
    if get_rank() == 0:
        print(msg, flush=True)


# ---- Benchmark ----

def benchmark(args):
    ddp_enabled = setup_distributed()
    rank = get_rank()
    world_size = get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0)) if ddp_enabled else 0
    device = torch.device(f"cuda:{local_rank}")

    log("=" * 70)
    log("PLUTO END-TO-END TRAINING BENCHMARK (Disk → CPU → GPU → Compute)")
    log("=" * 70)
    log(f"Mode:            {'DDP ({} GPUs)'.format(world_size) if ddp_enabled else 'Single GPU'}")
    log(f"GPU:             {torch.cuda.get_device_name(device)}")
    log(f"Batch per GPU:   {args.batch_size}")
    log(f"Global batch:    {args.batch_size * world_size}")
    log(f"DataLoader workers per GPU: {args.num_workers}")
    log(f"Pin memory:      {args.pin_memory}")
    log(f"Data dir:        {args.data_dir}")

    # Dataset & DataLoader
    dataset = PlutoBenchDataset(args.data_dir)
    log(f"Dataset size:    {len(dataset)} samples")

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
    log(f"Steps per epoch: {len(loader)}")

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
    log(f"Parameters:      {num_params / 1e6:.2f}M")

    if ddp_enabled:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        raw_model = model.module
    else:
        raw_model = model

    collision_loss_fn = ESDFCollisionLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    log("-" * 70)

    # Warmup
    log("Warming up (1 epoch) ...")
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

    # Benchmark epochs
    all_timings = {
        "dataload": [], "h2d": [], "forward": [],
        "backward": [], "optimizer": [], "step": [],
    }

    for epoch in range(args.num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch + 1)

        data_iter = iter(loader)
        step_count = 0

        # Pre-fetch first batch to separate dataloader wait from computation
        torch.cuda.synchronize(device)
        t_dl_start = time.perf_counter()
        try:
            batch_cpu = next(data_iter)
        except StopIteration:
            break
        t_dl_end = time.perf_counter()

        while True:
            # ---- [1] DataLoader wait time (already fetched) ----
            dl_time = t_dl_end - t_dl_start

            # ---- [2] CPU → GPU transfer ----
            torch.cuda.synchronize(device)
            t_h2d_start = time.perf_counter()
            data = batch_to_device(batch_cpu, device)
            torch.cuda.synchronize(device)
            t_h2d_end = time.perf_counter()
            h2d_time = t_h2d_end - t_h2d_start

            # ---- [3] Forward + Loss ----
            res = model(data)
            loss = compute_loss(res, data, collision_loss_fn, raw_model.num_modes, raw_model.radius, args.history_steps)
            torch.cuda.synchronize(device)
            t_fwd_end = time.perf_counter()
            fwd_time = t_fwd_end - t_h2d_end

            # ---- [4] Backward ----
            loss.backward()
            torch.cuda.synchronize(device)
            t_bwd_end = time.perf_counter()
            bwd_time = t_bwd_end - t_fwd_end

            # ---- [5] Optimizer ----
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

            if rank == 0 and (step_count % 20 == 0 or step_count == 1):
                print(
                    f"  Epoch {epoch} Step {step_count:4d} | loss={loss.item():.4f} | "
                    f"total={total_step*1000:.1f}ms "
                    f"(dl={dl_time*1000:.1f} h2d={h2d_time*1000:.1f} "
                    f"fwd={fwd_time*1000:.1f} bwd={bwd_time*1000:.1f} "
                    f"opt={opt_time*1000:.1f}ms)", flush=True
                )

            if step_count >= args.max_steps_per_epoch:
                break

            # ---- Pre-fetch next batch (overlaps with nothing - measures pure DL wait) ----
            t_dl_start = time.perf_counter()
            try:
                batch_cpu = next(data_iter)
            except StopIteration:
                break
            t_dl_end = time.perf_counter()

    # Sync
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

        # Throughput
        effective_step_time = mean_total
        global_throughput = (args.batch_size * world_size) / effective_step_time
        per_gpu_throughput = args.batch_size / effective_step_time

        print()
        print("  [Throughput]")
        print(f"    Global:    {global_throughput:.1f} samples/sec")
        print(f"    Per GPU:   {per_gpu_throughput:.1f} samples/sec")
        print(f"    Steps/sec: {1.0 / effective_step_time:.2f}")

        # Breakdown pie
        print()
        print("  [Time Breakdown]")
        for label, key in labels[1:]:  # skip total
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pluto end-to-end training benchmark")
    parser.add_argument("--data-dir", type=str, default="/tmp/pluto_bench_data")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size PER GPU")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers per GPU")
    parser.add_argument("--pin-memory", action="store_true", default=True)
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader prefetch factor")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--num-agents", type=int, default=20)
    parser.add_argument("--num-polygons", type=int, default=150)
    parser.add_argument("--history-steps", type=int, default=21)
    parser.add_argument("--future-steps", type=int, default=80)
    parser.add_argument("--encoder-depth", type=int, default=4)
    parser.add_argument("--decoder-depth", type=int, default=4)
    parser.add_argument("--num-modes", type=int, default=6)
    parser.add_argument("--num-epochs", type=int, default=2, help="Epochs to benchmark")
    parser.add_argument("--max-steps-per-epoch", type=int, default=100, help="Max steps per epoch")
    parser.add_argument("--warmup-steps", type=int, default=5)
    args = parser.parse_args()
    benchmark(args)
