#!/usr/bin/env python3
"""
Generate mock training data and save to disk for realistic I/O benchmarking.

Usage:
  python benchmark_generate_data.py --num-samples 2000 --output-dir /tmp/pluto_bench_data
"""

import os
import argparse
import torch


def generate_sample(
    num_agents=20,
    num_polygons=150,
    num_points_per_polygon=20,
    num_static_objects=10,
    num_ref_lines=6,
    num_ref_points=100,
    history_steps=21,
    future_steps=80,
    cost_map_size=200,
):
    T = history_steps + future_steps
    A, M, P = num_agents, num_polygons, num_points_per_polygon
    N_s, R, P_ref = num_static_objects, num_ref_lines, num_ref_points

    data = {
        "agent": {
            "position": torch.randn(A, T, 2),
            "heading": torch.randn(A, T),
            "velocity": torch.randn(A, T, 2),
            "shape": torch.rand(A, T, 2) + 0.5,
            "category": torch.randint(0, 4, (A,)),
            "valid_mask": torch.ones(A, T, dtype=torch.bool),
            "target": torch.randn(A, future_steps, 3),
        },
        "current_state": torch.randn(6),
        "map": {
            "polygon_center": torch.randn(M, 3),
            "polygon_type": torch.randint(0, 3, (M,)),
            "polygon_on_route": torch.randint(0, 2, (M,)),
            "polygon_tl_status": torch.randint(0, 4, (M,)),
            "polygon_has_speed_limit": torch.rand(M) > 0.5,
            "polygon_speed_limit": torch.rand(M) * 30,
            "point_position": torch.randn(M, 3, P, 2),
            "point_vector": torch.randn(M, 3, P, 2),
            "point_orientation": torch.randn(M, 3, P),
            "valid_mask": torch.ones(M, P, dtype=torch.bool),
        },
        "static_objects": {
            "position": torch.randn(N_s, 2),
            "heading": torch.randn(N_s),
            "shape": torch.rand(N_s, 2) + 0.5,
            "category": torch.randint(0, 4, (N_s,)),
            "valid_mask": torch.ones(N_s, dtype=torch.bool),
        },
        "reference_line": {
            "position": torch.randn(R, P_ref, 2),
            "vector": torch.randn(R, P_ref, 2),
            "orientation": torch.randn(R, P_ref),
            "valid_mask": torch.ones(R, P_ref, dtype=torch.bool),
            "future_projection": torch.randn(R, 8, 2),
        },
        "cost_maps": torch.rand(cost_map_size, cost_map_size, 1),
    }
    return data


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Generating {args.num_samples} samples to {args.output_dir} ...")

    for i in range(args.num_samples):
        sample = generate_sample(
            num_agents=args.num_agents,
            num_polygons=args.num_polygons,
            history_steps=args.history_steps,
            future_steps=args.future_steps,
        )
        path = os.path.join(args.output_dir, f"sample_{i:06d}.pt")
        torch.save(sample, path)

        if (i + 1) % 500 == 0 or i == 0:
            print(f"  {i+1}/{args.num_samples}")

    # Check size of one sample
    sample_path = os.path.join(args.output_dir, "sample_000000.pt")
    size_mb = os.path.getsize(sample_path) / 1e6
    total_gb = size_mb * args.num_samples / 1e3
    print(f"Done. Per-sample size: {size_mb:.2f} MB, Total: {total_gb:.2f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--output-dir", type=str, default="/tmp/pluto_bench_data")
    parser.add_argument("--num-agents", type=int, default=20)
    parser.add_argument("--num-polygons", type=int, default=150)
    parser.add_argument("--history-steps", type=int, default=21)
    parser.add_argument("--future-steps", type=int, default=80)
    args = parser.parse_args()
    main(args)
