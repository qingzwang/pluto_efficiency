#!/usr/bin/env python3
"""
Monitor per-GPU SM utilization during DDP training and plot the results.

This script:
1. Launches nvidia-smi dmon in background to sample GPU utilization
2. Runs DDP training via torchrun
3. Parses the monitoring data and plots per-GPU utilization timeline

Usage:
  python benchmark_gpu_utilization.py --data-dir /tmp/pluto_bench_data_20k --batch-size 32 --num-gpus 8
"""

import argparse
import os
import signal
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def parse_monitor_log(log_path):
    """Parse GPU monitoring CSV into per-GPU time series."""
    gpu_data = {}  # gpu_id -> {"time": [], "sm": [], "mem": [], "pwr": []}
    t0 = None

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("timestamp"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            try:
                ts = float(parts[0])
                gpu_id = int(parts[1])
                sm = int(parts[2])
                mem = int(parts[3])
                pwr = float(parts[4])
            except (ValueError, IndexError):
                continue

            if t0 is None:
                t0 = ts

            if gpu_id not in gpu_data:
                gpu_data[gpu_id] = {"time": [], "sm": [], "mem": [], "pwr": []}

            gpu_data[gpu_id]["time"].append((ts - t0) * 1000)  # ms
            gpu_data[gpu_id]["sm"].append(sm)
            gpu_data[gpu_id]["mem"].append(mem)
            gpu_data[gpu_id]["pwr"].append(pwr)

    return gpu_data


def plot_utilization(gpu_data, output_path, num_gpus):
    """Plot per-GPU SM and memory utilization timeline."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)

    colors = plt.cm.tab10(np.linspace(0, 1, num_gpus))

    # -- SM Utilization --
    ax = axes[0]
    for gpu_id in sorted(gpu_data.keys()):
        d = gpu_data[gpu_id]
        t_sec = np.array(d["time"]) / 1000.0
        ax.plot(t_sec, d["sm"], color=colors[gpu_id], alpha=0.8, linewidth=1.0,
                label=f"GPU {gpu_id}")
    ax.set_ylabel("SM Utilization (%)", fontsize=12)
    ax.set_ylim(-2, 105)
    ax.set_title("Per-GPU SM Utilization During DDP Training", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", ncol=4, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.3)

    # Average SM line
    all_sm = []
    min_len = min(len(gpu_data[g]["sm"]) for g in gpu_data)
    for g in sorted(gpu_data.keys()):
        all_sm.append(gpu_data[g]["sm"][:min_len])
    avg_sm = np.mean(all_sm, axis=0)
    t_avg = np.array(gpu_data[sorted(gpu_data.keys())[0]]["time"][:min_len]) / 1000.0
    ax.plot(t_avg, avg_sm, color="black", linewidth=2, linestyle="--", alpha=0.6, label="Average")
    ax.legend(loc="upper right", ncol=5, fontsize=9)

    # -- Memory Bandwidth Utilization --
    ax = axes[1]
    for gpu_id in sorted(gpu_data.keys()):
        d = gpu_data[gpu_id]
        t_sec = np.array(d["time"]) / 1000.0
        ax.plot(t_sec, d["mem"], color=colors[gpu_id], alpha=0.8, linewidth=1.0,
                label=f"GPU {gpu_id}")
    ax.set_ylabel("Memory BW Utilization (%)", fontsize=12)
    ax.set_ylim(-2, 105)
    ax.set_title("Per-GPU Memory Bandwidth Utilization", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", ncol=4, fontsize=9)
    ax.grid(True, alpha=0.3)

    # -- Power --
    ax = axes[2]
    for gpu_id in sorted(gpu_data.keys()):
        d = gpu_data[gpu_id]
        t_sec = np.array(d["time"]) / 1000.0
        ax.plot(t_sec, d["pwr"], color=colors[gpu_id], alpha=0.8, linewidth=1.0,
                label=f"GPU {gpu_id}")
    ax.set_ylabel("Power (W)", fontsize=12)
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_title("Per-GPU Power Consumption", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", ncol=4, fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {output_path}")


def plot_summary_bar(gpu_data, output_path, num_gpus):
    """Plot a summary bar chart of average SM utilization per GPU."""
    gpu_ids = sorted(gpu_data.keys())

    # Skip first few samples (warmup ramp-up)
    skip = 5
    avg_sm = []
    avg_mem = []
    for g in gpu_ids:
        sm = gpu_data[g]["sm"][skip:]
        mem = gpu_data[g]["mem"][skip:]
        avg_sm.append(np.mean(sm) if sm else 0)
        avg_mem.append(np.mean(mem) if mem else 0)

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    x = np.arange(len(gpu_ids))
    width = 0.35

    bars1 = ax.bar(x - width/2, avg_sm, width, label="SM Utilization", color="#4C72B0", alpha=0.85)
    bars2 = ax.bar(x + width/2, avg_mem, width, label="Memory BW Utilization", color="#DD8452", alpha=0.85)

    ax.set_xlabel("GPU ID", fontsize=12)
    ax.set_ylabel("Utilization (%)", fontsize=12)
    ax.set_title("Average GPU Utilization During Training (excluding warmup)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"GPU {g}" for g in gpu_ids])
    ax.set_ylim(0, 110)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Summary plot saved to: {output_path}")


def main(args):
    dmon_log = "/tmp/pluto_dmon.log"
    timeline_plot = os.path.join(args.output_dir, "gpu_utilization_timeline.png")
    summary_plot = os.path.join(args.output_dir, "gpu_utilization_summary.png")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Start nvidia-smi sampling in background (200ms interval via loop)
    sample_interval_ms = args.sample_interval_ms
    print(f"Starting GPU monitoring ({sample_interval_ms}ms interval) ...")
    monitor_script = f"""
import subprocess, time, sys
f = open("{dmon_log}", "w")
f.write("timestamp,gpu,sm,mem,power\\n")
while True:
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,utilization.memory,power.draw",
            "--format=csv,noheader,nounits"
        ], text=True)
        t = time.time()
        for line in out.strip().split("\\n"):
            f.write(f"{{t}},{{line.strip()}}\\n")
        f.flush()
        time.sleep({sample_interval_ms / 1000.0})
    except KeyboardInterrupt:
        break
    except:
        break
f.close()
"""
    monitor_proc = subprocess.Popen(
        [sys.executable, "-c", monitor_script],
        stderr=subprocess.DEVNULL,
    )
    time.sleep(0.5)  # let monitor start

    # 2. Run training
    print(f"Launching DDP training ({args.num_gpus} GPUs, bs={args.batch_size}) ...")
    train_cmd = [
        "torchrun", f"--nproc_per_node={args.num_gpus}",
        "benchmark_training_e2e.py",
        "--data-dir", args.data_dir,
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--num-epochs", str(args.num_epochs),
        "--max-steps-per-epoch", str(args.max_steps),
        "--pin-memory",
    ]

    start_time = time.time()
    train_proc = subprocess.run(train_cmd, capture_output=False)
    elapsed = time.time() - start_time

    # 3. Stop monitor
    time.sleep(0.5)
    monitor_proc.send_signal(signal.SIGINT)
    try:
        monitor_proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        monitor_proc.kill()
    print(f"Training finished in {elapsed:.1f}s. Monitoring stopped.")

    # 4. Parse and plot
    print("Parsing monitoring data ...")
    gpu_data = parse_monitor_log(dmon_log)

    if not gpu_data:
        print("ERROR: No GPU data collected. Check nvidia-smi dmon output.")
        return

    print(f"Collected data for {len(gpu_data)} GPUs, "
          f"{min(len(d['sm']) for d in gpu_data.values())} samples each")

    # Print summary stats
    print("\n--- GPU Utilization Summary (excluding first 5 samples) ---")
    print(f"{'GPU':>5s} {'SM avg':>8s} {'SM med':>8s} {'SM min':>8s} {'SM max':>8s} "
          f"{'Mem avg':>8s} {'Pwr avg':>8s}")
    for g in sorted(gpu_data.keys()):
        sm = gpu_data[g]["sm"][5:]
        mem = gpu_data[g]["mem"][5:]
        pwr = gpu_data[g]["pwr"][5:]
        if sm:
            print(f"  {g:>3d} {np.mean(sm):7.1f}% {np.median(sm):7.1f}% "
                  f"{np.min(sm):7.1f}% {np.max(sm):7.1f}% "
                  f"{np.mean(mem):7.1f}% {np.mean(pwr):7.1f}W")

    # 5. Plot
    plot_utilization(gpu_data, timeline_plot, args.num_gpus)
    plot_summary_bar(gpu_data, summary_plot, args.num_gpus)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="/tmp/pluto_bench_data_20k")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--output-dir", type=str, default="/home/ubuntu/lotus/pluto/benchmark_results")
    parser.add_argument("--sample-interval-ms", type=int, default=200, help="GPU sampling interval in ms")
    args = parser.parse_args()
    main(args)
