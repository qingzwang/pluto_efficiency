#!/usr/bin/env python3
"""
Monitor GPU utilization during training WITH vs WITHOUT augmentation, then plot comparison.

Usage:
  python benchmark_gpu_util_augment.py --cache-dir /tmp/pluto_augment_cache --num-gpus 8
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


def start_monitor(log_path, interval_ms=200):
    """Start GPU monitoring in background, return process handle."""
    script = f"""
import subprocess, time
f = open("{log_path}", "w")
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
        time.sleep({interval_ms / 1000.0})
    except KeyboardInterrupt:
        break
    except:
        break
f.close()
"""
    proc = subprocess.Popen([sys.executable, "-c", script], stderr=subprocess.DEVNULL)
    time.sleep(0.5)
    return proc


def stop_monitor(proc):
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()


def parse_monitor_log(log_path):
    gpu_data = {}
    t0 = None
    with open(log_path) as f:
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
            gpu_data[gpu_id]["time"].append(ts - t0)
            gpu_data[gpu_id]["sm"].append(sm)
            gpu_data[gpu_id]["mem"].append(mem)
            gpu_data[gpu_id]["pwr"].append(pwr)
    return gpu_data


def run_training(args, augment, log_path):
    """Run training and monitor GPU."""
    label = "WITH" if augment else "WITHOUT"
    print(f"\n{'='*70}")
    print(f"  Running training {label} augmentation ...")
    print(f"{'='*70}")

    mon = start_monitor(log_path, args.sample_interval_ms)

    cmd = [
        "torchrun", f"--nproc_per_node={args.num_gpus}",
        "benchmark_augmentation.py", "--mode", "train",
        "--cache-dir", args.cache_dir,
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--num-epochs", str(args.num_epochs),
        "--max-steps-per-epoch", str(args.max_steps),
    ]
    if augment:
        cmd.append("--augment")
    else:
        cmd.append("--no-augment")

    t0 = time.time()
    subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0

    time.sleep(0.5)
    stop_monitor(mon)
    print(f"  Training {label} augmentation finished in {elapsed:.1f}s")

    return parse_monitor_log(log_path)


def plot_comparison(data_no_aug, data_with_aug, output_dir, num_gpus):
    """Generate comparison plots: timeline + summary bar."""
    os.makedirs(output_dir, exist_ok=True)
    colors = plt.cm.tab10(np.linspace(0, 1, num_gpus))
    skip = 5  # skip warmup samples

    # ====================================================================
    # Plot 1: Side-by-side SM utilization timeline
    # ====================================================================
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=False)

    for ax_idx, (gpu_data, label) in enumerate([
        (data_no_aug, "WITHOUT Augmentation (GPU batch=32)"),
        (data_with_aug, "WITH Augmentation (GPU batch=3x32=96)"),
    ]):
        ax = axes[ax_idx]
        for gpu_id in sorted(gpu_data.keys()):
            d = gpu_data[gpu_id]
            ax.plot(d["time"], d["sm"], color=colors[gpu_id], alpha=0.7,
                    linewidth=1.0, label=f"GPU {gpu_id}")

        # Average line
        min_len = min(len(gpu_data[g]["sm"]) for g in gpu_data)
        all_sm = np.array([gpu_data[g]["sm"][:min_len] for g in sorted(gpu_data.keys())])
        avg_sm = np.mean(all_sm, axis=0)
        t_avg = np.array(gpu_data[sorted(gpu_data.keys())[0]]["time"][:min_len])
        ax.plot(t_avg, avg_sm, color="black", linewidth=2.5, linestyle="--",
                alpha=0.8, label="Average")

        avg_val = np.mean(avg_sm[skip:]) if len(avg_sm) > skip else np.mean(avg_sm)
        ax.axhline(y=avg_val, color="red", linestyle=":", alpha=0.5)
        ax.text(ax.get_xlim()[0] if ax.get_xlim()[1] > 0 else 0, avg_val + 2,
                f"avg={avg_val:.0f}%", color="red", fontsize=11, fontweight="bold")

        ax.set_ylabel("SM Utilization (%)", fontsize=12)
        ax.set_ylim(-2, 105)
        ax.set_title(label, fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", ncol=5, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=100, color="gray", linestyle="--", alpha=0.2)

    axes[1].set_xlabel("Time (seconds)", fontsize=12)
    plt.tight_layout()
    path1 = os.path.join(output_dir, "gpu_util_augment_timeline.png")
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Timeline plot saved: {path1}")

    # ====================================================================
    # Plot 2: Summary bar chart (avg SM, avg Mem BW, avg Power)
    # ====================================================================
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for metric_idx, (metric, ylabel, title) in enumerate([
        ("sm", "SM Utilization (%)", "Average SM Utilization"),
        ("mem", "Memory BW Utilization (%)", "Average Memory BW Utilization"),
        ("pwr", "Power (W)", "Average Power Consumption"),
    ]):
        ax = axes[metric_idx]
        gpu_ids = sorted(data_no_aug.keys())

        vals_no = [np.mean(data_no_aug[g][metric][skip:]) for g in gpu_ids]
        vals_aug = [np.mean(data_with_aug[g][metric][skip:]) for g in gpu_ids]

        x = np.arange(len(gpu_ids))
        width = 0.35
        bars1 = ax.bar(x - width/2, vals_no, width, label="No Augmentation (bs=32)",
                        color="#4C72B0", alpha=0.85)
        bars2 = ax.bar(x + width/2, vals_aug, width, label="With Augmentation (3x32=96)",
                        color="#DD8452", alpha=0.85)

        ax.set_xlabel("GPU ID", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{g}" for g in gpu_ids])
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    path2 = os.path.join(output_dir, "gpu_util_augment_summary.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Summary plot saved: {path2}")

    # ====================================================================
    # Plot 3: Stacked area - time breakdown comparison
    # ====================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax_idx, (gpu_data, label, bs_label) in enumerate([
        (data_no_aug, "No Augmentation", "bs=32"),
        (data_with_aug, "With Augmentation", "bs=3x32=96"),
    ]):
        ax = axes[ax_idx]
        gpu_ids = sorted(gpu_data.keys())

        # Show per-GPU SM as heatmap-like stacked bars
        avg_sm_list = []
        avg_mem_list = []
        avg_pwr_list = []
        for g in gpu_ids:
            avg_sm_list.append(np.mean(gpu_data[g]["sm"][skip:]))
            avg_mem_list.append(np.mean(gpu_data[g]["mem"][skip:]))
            avg_pwr_list.append(np.mean(gpu_data[g]["pwr"][skip:]))

        overall_sm = np.mean(avg_sm_list)
        overall_mem = np.mean(avg_mem_list)
        overall_pwr = np.mean(avg_pwr_list)

        categories = ["SM Util (%)", "Mem BW (%)", f"Power (W)"]
        values = [overall_sm, overall_mem, overall_pwr]

        bar_colors = ["#4C72B0", "#55A868", "#DD8452"]
        bars = ax.barh(categories, values, color=bar_colors, alpha=0.85, height=0.5)

        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2.,
                    f'{val:.1f}', ha='left', va='center', fontsize=12, fontweight="bold")

        ax.set_xlim(0, max(max(values) * 1.2, 110))
        ax.set_title(f"{label}\n({bs_label}, 8 GPUs avg)", fontsize=13, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    path3 = os.path.join(output_dir, "gpu_util_augment_overview.png")
    plt.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Overview plot saved: {path3}")

    # ====================================================================
    # Print text summary
    # ====================================================================
    print(f"\n{'='*70}")
    print("  GPU UTILIZATION COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"  {'':20s} {'No Augment':>14s} {'With Augment':>14s} {'Ratio':>8s}")
    print(f"  {'-'*58}")

    gpu_ids = sorted(data_no_aug.keys())
    sm_no = np.mean([np.mean(data_no_aug[g]["sm"][skip:]) for g in gpu_ids])
    sm_aug = np.mean([np.mean(data_with_aug[g]["sm"][skip:]) for g in gpu_ids])
    mem_no = np.mean([np.mean(data_no_aug[g]["mem"][skip:]) for g in gpu_ids])
    mem_aug = np.mean([np.mean(data_with_aug[g]["mem"][skip:]) for g in gpu_ids])
    pwr_no = np.mean([np.mean(data_no_aug[g]["pwr"][skip:]) for g in gpu_ids])
    pwr_aug = np.mean([np.mean(data_with_aug[g]["pwr"][skip:]) for g in gpu_ids])

    print(f"  {'SM Utilization':20s} {sm_no:12.1f}% {sm_aug:12.1f}% {sm_aug/sm_no:7.2f}x")
    print(f"  {'Mem BW Utilization':20s} {mem_no:12.1f}% {mem_aug:12.1f}% {mem_aug/mem_no:7.2f}x")
    print(f"  {'Power':20s} {pwr_no:11.1f}W {pwr_aug:11.1f}W {pwr_aug/pwr_no:7.2f}x")
    print()

    # Per-GPU detail
    print(f"  Per-GPU SM Utilization (avg, excluding warmup):")
    print(f"  {'GPU':>5s} {'No Aug':>10s} {'With Aug':>10s}")
    for g in gpu_ids:
        sn = np.mean(data_no_aug[g]["sm"][skip:])
        sa = np.mean(data_with_aug[g]["sm"][skip:])
        print(f"  {g:>5d} {sn:9.1f}% {sa:9.1f}%")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=str, default="/tmp/pluto_augment_cache")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="/home/ubuntu/lotus/pluto/benchmark_results")
    parser.add_argument("--sample-interval-ms", type=int, default=200)
    args = parser.parse_args()

    log_no_aug = "/tmp/pluto_dmon_no_aug.log"
    log_with_aug = "/tmp/pluto_dmon_with_aug.log"

    # Run both scenarios
    data_no_aug = run_training(args, augment=False, log_path=log_no_aug)
    data_with_aug = run_training(args, augment=True, log_path=log_with_aug)

    if not data_no_aug or not data_with_aug:
        print("ERROR: No GPU data collected.")
        return

    plot_comparison(data_no_aug, data_with_aug, args.output_dir, args.num_gpus)


if __name__ == "__main__":
    main()
