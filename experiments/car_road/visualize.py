"""
可视化脚本:
  mode=trajectories : 展示历史轨迹 vs 未来真值, 可选加载模型叠加预测轨迹
  mode=metrics      : 从 CSV 读取评估指标, 画分布图 (直方图 + 箱线图 + 汇总统计)

用法:
  # 画指标分布图 (从 CSV):
  python visualize.py --mode metrics

  # 仅展示历史/未来真值 (无需模型):
  python visualize.py --mode trajectories

  # 叠加模型预测轨迹:
  python visualize.py --mode trajectories \
      --model logs/models_xxx_car_road \
      --checkpoint 100
"""
import sys
import os
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.append('../../trajectron')

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="metrics",
                    choices=["trajectories", "metrics"],
                    help="可视化模式: trajectories(轨迹) 或 metrics(指标分布)")
parser.add_argument("--data", type=str, default="../processed/car_road_test_full.pkl",
                    help="测试集 pickle 文件路径")
parser.add_argument("--model", type=str, default=None,
                    help="模型目录路径 (不提供则只画真值)")
parser.add_argument("--checkpoint", type=int, default=100,
                    help="checkpoint 编号")
parser.add_argument("--ph", type=int, default=6,
                    help="预测时域 (未来步数)")
parser.add_argument("--num_scenes", type=int, default=3,
                    help="可视化的场景数量")
parser.add_argument("--num_samples", type=int, default=20,
                    help="模型预测采样数")
parser.add_argument("--output_path", type=str, default="results",
                    help="输出图片目录")
parser.add_argument("--csv_dir", type=str, default="results",
                    help="CSV 结果文件所在目录")
args = parser.parse_args()

os.makedirs(args.output_path, exist_ok=True)


# ============================================================
# MODE: metrics — 从 CSV 画指标分布图
# ============================================================
def run_metrics():
    import pandas as pd

    csv_files = sorted(glob.glob(os.path.join(args.csv_dir, '*.csv')))
    if not csv_files:
        print(f"错误: 在 {args.csv_dir} 下未找到 CSV 文件")
        sys.exit(1)

    # 读取并合并所有 CSV
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f, index_col=0)
        dfs.append(df)
        print(f"  已读取: {os.path.basename(f)}  ({len(df)} 条)")
    data = pd.concat(dfs, ignore_index=True)

    # 按 (metric, type) 分组
    groups = data.groupby(['metric', 'type'])
    metric_keys = list(groups.groups.keys())  # e.g. [('ade','ml'), ('fde','ml'), ('kde','full')]
    n = len(metric_keys)

    # ── 1) 汇总统计表 ──
    print("\n" + "=" * 60)
    print(f"{'Metric':<8} {'Type':<6} {'Mean':>10} {'Median':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 60)
    for (metric, mtype), grp in groups:
        vals = grp['value']
        print(f"{metric:<8} {mtype:<6} {vals.mean():>10.4f} {vals.median():>10.4f} "
              f"{vals.std():>10.4f} {vals.min():>10.4f} {vals.max():>10.4f}")
    print("=" * 60)

    # ── 2) 直方图 ──
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    colors = {'ade': '#2196F3', 'fde': '#FF9800', 'kde': '#4CAF50'}

    for ax, (key, grp) in zip(axes, groups):
        metric, mtype = key
        vals = grp['value'].values
        color = colors.get(metric, '#9C27B0')
        ax.hist(vals, bins=50, color=color, alpha=0.7, edgecolor='white')
        ax.axvline(np.mean(vals), color='red', linestyle='--', linewidth=1.5,
                   label=f'Mean={np.mean(vals):.4f}')
        ax.axvline(np.median(vals), color='black', linestyle=':', linewidth=1.5,
                   label=f'Median={np.median(vals):.4f}')
        ax.set_title(f'{metric.upper()} ({mtype})', fontsize=13)
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(args.output_path, 'metrics_histogram.png')
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")

    # ── 3) 箱线图 ──
    fig, ax = plt.subplots(figsize=(max(4, 2 * n), 5))
    box_data = []
    labels = []
    for key, grp in groups:
        metric, mtype = key
        box_data.append(grp['value'].values)
        labels.append(f'{metric.upper()}\n({mtype})')

    bp = ax.boxplot(box_data, labels=labels, patch_artist=True, showfliers=False)
    palette = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0']
    for patch, c in zip(bp['boxes'], palette[:n]):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_title('Metric Distributions (car_road, PH=6)', fontsize=13)
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    out = os.path.join(args.output_path, 'metrics_boxplot.png')
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")

    plt.close('all')
    print("\nDone!")


# ============================================================
# MODE: trajectories — 轨迹可视化 (需要 pkl 数据)
# ============================================================
def run_trajectories():
    import dill
    import json
    import torch

    # ── 加载数据 ──
    with open(args.data, 'rb') as f:
        env = dill.load(f, encoding='latin1')

    # ── 可选: 加载模型 ──
    eval_stg = None
    if args.model is not None:
        from model.model_registrar import ModelRegistrar
        from model.trajectron import Trajectron

        model_registrar = ModelRegistrar(args.model, 'cpu')
        model_registrar.load_models(args.checkpoint)
        with open(os.path.join(args.model, 'config.json'), 'r') as f:
            hyperparams = json.load(f)

        eval_stg = Trajectron(model_registrar, hyperparams, None, 'cpu')
        eval_stg.set_environment(env)
        eval_stg.set_annealing_params()

        if 'override_attention_radius' in hyperparams:
            for att in hyperparams['override_attention_radius']:
                nt1, nt2, radius = att.split(' ')
                env.attention_radius[(nt1, nt2)] = float(radius)

        print("模型已加载, 将叠加预测轨迹")
    else:
        print("未指定模型, 仅展示历史/未来真值")

    ph = args.ph

    for si, scene in enumerate(env.scenes[:args.num_scenes]):
        print(f"\nScene {si}: {scene.name}, total_timesteps={scene.timesteps}")

        obs_t = min(scene.timesteps - ph - 1, int(scene.timesteps * 0.6))
        obs_t = max(obs_t, 8)
        print(f"  观测时刻 t={obs_t}, 预测未来 {ph} 步")

        if eval_stg is not None:
            scene.calculate_scene_graph(env.attention_radius,
                                        hyperparams['edge_addition_filter'],
                                        hyperparams['edge_removal_filter'])
            with torch.no_grad():
                predictions = eval_stg.predict(scene,
                                               np.array([obs_t]),
                                               ph,
                                               num_samples=args.num_samples,
                                               min_future_timesteps=ph,
                                               z_mode=False,
                                               gmm_mode=False,
                                               full_dist=False)
        else:
            predictions = {}

        fig, ax = plt.subplots(figsize=(12, 12))

        node_count = 0
        for node in scene.nodes:
            t_start = node.first_timestep
            t_end = t_start + node.data.data.shape[0] - 1

            if t_start > obs_t or t_end < obs_t:
                continue

            xy = node.data.data[:, 0:2]
            obs_idx = obs_t - t_start
            future_end_idx = min(obs_idx + ph, xy.shape[0] - 1)

            hist = xy[:obs_idx + 1]
            future = xy[obs_idx:future_end_idx + 1]

            is_vehicle = node.type.name == 'VEHICLE'
            hist_color = '#2196F3' if is_vehicle else '#FF9800'
            future_color = '#F44336'

            if len(hist) > 1:
                ax.plot(hist[:, 0], hist[:, 1], '-', color=hist_color, alpha=0.6, linewidth=1.5)
            ax.plot(hist[-1, 0], hist[-1, 1], 'o', color=hist_color, markersize=6, zorder=5)

            if len(future) > 1:
                ax.plot(future[:, 0], future[:, 1], '--', color=future_color, alpha=0.8, linewidth=2)
                ax.plot(future[-1, 0], future[-1, 1], 'x', color=future_color, markersize=6, zorder=5)

            if obs_t in predictions and node in predictions[obs_t]:
                pred = predictions[obs_t][node][0]
                for s in range(pred.shape[0]):
                    traj = pred[s]
                    full_pred = np.vstack([hist[-1:], traj])
                    ax.plot(full_pred[:, 0], full_pred[:, 1], '-', color='#4CAF50',
                            alpha=0.3, linewidth=1)

            node_count += 1

        legend_elements = [
            Line2D([0], [0], color='#2196F3', linewidth=2, label='Vehicle history'),
            Line2D([0], [0], color='#FF9800', linewidth=2, label='Pedestrian history'),
            Line2D([0], [0], color='#F44336', linewidth=2, linestyle='--', label='Ground truth future'),
            Line2D([0], [0], marker='o', color='gray', markersize=6, linestyle='', label=f'Current pos (t={obs_t})'),
        ]
        if eval_stg is not None:
            legend_elements.append(
                Line2D([0], [0], color='#4CAF50', linewidth=2, alpha=0.5, label='Predicted trajectories'))

        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
        ax.set_title(f'{scene.name}  |  t={obs_t}  |  PH={ph}  |  {node_count} nodes', fontsize=13)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_file = os.path.join(args.output_path, f'scene_{si}_pred_vs_gt.png')
        plt.savefig(out_file, dpi=150)
        print(f"  Saved: {out_file}")

    plt.close('all')
    print("\nDone!")


# ── 入口 ──
if args.mode == 'metrics':
    run_metrics()
else:
    run_trajectories()
