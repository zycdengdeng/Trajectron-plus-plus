"""
可视化脚本: 展示历史轨迹 vs 未来真值, 可选加载模型叠加预测轨迹

用法:
  # 仅展示历史/未来真值 (无需模型):
  python visualize.py

  # 叠加模型预测轨迹:
  python visualize.py \
      --model logs/models_xxx_car_road \
      --checkpoint 100

  # 自定义参数:
  python visualize.py --ph 6 --num_scenes 5 --num_samples 20
"""
import sys
import os
import argparse
import dill
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.append('../../trajectron')

parser = argparse.ArgumentParser()
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
args = parser.parse_args()

# ── 加载数据 ──
with open(args.data, 'rb') as f:
    env = dill.load(f, encoding='latin1')

os.makedirs(args.output_path, exist_ok=True)

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

ph = args.ph  # prediction horizon

# ── 为每个场景可视化 ──
for si, scene in enumerate(env.scenes[:args.num_scenes]):
    print(f"\nScene {si}: {scene.name}, total_timesteps={scene.timesteps}")

    # 选择观测时刻: 场景中间偏后的位置, 确保有足够的历史和未来
    obs_t = min(scene.timesteps - ph - 1, int(scene.timesteps * 0.6))
    obs_t = max(obs_t, 8)  # 至少8步历史
    print(f"  观测时刻 t={obs_t}, 预测未来 {ph} 步")

    # 如果有模型, 先构建场景图并预测
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
        # 节点在场景中的时间范围
        t_start = node.first_timestep
        t_end = t_start + node.data.data.shape[0] - 1

        # 跳过观测时刻不存在的节点
        if t_start > obs_t or t_end < obs_t:
            continue

        xy = node.data.data[:, 0:2]
        # 节点局部索引
        obs_idx = obs_t - t_start  # 观测时刻在节点数据中的索引
        future_end_idx = min(obs_idx + ph, xy.shape[0] - 1)

        hist = xy[:obs_idx + 1]       # 历史 (含当前)
        future = xy[obs_idx:future_end_idx + 1]  # 未来真值 (从当前开始, 连续)

        is_vehicle = node.type.name == 'VEHICLE'
        hist_color = '#2196F3' if is_vehicle else '#FF9800'   # 蓝/橙
        future_color = '#F44336'  # 红色 = 真值未来

        # 画历史轨迹
        if len(hist) > 1:
            ax.plot(hist[:, 0], hist[:, 1], '-', color=hist_color, alpha=0.6, linewidth=1.5)
        # 当前位置 (大圆点)
        ax.plot(hist[-1, 0], hist[-1, 1], 'o', color=hist_color, markersize=6, zorder=5)

        # 画未来真值
        if len(future) > 1:
            ax.plot(future[:, 0], future[:, 1], '--', color=future_color, alpha=0.8, linewidth=2)
            ax.plot(future[-1, 0], future[-1, 1], 'x', color=future_color, markersize=6, zorder=5)

        # 画模型预测轨迹
        if obs_t in predictions and node in predictions[obs_t]:
            pred = predictions[obs_t][node]  # shape: (1, num_samples, ph, 2)
            pred = pred[0]  # (num_samples, ph, 2)
            for s in range(pred.shape[0]):
                traj = pred[s]  # (ph, 2)
                # 预测轨迹从当前位置开始画
                full_pred = np.vstack([hist[-1:], traj])
                ax.plot(full_pred[:, 0], full_pred[:, 1], '-', color='#4CAF50',
                        alpha=0.3, linewidth=1)

        node_count += 1

    # ── 图例 ──
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
