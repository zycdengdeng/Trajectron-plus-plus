"""
BEV (Bird's Eye View) 可视化脚本

绘制路侧视角的鸟瞰图:
- 带方向的车辆矩形包围框 (length × width + heading)
- 行人圆形标记
- 历史轨迹 + 未来真值 + 模型预测
- 多帧动画或关键帧快照

用法:
    # 静态 BEV 快照
    python bev_visualize.py \
        --data ../processed/car_road_test_full.pkl \
        --model logs/models_05_Mar_2026_16_25_06_car_road \
        --checkpoint 80 \
        --output_path results/bev

    # 指定场景和时刻
    python bev_visualize.py \
        --data ../processed/car_road_test_full.pkl \
        --scene_idx 0 --timestep 30 \
        --output_path results/bev
"""
import sys
import os
import dill
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.collections import PatchCollection

sys.path.append("../../trajectron")

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="../processed/car_road_test_full.pkl")
parser.add_argument("--model", type=str, default=None)
parser.add_argument("--checkpoint", type=int, default=80)
parser.add_argument("--ph", type=int, default=6)
parser.add_argument("--num_samples", type=int, default=20, help="模型预测采样数")
parser.add_argument("--num_scenes", type=int, default=5, help="可视化场景数")
parser.add_argument("--scene_idx", type=int, default=None, help="指定场景索引")
parser.add_argument("--timestep", type=int, default=None, help="指定观测时刻")
parser.add_argument("--history_len", type=int, default=8, help="显示历史轨迹长度")
parser.add_argument("--output_path", type=str, default="results/bev")
parser.add_argument("--zoom", type=float, default=None,
                    help="缩放半径 (m)，聚焦特定区域")
args = parser.parse_args()


def draw_vehicle_box(ax, x, y, heading, length, width, color='#2196F3',
                     alpha=0.6, linewidth=1.5, fill=False, label=None):
    """绘制带方向的车辆矩形"""
    if length is None or width is None:
        length = 4.5
        width = 1.8

    # 矩形的左下角 (相对车辆中心)
    rect = patches.FancyBboxPatch(
        (-length / 2, -width / 2), length, width,
        boxstyle="round,pad=0.1",
        linewidth=linewidth, edgecolor=color, facecolor=color if fill else 'none',
        alpha=alpha
    )

    # 旋转和平移
    t = matplotlib.transforms.Affine2D().rotate(heading).translate(x, y) + ax.transData
    rect.set_transform(t)
    ax.add_patch(rect)

    # 朝向箭头
    dx = length / 2 * np.cos(heading)
    dy = length / 2 * np.sin(heading)
    ax.arrow(x, y, dx * 0.6, dy * 0.6, head_width=0.5, head_length=0.3,
             fc=color, ec=color, alpha=alpha, zorder=10)

    return rect


def draw_pedestrian(ax, x, y, color='#FF9800', radius=0.5, alpha=0.7):
    """绘制行人圆形标记"""
    circle = plt.Circle((x, y), radius, color=color, alpha=alpha, zorder=10)
    ax.add_patch(circle)
    return circle


def draw_trajectory(ax, traj, color, linestyle='-', alpha=0.7, linewidth=2.0,
                    marker=None, markersize=3, label=None):
    """绘制轨迹线"""
    if len(traj) < 2:
        return
    ax.plot(traj[:, 0], traj[:, 1], linestyle=linestyle, color=color,
            alpha=alpha, linewidth=linewidth, marker=marker,
            markersize=markersize, label=label, zorder=5)


def draw_prediction_fan(ax, current_pos, pred_trajs, color='#4CAF50', alpha=0.15):
    """绘制预测轨迹扇形区域"""
    n_samples = pred_trajs.shape[0]
    for s in range(n_samples):
        full_traj = np.vstack([current_pos.reshape(1, 2), pred_trajs[s]])
        ax.plot(full_traj[:, 0], full_traj[:, 1], '-', color=color,
                alpha=alpha, linewidth=0.8, zorder=3)

    # 终点散布
    endpoints = pred_trajs[:, -1, :]
    ax.scatter(endpoints[:, 0], endpoints[:, 1], c=color, s=8,
               alpha=0.3, zorder=4, edgecolors='none')


def render_bev(scene, obs_t, ph, history_len, predictions=None,
               zoom=None, output_file=None):
    """渲染单帧 BEV"""
    fig, ax = plt.subplots(figsize=(14, 14))

    vehicle_count = 0
    ped_count = 0

    all_x, all_y = [], []

    for node in scene.nodes:
        t_start = node.first_timestep
        t_end = t_start + node.data.data.shape[0] - 1

        if t_start > obs_t or t_end < obs_t:
            continue

        xy = node.data.data[:, 0:2]
        local_obs = obs_t - t_start

        # 当前位置
        cur_x, cur_y = xy[local_obs]
        all_x.append(cur_x)
        all_y.append(cur_y)

        is_vehicle = node.type.name == 'VEHICLE'

        # 历史轨迹
        hist_start = max(0, local_obs - history_len)
        hist = xy[hist_start:local_obs + 1]

        # 未来真值
        future_end = min(local_obs + ph, xy.shape[0] - 1)
        future = xy[local_obs:future_end + 1]

        if is_vehicle:
            # 获取 heading
            heading_data = node.data[:, ('heading', '°')]
            heading = float(heading_data.flatten()[local_obs])

            # 绘制车辆包围框
            draw_vehicle_box(ax, cur_x, cur_y, heading,
                             node.length, node.width,
                             color='#1565C0', alpha=0.8, fill=True)

            # 历史轨迹
            draw_trajectory(ax, hist, color='#64B5F6', linestyle='-',
                            alpha=0.5, linewidth=1.5, marker='.', markersize=2)
            vehicle_count += 1
        else:
            draw_pedestrian(ax, cur_x, cur_y, color='#E65100', radius=0.4)

            draw_trajectory(ax, hist, color='#FFB74D', linestyle='-',
                            alpha=0.5, linewidth=1.5, marker='.', markersize=2)
            ped_count += 1

        # 未来真值
        if len(future) > 1:
            draw_trajectory(ax, future, color='#F44336', linestyle='--',
                            alpha=0.7, linewidth=2.0)
            ax.plot(future[-1, 0], future[-1, 1], 'x', color='#F44336',
                    markersize=8, markeredgewidth=2, zorder=10)

        # 模型预测
        if predictions is not None and obs_t in predictions and node in predictions[obs_t]:
            pred = predictions[obs_t][node][0]  # shape: [num_samples, ph, 2]
            draw_prediction_fan(ax, xy[local_obs], pred, color='#4CAF50', alpha=0.2)

    # 布局和标注
    if zoom is not None and len(all_x) > 0:
        cx, cy = np.mean(all_x), np.mean(all_y)
        ax.set_xlim(cx - zoom, cx + zoom)
        ax.set_ylim(cy - zoom, cy + zoom)
    elif len(all_x) > 0:
        margin = 15
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_facecolor('#F5F5F5')

    # 图例
    legend_elements = [
        patches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.1",
                                fc='#1565C0', ec='#1565C0', alpha=0.8,
                                label=f'Vehicle ({vehicle_count})'),
        plt.Circle((0, 0), 1, color='#E65100', alpha=0.7,
                   label=f'Pedestrian ({ped_count})'),
        Line2D([0], [0], color='#64B5F6', linewidth=2, label='History'),
        Line2D([0], [0], color='#F44336', linewidth=2, linestyle='--',
               label='Ground truth'),
    ]
    if predictions is not None:
        legend_elements.append(
            Line2D([0], [0], color='#4CAF50', linewidth=2, alpha=0.5,
                   label='Predictions'))

    ax.legend(handles=legend_elements, loc='upper left', fontsize=10,
              framealpha=0.9)

    dt = scene.dt
    title = (f'BEV: {scene.name}  |  t={obs_t} ({obs_t*dt:.1f}s)  |  '
             f'PH={ph} ({ph*dt:.1f}s)  |  '
             f'{vehicle_count}V + {ped_count}P')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)

    # 添加比例尺
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    scale_len = 10  # 10 米
    sx = xlim[0] + (xlim[1] - xlim[0]) * 0.05
    sy = ylim[0] + (ylim[1] - ylim[0]) * 0.05
    ax.plot([sx, sx + scale_len], [sy, sy], 'k-', linewidth=3)
    ax.text(sx + scale_len / 2, sy + 1, f'{scale_len}m', ha='center',
            fontsize=9, fontweight='bold')

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight',
                    facecolor='white')
        print(f"  Saved: {output_file}")
    plt.close(fig)


if __name__ == "__main__":
    import torch

    with open(args.data, 'rb') as f:
        env = dill.load(f, encoding='latin1')

    # 加载模型
    eval_stg = None
    hyperparams = None
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
        print("模型已加载")

    os.makedirs(args.output_path, exist_ok=True)

    ph = args.ph

    # 确定要可视化的场景
    if args.scene_idx is not None:
        scene_indices = [args.scene_idx]
    else:
        scene_indices = list(range(min(args.num_scenes, len(env.scenes))))

    for si in scene_indices:
        scene = env.scenes[si]
        print(f"\n场景 {si}: {scene.name} (timesteps={scene.timesteps})")

        # 构建场景图
        if eval_stg is not None:
            scene.calculate_scene_graph(env.attention_radius,
                                        hyperparams['edge_addition_filter'],
                                        hyperparams['edge_removal_filter'])

        # 确定观测时刻
        if args.timestep is not None:
            timesteps_to_render = [args.timestep]
        else:
            # 选择 3 个代表性时刻: 20%, 50%, 80%
            ts = scene.timesteps
            timesteps_to_render = [
                max(args.history_len, int(ts * 0.2)),
                max(args.history_len, int(ts * 0.5)),
                max(args.history_len, min(ts - ph - 1, int(ts * 0.8))),
            ]

        for obs_t in timesteps_to_render:
            obs_t = min(obs_t, scene.timesteps - ph - 1)
            obs_t = max(obs_t, args.history_len)

            predictions = {}
            if eval_stg is not None:
                with torch.no_grad():
                    predictions = eval_stg.predict(
                        scene, np.array([obs_t]), ph,
                        num_samples=args.num_samples,
                        min_future_timesteps=ph,
                        z_mode=False, gmm_mode=False,
                        full_dist=False)

            output_file = os.path.join(
                args.output_path,
                f'bev_scene{si}_{scene.name}_t{obs_t}.png')

            render_bev(scene, obs_t, ph, args.history_len,
                       predictions=predictions if predictions else None,
                       zoom=args.zoom, output_file=output_file)

    print(f"\n完成! 图片保存在 {args.output_path}/")
