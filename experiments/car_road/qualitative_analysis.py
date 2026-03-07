"""
定性分析脚本: 选取 Good/Bad/Interesting Cases 进行可视化

功能:
1. 从评估结果中筛选 best/worst/边界 case
2. 对每个 case 绘制详细的轨迹对比图 (历史 + 真值 + 预测)
3. 标注 ADE/FDE 数值、速度、转弯角度等关键信息
4. 自动分类: 直行/转弯/减速/变道等场景

用法:
    python qualitative_analysis.py \
        --data ../processed/car_road_test_full.pkl \
        --model logs/models_05_Mar_2026_16_25_06_car_road \
        --checkpoint 80 \
        --output_path results/qualitative \
        --node_type VEHICLE \
        --num_cases 5
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

sys.path.append("../../trajectron")

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="../processed/car_road_test_full.pkl")
parser.add_argument("--model", type=str, default=None)
parser.add_argument("--checkpoint", type=int, default=80)
parser.add_argument("--ph", type=int, default=6)
parser.add_argument("--node_type", type=str, default="VEHICLE")
parser.add_argument("--num_cases", type=int, default=5,
                    help="每类 case 选取数量")
parser.add_argument("--num_samples", type=int, default=20)
parser.add_argument("--output_path", type=str, default="results/qualitative")
args = parser.parse_args()


def compute_curvature(traj):
    """计算轨迹总曲率 (累积转角)"""
    if len(traj) < 3:
        return 0.0
    dx = np.diff(traj[:, 0])
    dy = np.diff(traj[:, 1])
    angles = np.arctan2(dy, dx)
    d_angles = np.diff(angles)
    # 处理角度跳变
    d_angles = (d_angles + np.pi) % (2 * np.pi) - np.pi
    return np.abs(d_angles).sum()


def classify_scenario(history, future, speed):
    """根据轨迹特征分类场景类型"""
    if speed < 0.5:
        return "stationary"

    curv = compute_curvature(future)

    if len(history) >= 3:
        hist_speed = np.linalg.norm(np.diff(history[-3:], axis=0), axis=1)
        if len(hist_speed) >= 2 and hist_speed[-1] < hist_speed[0] * 0.5:
            return "braking"

    if curv > 0.5:
        return "turning"
    elif curv > 0.2:
        return "curved"
    else:
        return "straight"


def compute_sample_metrics(gt_future, pred_trajs, dt=0.5):
    """计算单个样本的 ADE/FDE"""
    # pred_trajs: [num_samples, ph, 2]
    # gt_future: [ph, 2]
    ph = min(len(gt_future), pred_trajs.shape[1])
    errors = np.linalg.norm(pred_trajs[:, :ph] - gt_future[:ph], axis=-1)
    ade = errors.mean(axis=1)  # [num_samples]
    fde = np.linalg.norm(pred_trajs[:, ph-1] - gt_future[ph-1], axis=-1)
    return ade.mean(), fde.mean()


def draw_case(ax, history, gt_future, pred_trajs, current_pos,
              node_type, heading=None, length=None, width=None,
              ade=None, fde=None, speed=None, scenario=None,
              title=""):
    """绘制单个 case 的详细图"""

    # 历史轨迹
    hist_color = '#1565C0' if node_type == 'VEHICLE' else '#E65100'
    if len(history) > 1:
        ax.plot(history[:, 0], history[:, 1], '-', color=hist_color,
                linewidth=2, alpha=0.7, zorder=5)
        # 历史点渐变 (越近越深)
        alphas = np.linspace(0.2, 0.8, len(history))
        ax.scatter(history[:, 0], history[:, 1], c=[hist_color],
                   s=15, alpha=alphas, zorder=6, edgecolors='none')

    # 当前位置
    if node_type == 'VEHICLE' and heading is not None:
        _length = length if length else 4.5
        _width = width if width else 1.8
        rect = patches.FancyBboxPatch(
            (-_length/2, -_width/2), _length, _width,
            boxstyle="round,pad=0.1",
            linewidth=2, edgecolor=hist_color, facecolor=hist_color,
            alpha=0.6)
        t = matplotlib.transforms.Affine2D().rotate(heading).translate(
            current_pos[0], current_pos[1]) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)
    else:
        ax.plot(current_pos[0], current_pos[1], 'o', color=hist_color,
                markersize=10, zorder=10)

    # 未来真值
    gt_full = np.vstack([current_pos.reshape(1, 2), gt_future])
    ax.plot(gt_full[:, 0], gt_full[:, 1], '--', color='#F44336',
            linewidth=2.5, alpha=0.9, zorder=7)
    ax.plot(gt_future[-1, 0], gt_future[-1, 1], 'x', color='#F44336',
            markersize=12, markeredgewidth=3, zorder=10)

    # 预测轨迹
    if pred_trajs is not None:
        for s in range(pred_trajs.shape[0]):
            pred_full = np.vstack([current_pos.reshape(1, 2), pred_trajs[s]])
            ax.plot(pred_full[:, 0], pred_full[:, 1], '-', color='#4CAF50',
                    alpha=0.25, linewidth=1, zorder=3)

        # 最佳预测 (最接近 GT 的样本)
        errors = np.linalg.norm(pred_trajs - gt_future[:pred_trajs.shape[1]],
                                axis=-1).mean(axis=1)
        best_idx = errors.argmin()
        best_full = np.vstack([current_pos.reshape(1, 2), pred_trajs[best_idx]])
        ax.plot(best_full[:, 0], best_full[:, 1], '-', color='#2E7D32',
                linewidth=2, alpha=0.8, zorder=8)
        ax.plot(pred_trajs[best_idx, -1, 0], pred_trajs[best_idx, -1, 1],
                '*', color='#2E7D32', markersize=10, zorder=10)

    # 标注信息
    info_lines = []
    if ade is not None:
        info_lines.append(f'ADE: {ade:.3f}m')
    if fde is not None:
        info_lines.append(f'FDE: {fde:.3f}m')
    if speed is not None:
        info_lines.append(f'Speed: {speed:.1f}m/s')
    if scenario is not None:
        info_lines.append(f'Type: {scenario}')

    if info_lines:
        info_text = '\n'.join(info_lines)
        ax.text(0.98, 0.98, info_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                          alpha=0.8, edgecolor='gray'))

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('X (m)', fontsize=9)
    ax.set_ylabel('Y (m)', fontsize=9)

    # 自动缩放
    all_pts = [current_pos]
    if len(history) > 0:
        all_pts.append(history)
    if len(gt_future) > 0:
        all_pts.append(gt_future)
    all_pts = np.vstack(all_pts)
    margin = 5
    ax.set_xlim(all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
    ax.set_ylim(all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)


if __name__ == "__main__":
    import torch
    from model.model_registrar import ModelRegistrar
    from model.trajectron import Trajectron
    from utils import prediction_output_to_trajectories
    import evaluation

    with open(args.data, 'rb') as f:
        env = dill.load(f, encoding='latin1')

    if args.model is None:
        print("错误: 定性分析需要指定 --model")
        sys.exit(1)

    # 加载模型
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
    max_hl = hyperparams['maximum_history_length']
    dt = env.scenes[0].dt

    # ── 第一阶段: 收集所有预测样本的指标 ──
    print("\n第一阶段: 收集所有预测样本的 ADE/FDE ...")
    all_cases = []

    for si, scene in enumerate(env.scenes):
        scene.calculate_scene_graph(env.attention_radius,
                                    hyperparams['edge_addition_filter'],
                                    hyperparams['edge_removal_filter'])

        timesteps = np.arange(scene.timesteps)
        with torch.no_grad():
            predictions = eval_stg.predict(scene, timesteps, ph,
                                           num_samples=args.num_samples,
                                           min_future_timesteps=ph,
                                           z_mode=False, gmm_mode=False,
                                           full_dist=False)

        if not predictions:
            continue

        for t in predictions.keys():
            for node in predictions[t].keys():
                if node.type.name != args.node_type:
                    continue

                pred = predictions[t][node][0]  # [num_samples, ph, 2]

                # 获取真值
                t_start = node.first_timestep
                local_t = t - t_start
                xy = node.data.data[:, 0:2]

                if local_t + ph >= xy.shape[0]:
                    continue

                gt_future = xy[local_t + 1: local_t + ph + 1]
                if len(gt_future) < ph:
                    continue

                # 历史轨迹
                hist_start = max(0, local_t - max_hl)
                history = xy[hist_start:local_t + 1]
                current_pos = xy[local_t]

                # ADE / FDE
                ade, fde = compute_sample_metrics(gt_future, pred, dt)

                # 速度
                vx = float(node.data[:, ('velocity', 'x')].flatten()[local_t])
                vy = float(node.data[:, ('velocity', 'y')].flatten()[local_t])
                speed = np.sqrt(vx**2 + vy**2)

                # 场景分类
                scenario = classify_scenario(history, gt_future, speed)

                # heading (VEHICLE only)
                heading = None
                if node.type.name == 'VEHICLE':
                    heading_data = node.data[:, ('heading', '°')]
                    heading = float(heading_data.flatten()[local_t])

                all_cases.append({
                    'scene_idx': si,
                    'scene_name': scene.name,
                    'timestep': t,
                    'node': node,
                    'node_id': str(node),
                    'ade': ade,
                    'fde': fde,
                    'speed': speed,
                    'scenario': scenario,
                    'curvature': compute_curvature(gt_future),
                    'history': history,
                    'gt_future': gt_future,
                    'current_pos': current_pos,
                    'pred_trajs': pred,
                    'heading': heading,
                    'length': node.length,
                    'width': node.width,
                })

    print(f"  共收集 {len(all_cases)} 个预测样本")

    if len(all_cases) == 0:
        print("无有效样本")
        sys.exit(0)

    # ── 第二阶段: 筛选代表性 case ──
    print("\n第二阶段: 筛选代表性 case ...")

    cases_df = pd.DataFrame([{
        'idx': i, 'ade': c['ade'], 'fde': c['fde'],
        'speed': c['speed'], 'scenario': c['scenario'],
        'curvature': c['curvature'], 'node_id': c['node_id']
    } for i, c in enumerate(all_cases)])

    # 只选运动样本 (speed >= 1.0)
    moving = cases_df[cases_df['speed'] >= 1.0]

    import pandas as pd

    categories = {}

    # Best cases (最低 ADE)
    best = moving.nsmallest(args.num_cases, 'ade')
    categories['best'] = best['idx'].tolist()

    # Worst cases (最高 ADE)
    worst = moving.nlargest(args.num_cases, 'ade')
    categories['worst'] = worst['idx'].tolist()

    # Turning cases (高曲率)
    turning = moving[moving['scenario'] == 'turning']
    if len(turning) >= args.num_cases:
        # 选 ADE 最中间的几个
        turning_sorted = turning.sort_values('ade')
        mid = len(turning_sorted) // 2
        start = max(0, mid - args.num_cases // 2)
        categories['turning'] = turning_sorted.iloc[start:start + args.num_cases]['idx'].tolist()
    elif len(turning) > 0:
        categories['turning'] = turning['idx'].tolist()

    # High speed cases
    fast = moving[moving['speed'] >= 15.0]
    if len(fast) >= args.num_cases:
        fast_sorted = fast.sort_values('ade')
        mid = len(fast_sorted) // 2
        start = max(0, mid - args.num_cases // 2)
        categories['high_speed'] = fast_sorted.iloc[start:start + args.num_cases]['idx'].tolist()
    elif len(fast) > 0:
        categories['high_speed'] = fast['idx'].tolist()

    # Median cases (中位数附近)
    median_ade = moving['ade'].median()
    near_median = moving.iloc[(moving['ade'] - median_ade).abs().argsort()[:args.num_cases]]
    categories['median'] = near_median['idx'].tolist()

    # ── 第三阶段: 绘制每类 case 的图 ──
    print("\n第三阶段: 绘制定性分析图 ...")

    # 汇总统计
    print(f"\n场景分类统计:")
    for scenario, grp in cases_df.groupby('scenario'):
        print(f"  {scenario}: {len(grp)} 个样本, ADE mean={grp['ade'].mean():.3f}")

    for cat_name, indices in categories.items():
        n = len(indices)
        if n == 0:
            continue

        cols = min(n, 3)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows))
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        axes = np.array(axes).flatten()

        for k, idx in enumerate(indices):
            case = all_cases[idx]
            ax = axes[k]

            title = (f"{case['node_id']} @ {case['scene_name']} t={case['timestep']}")

            draw_case(ax, case['history'], case['gt_future'],
                      case['pred_trajs'], case['current_pos'],
                      args.node_type, heading=case['heading'],
                      length=case['length'], width=case['width'],
                      ade=case['ade'], fde=case['fde'],
                      speed=case['speed'], scenario=case['scenario'],
                      title=title)

        # 隐藏多余子图
        for k in range(n, len(axes)):
            axes[k].set_visible(False)

        # 图例
        legend_elements = [
            Line2D([0], [0], color='#1565C0', linewidth=2, label='History'),
            Line2D([0], [0], color='#F44336', linewidth=2, linestyle='--',
                   label='Ground truth'),
            Line2D([0], [0], color='#4CAF50', linewidth=2, alpha=0.5,
                   label='Predictions'),
            Line2D([0], [0], color='#2E7D32', linewidth=2,
                   label='Best prediction'),
        ]

        cat_labels = {
            'best': 'Best Cases (低 ADE)',
            'worst': 'Worst Cases (高 ADE)',
            'turning': 'Turning Cases (转弯场景)',
            'high_speed': 'High Speed Cases (>15 m/s)',
            'median': 'Median Cases (中位数 ADE)',
        }

        fig.suptitle(f'{args.node_type} 定性分析: {cat_labels.get(cat_name, cat_name)}',
                     fontsize=14, fontweight='bold', y=1.02)

        out_file = os.path.join(args.output_path,
                                f'qualitative_{args.node_type}_{cat_name}.png')
        plt.savefig(out_file, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  Saved: {out_file} ({n} cases)")

    # ── 第四阶段: 汇总报告 ──
    print("\n" + "=" * 70)
    print(f"定性分析汇总 ({args.node_type})")
    print("=" * 70)

    for cat_name, indices in categories.items():
        if not indices:
            continue
        cat_cases = [all_cases[i] for i in indices]
        ades = [c['ade'] for c in cat_cases]
        fdes = [c['fde'] for c in cat_cases]
        speeds = [c['speed'] for c in cat_cases]
        scenarios = [c['scenario'] for c in cat_cases]

        print(f"\n{cat_labels.get(cat_name, cat_name)}:")
        print(f"  ADE:   {np.mean(ades):.3f} ± {np.std(ades):.3f}m "
              f"(range: {np.min(ades):.3f} ~ {np.max(ades):.3f})")
        print(f"  FDE:   {np.mean(fdes):.3f} ± {np.std(fdes):.3f}m")
        print(f"  Speed: {np.mean(speeds):.1f} ± {np.std(speeds):.1f} m/s")
        print(f"  Scenarios: {', '.join(scenarios)}")

    print(f"\n完成! 图片保存在 {args.output_path}/")
