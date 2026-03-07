"""
TTC/PET 危险事件可视化

从 safety_metrics.csv 中筛选高危交互事件，在 BEV 图上绘制:
- 两个 agent 的历史 + 未来轨迹
- 冲突点标注 (星形)
- TTC/PET 数值标注
- 危险等级颜色编码

用法:
    # 基本用法: 从 safety_metrics.csv 筛选并可视化
    python safety_event_viz.py \
        --data ../processed/car_road_test_full.pkl \
        --csv results/safety_metrics.csv \
        --output_path results/safety_events

    # 带模型预测
    python safety_event_viz.py \
        --data ../processed/car_road_test_full.pkl \
        --csv results/safety_metrics.csv \
        --model logs/models_05_Mar_2026_16_25_06_car_road \
        --checkpoint 80 \
        --output_path results/safety_events

    # 只看 V-V 冲突, TTC < 1s
    python safety_event_viz.py \
        --data ../processed/car_road_test_full.pkl \
        --csv results/safety_metrics.csv \
        --pair_type VEHICLE-VEHICLE \
        --ttc_threshold 1.0 \
        --output_path results/safety_events
"""
import sys
import os
import dill
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

sys.path.append("../../trajectron")

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="../processed/car_road_test_full.pkl")
parser.add_argument("--csv", type=str, default="results/safety_metrics.csv",
                    help="safety_metrics.csv 路径")
parser.add_argument("--model", type=str, default=None)
parser.add_argument("--checkpoint", type=int, default=80)
parser.add_argument("--ph", type=int, default=6)
parser.add_argument("--num_samples", type=int, default=20)
parser.add_argument("--ttc_threshold", type=float, default=2.0,
                    help="TTC 筛选阈值 (秒), 只显示 TTC < 阈值的事件")
parser.add_argument("--pet_threshold", type=float, default=2.0,
                    help="PET 筛选阈值 (秒)")
parser.add_argument("--max_events", type=int, default=20,
                    help="最多可视化的事件数")
parser.add_argument("--pair_type", type=str, default=None,
                    help="只筛选特定交互类型, 如 VEHICLE-VEHICLE")
parser.add_argument("--history_len", type=int, default=8)
parser.add_argument("--context_radius", type=float, default=30.0,
                    help="可视化范围半径 (m)")
parser.add_argument("--output_path", type=str, default="results/safety_events")
parser.add_argument("--dpi", type=int, default=150)
args = parser.parse_args()


# ── 危险等级颜色 ──
DANGER_COLORS = {
    'critical': '#D32F2F',   # 深红: TTC < 1s
    'high':     '#F44336',   # 红: TTC 1-2s
    'medium':   '#FF9800',   # 橙: TTC 2-3s
    'low':      '#FFC107',   # 黄: TTC > 3s
}


def get_danger_level(ttc):
    if ttc <= 0.5:
        return 'critical', DANGER_COLORS['critical']
    elif ttc <= 1.0:
        return 'high', DANGER_COLORS['high']
    elif ttc <= 2.0:
        return 'medium', DANGER_COLORS['medium']
    else:
        return 'low', DANGER_COLORS['low']


def find_node_by_str(scene, node_str):
    """通过字符串表示查找 node"""
    for node in scene.nodes:
        if str(node) == node_str:
            return node
    return None


def draw_agent(ax, node, t_obs, history_len, ph, color, label_prefix=""):
    """绘制单个 agent 的当前位置、历史、未来"""
    t_start = node.first_timestep
    t_end = t_start + node.data.data.shape[0] - 1
    xy = node.data.data[:, 0:2]
    local_obs = t_obs - t_start

    cur_x, cur_y = xy[local_obs]
    is_vehicle = node.type.name == 'VEHICLE'

    # 当前位置
    if is_vehicle:
        heading_data = node.data[:, ('heading', '°')]
        heading = float(heading_data.flatten()[local_obs])
        length = node.length if node.length else 4.5
        width = node.width if node.width else 1.8
        rect = patches.FancyBboxPatch(
            (-length/2, -width/2), length, width,
            boxstyle="round,pad=0.1",
            linewidth=2.0, edgecolor=color, facecolor=color, alpha=0.5)
        t = matplotlib.transforms.Affine2D().rotate(heading).translate(cur_x, cur_y) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)
        # 朝向箭头
        dx = length/2 * np.cos(heading)
        dy = length/2 * np.sin(heading)
        ax.arrow(cur_x, cur_y, dx*0.6, dy*0.6, head_width=0.6, head_length=0.3,
                 fc=color, ec=color, alpha=0.8, zorder=12)
    else:
        circle = plt.Circle((cur_x, cur_y), 0.6, color=color, alpha=0.6, zorder=12)
        ax.add_patch(circle)

    # 历史轨迹
    hist_start = max(0, local_obs - history_len)
    hist = xy[hist_start:local_obs + 1]
    if len(hist) >= 2:
        ax.plot(hist[:, 0], hist[:, 1], '-', color=color, alpha=0.4,
                linewidth=1.5, marker='.', markersize=3, zorder=5)

    # 未来真值
    future_end = min(local_obs + ph, xy.shape[0] - 1)
    future = xy[local_obs:future_end + 1]
    if len(future) >= 2:
        ax.plot(future[:, 0], future[:, 1], '--', color=color, alpha=0.7,
                linewidth=2.5, zorder=8)
        # 逐步时间标记
        for step in range(1, len(future)):
            ax.plot(future[step, 0], future[step, 1], 'o', color=color,
                    markersize=4, alpha=0.6, zorder=9)
        # 终点
        ax.plot(future[-1, 0], future[-1, 1], 'X', color=color,
                markersize=10, markeredgewidth=2, zorder=10)

    # 标签
    type_short = 'V' if is_vehicle else 'P'
    node_id = str(node).split('/')[-1] if '/' in str(node) else str(node)
    ax.annotate(f'{label_prefix}{type_short}:{node_id}',
                xy=(cur_x, cur_y), xytext=(5, 5),
                textcoords='offset points', fontsize=8, fontweight='bold',
                color=color, zorder=15,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor=color, alpha=0.8))

    return cur_x, cur_y, future


def draw_conflict_point(ax, pos, ttc, pet, danger_color):
    """绘制冲突点标记"""
    ax.plot(pos[0], pos[1], '*', color=danger_color, markersize=20,
            markeredgewidth=1.5, markeredgecolor='black', zorder=20)

    # TTC/PET 标注
    text_parts = []
    if ttc is not None and not np.isinf(ttc):
        text_parts.append(f'TTC={ttc:.1f}s')
    if pet is not None and not np.isnan(pet):
        text_parts.append(f'PET={pet:.1f}s')
    if text_parts:
        ax.annotate('\n'.join(text_parts),
                    xy=(pos[0], pos[1]), xytext=(12, 12),
                    textcoords='offset points', fontsize=9, fontweight='bold',
                    color=danger_color, zorder=20,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor=danger_color, alpha=0.9),
                    arrowprops=dict(arrowstyle='->', color=danger_color, lw=1.5))


def draw_context_agents(ax, scene, t_obs, exclude_nodes, radius, center):
    """绘制周围的上下文 agent (灰色半透明)"""
    cx, cy = center
    for node in scene.nodes:
        if node in exclude_nodes:
            continue
        t_start = node.first_timestep
        t_end = t_start + node.data.data.shape[0] - 1
        if t_start > t_obs or t_end < t_obs:
            continue
        xy = node.data.data[:, 0:2]
        local_obs = t_obs - t_start
        x, y = xy[local_obs]
        if abs(x - cx) > radius or abs(y - cy) > radius:
            continue

        is_vehicle = node.type.name == 'VEHICLE'
        if is_vehicle:
            heading_data = node.data[:, ('heading', '°')]
            heading = float(heading_data.flatten()[local_obs])
            length = node.length if node.length else 4.5
            width = node.width if node.width else 1.8
            rect = patches.FancyBboxPatch(
                (-length/2, -width/2), length, width,
                boxstyle="round,pad=0.1",
                linewidth=0.8, edgecolor='#9E9E9E', facecolor='#E0E0E0', alpha=0.4)
            t = matplotlib.transforms.Affine2D().rotate(heading).translate(x, y) + ax.transData
            rect.set_transform(t)
            ax.add_patch(rect)
        else:
            circle = plt.Circle((x, y), 0.4, color='#BDBDBD', alpha=0.4, zorder=1)
            ax.add_patch(circle)


def render_safety_event(scene, node_a, node_b, t_obs, ph, history_len,
                        ttc, pet, min_dist, pair_type, predictions,
                        context_radius, output_file, dpi):
    """渲染单个危险事件"""
    fig, ax = plt.subplots(figsize=(12, 12))

    color_a = '#1565C0'  # 蓝
    color_b = '#E65100'  # 橙

    # 绘制两个主要 agent
    x_a, y_a, future_a = draw_agent(ax, node_a, t_obs, history_len, ph,
                                     color_a, label_prefix="A: ")
    x_b, y_b, future_b = draw_agent(ax, node_b, t_obs, history_len, ph,
                                     color_b, label_prefix="B: ")

    # 模型预测
    if predictions is not None and t_obs in predictions:
        for node, color in [(node_a, color_a), (node_b, color_b)]:
            if node in predictions[t_obs]:
                pred = predictions[t_obs][node][0]  # [num_samples, ph, 2]
                local_obs = t_obs - node.first_timestep
                cur_pos = node.data.data[local_obs, 0:2]
                n_samples = pred.shape[0]
                for s in range(n_samples):
                    full = np.vstack([cur_pos.reshape(1, 2), pred[s]])
                    ax.plot(full[:, 0], full[:, 1], '-', color=color,
                            alpha=0.1, linewidth=0.8, zorder=3)

    # 冲突点 (两条未来轨迹的最近接近点)
    if len(future_a) > 1 and len(future_b) > 1:
        T = min(len(future_a), len(future_b))
        dists = np.linalg.norm(future_a[:T] - future_b[:T], axis=1)
        min_idx = np.argmin(dists)
        conflict_pos = (future_a[min_idx] + future_b[min_idx]) / 2

        danger_level, danger_color = get_danger_level(ttc if ttc < np.inf else 5.0)
        draw_conflict_point(ax, conflict_pos, ttc, pet, danger_color)

        # 连接两个 agent 到冲突点的虚线
        ax.plot([future_a[min_idx, 0], future_b[min_idx, 0]],
                [future_a[min_idx, 1], future_b[min_idx, 1]],
                ':', color=danger_color, linewidth=1.5, alpha=0.5, zorder=7)

    # 当前两 agent 之间的距离标注
    mid_x = (x_a + x_b) / 2
    mid_y = (y_a + y_b) / 2
    cur_dist = np.sqrt((x_a - x_b)**2 + (y_a - y_b)**2)
    ax.plot([x_a, x_b], [y_a, y_b], ':', color='gray', linewidth=1, alpha=0.5)
    ax.text(mid_x, mid_y, f'{cur_dist:.1f}m', fontsize=8, ha='center',
            color='gray', alpha=0.7,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6))

    # 上下文 agent
    center = ((x_a + x_b) / 2, (y_a + y_b) / 2)
    draw_context_agents(ax, scene, t_obs, {node_a, node_b},
                        context_radius, center)

    # ── 布局 ──
    cx, cy = center
    r = context_radius
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15, linestyle='--')
    ax.set_facecolor('#FAFAFA')

    # 图例
    danger_level, danger_color = get_danger_level(ttc if ttc < np.inf else 5.0)
    legend_elements = [
        Line2D([0], [0], color=color_a, linewidth=2, label=f'Agent A ({node_a.type.name})'),
        Line2D([0], [0], color=color_b, linewidth=2, label=f'Agent B ({node_b.type.name})'),
        Line2D([0], [0], color=color_a, linewidth=2, linestyle='--', label='GT Future'),
        Line2D([0], [0], marker='*', color=danger_color, markersize=15,
               linestyle='None', label=f'Conflict ({danger_level})'),
        Line2D([0], [0], color='#9E9E9E', linewidth=1, label='Context agents'),
    ]
    if predictions is not None:
        legend_elements.insert(3,
            Line2D([0], [0], color='gray', linewidth=1, alpha=0.4,
                   label='Model predictions'))
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10,
              framealpha=0.9)

    # 标题
    dt = scene.dt
    ttc_str = f'{ttc:.2f}s' if ttc < np.inf else '∞'
    pet_str = f'{pet:.2f}s' if not np.isnan(pet) else 'N/A'
    title = (f'Safety Event: {scene.name}  |  t={t_obs} ({t_obs*dt:.1f}s)\n'
             f'{pair_type}  |  TTC={ttc_str}  |  PET={pet_str}  |  '
             f'MinDist={min_dist:.2f}m  |  Danger: {danger_level.upper()}')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)

    # 比例尺
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    scale_len = 10
    sx = xlim[0] + (xlim[1] - xlim[0]) * 0.05
    sy = ylim[0] + (ylim[1] - ylim[0]) * 0.05
    ax.plot([sx, sx + scale_len], [sy, sy], 'k-', linewidth=3, zorder=20)
    ax.text(sx + scale_len/2, sy + 1, f'{scale_len}m', ha='center',
            fontsize=9, fontweight='bold', zorder=20)

    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def render_summary_grid(events_info, output_file, dpi=150):
    """渲染危险事件汇总网格 (每个事件一小图)"""
    n = len(events_info)
    if n == 0:
        return

    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, info in enumerate(events_info):
        r, c = idx // cols, idx % cols
        ax = axes[r, c]

        scene = info['scene_obj']
        node_a = info['node_a_obj']
        node_b = info['node_b_obj']
        t_obs = info['timestep']
        ttc = info['ttc']
        pet = info['pet']

        # 简化绘制: 只画两条轨迹
        for node, color in [(node_a, '#1565C0'), (node_b, '#E65100')]:
            t_start = node.first_timestep
            xy = node.data.data[:, 0:2]
            local_obs = t_obs - t_start
            ph = 6

            # 历史
            hist_start = max(0, local_obs - 8)
            hist = xy[hist_start:local_obs + 1]
            if len(hist) >= 2:
                ax.plot(hist[:, 0], hist[:, 1], '-', color=color, alpha=0.4,
                        linewidth=1.0, zorder=3)

            # 当前位置
            ax.plot(xy[local_obs, 0], xy[local_obs, 1], 'o', color=color,
                    markersize=6, zorder=10)

            # 未来
            future_end = min(local_obs + ph, xy.shape[0] - 1)
            future = xy[local_obs:future_end + 1]
            if len(future) >= 2:
                ax.plot(future[:, 0], future[:, 1], '--', color=color,
                        alpha=0.6, linewidth=1.5, zorder=5)
                ax.plot(future[-1, 0], future[-1, 1], 'x', color=color,
                        markersize=6, markeredgewidth=1.5, zorder=10)

        # 冲突标记
        x_a, y_a = node_a.data.data[t_obs - node_a.first_timestep, 0:2]
        x_b, y_b = node_b.data.data[t_obs - node_b.first_timestep, 0:2]
        cx, cy = (x_a + x_b) / 2, (y_a + y_b) / 2
        _, danger_color = get_danger_level(ttc)
        ax.plot(cx, cy, '*', color=danger_color, markersize=12,
                markeredgecolor='black', markeredgewidth=0.5, zorder=15)

        # 范围
        r_view = 25
        ax.set_xlim(cx - r_view, cx + r_view)
        ax.set_ylim(cy - r_view, cy + r_view)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.1, linestyle='--')
        ax.set_facecolor('#FAFAFA')

        ttc_str = f'{ttc:.1f}s' if ttc < np.inf else '∞'
        pet_str = f'{pet:.1f}s' if not np.isnan(pet) else '-'
        ax.set_title(f'{info["pair_type"]}  TTC={ttc_str}  PET={pet_str}',
                     fontsize=9, fontweight='bold')
        ax.tick_params(labelsize=7)

    # 隐藏多余的子图
    for idx in range(n, rows * cols):
        r, c = idx // cols, idx % cols
        axes[r, c].set_visible(False)

    fig.suptitle('Safety Events Summary (sorted by TTC)', fontsize=14,
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"  Saved summary: {output_file}")
    plt.close(fig)


if __name__ == "__main__":
    import torch

    # ── 加载数据 ──
    with open(args.data, 'rb') as f:
        env = dill.load(f, encoding='latin1')

    # 建立场景名到场景对象的映射
    scene_map = {scene.name: scene for scene in env.scenes}

    # ── 加载模型 (可选) ──
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

    # ── 读取 safety_metrics.csv ──
    df = pd.read_csv(args.csv)
    print(f"读取 {len(df)} 条安全指标记录")

    # ── 筛选高危事件 ──
    # 条件: TTC < 阈值 且 TTC > 0 (排除已碰撞的)
    mask = (df['gt_ttc'] > 0) & (df['gt_ttc'] < args.ttc_threshold)

    if args.pair_type is not None:
        mask = mask & (df['pair_type'] == args.pair_type)

    dangerous = df[mask].copy()
    dangerous = dangerous.sort_values('gt_ttc').head(args.max_events)
    print(f"筛选到 {len(dangerous)} 个高危事件 (TTC < {args.ttc_threshold}s)")

    if len(dangerous) == 0:
        # 退而求其次: 用 PET 筛选
        print("未找到 TTC 高危事件, 尝试 PET 筛选...")
        mask = df['gt_pet'].notna() & (df['gt_pet'] < args.pet_threshold)
        if args.pair_type is not None:
            mask = mask & (df['pair_type'] == args.pair_type)
        dangerous = df[mask].copy()
        dangerous = dangerous.sort_values('gt_pet').head(args.max_events)
        print(f"筛选到 {len(dangerous)} 个 PET 冲突事件 (PET < {args.pet_threshold}s)")

    if len(dangerous) == 0:
        print("未找到任何高危事件, 退出")
        sys.exit(0)

    os.makedirs(args.output_path, exist_ok=True)

    # ── 逐事件可视化 ──
    events_for_summary = []

    for rank, (_, row) in enumerate(dangerous.iterrows()):
        scene_name = row['scene']
        t_obs = int(row['timestep'])
        node_a_str = row['node_a']
        node_b_str = row['node_b']
        ttc = row['gt_ttc']
        pet = row['gt_pet'] if not np.isnan(row['gt_pet']) else np.nan
        min_dist = row['gt_min_dist']
        pair_type = row['pair_type']

        if scene_name not in scene_map:
            print(f"  场景 {scene_name} 未找到, 跳过")
            continue

        scene = scene_map[scene_name]
        node_a = find_node_by_str(scene, node_a_str)
        node_b = find_node_by_str(scene, node_b_str)

        if node_a is None or node_b is None:
            print(f"  节点未找到: {node_a_str} / {node_b_str}, 跳过")
            continue

        # 模型预测
        predictions = None
        if eval_stg is not None:
            scene.calculate_scene_graph(env.attention_radius,
                                        hyperparams['edge_addition_filter'],
                                        hyperparams['edge_removal_filter'])
            with torch.no_grad():
                predictions = eval_stg.predict(
                    scene, np.array([t_obs]), args.ph,
                    num_samples=args.num_samples,
                    min_future_timesteps=args.ph,
                    z_mode=False, gmm_mode=False,
                    full_dist=False)

        output_file = os.path.join(
            args.output_path,
            f'safety_event_{rank:02d}_{pair_type}_ttc{ttc:.1f}s_{scene_name}_t{t_obs}.png')

        print(f"\n  [{rank+1}/{len(dangerous)}] {pair_type} | "
              f"TTC={ttc:.2f}s | PET={pet:.2f}s | "
              f"scene={scene_name} t={t_obs}")

        render_safety_event(scene, node_a, node_b, t_obs, args.ph,
                            args.history_len, ttc, pet, min_dist, pair_type,
                            predictions, args.context_radius,
                            output_file, args.dpi)
        print(f"  Saved: {output_file}")

        events_for_summary.append({
            'scene_obj': scene, 'node_a_obj': node_a, 'node_b_obj': node_b,
            'timestep': t_obs, 'ttc': ttc, 'pet': pet, 'pair_type': pair_type,
        })

    # ── 汇总网格图 ──
    if len(events_for_summary) > 0:
        summary_file = os.path.join(args.output_path, 'safety_events_summary.png')
        render_summary_grid(events_for_summary[:16], summary_file, args.dpi)

    print(f"\n完成! 图片保存在 {args.output_path}/")
