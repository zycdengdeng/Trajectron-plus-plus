"""
交叉口全局交通流可视化

叠加单场景内所有 agent 的完整轨迹，展示交通流模式。
按类型/速度/方向着色，呈现路口整体交通态势。

用法:
    # 基本用法: 可视化前 3 个场景
    python traffic_flow_viz.py \
        --data ../processed/car_road_test_full.pkl \
        --output_path results/traffic_flow

    # 指定场景 + 按速度着色
    python traffic_flow_viz.py \
        --data ../processed/car_road_test_full.pkl \
        --scene_idx 0 \
        --color_by speed \
        --output_path results/traffic_flow
"""
import sys
import os
import dill
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib import cm

sys.path.append("../../trajectron")

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="../processed/car_road_test_full.pkl")
parser.add_argument("--num_scenes", type=int, default=3, help="可视化场景数")
parser.add_argument("--scene_idx", type=int, default=None, help="指定场景索引")
parser.add_argument("--color_by", type=str, default="type",
                    choices=["type", "speed", "direction", "time"],
                    help="着色方式: type=按类型, speed=按速度, direction=按行驶方向, time=按时间")
parser.add_argument("--min_track_len", type=int, default=6,
                    help="最小轨迹长度 (帧), 过滤短轨迹")
parser.add_argument("--output_path", type=str, default="results/traffic_flow")
parser.add_argument("--zoom", type=float, default=None, help="缩放半径 (m)")
parser.add_argument("--dpi", type=int, default=150)
args = parser.parse_args()


def colored_line(ax, x, y, c, cmap, norm, linewidth=1.2, alpha=0.6):
    """绘制按标量值着色的渐变线段"""
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = mcoll.LineCollection(segments, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha, zorder=2)
    lc.set_array(np.array(c[:-1]))
    ax.add_collection(lc)
    return lc


def compute_track_stats(node):
    """计算轨迹统计信息: 平均速度、主方向"""
    xy = node.data.data[:, 0:2]
    if len(xy) < 2:
        return 0.0, 0.0
    diffs = np.diff(xy, axis=0)
    speeds = np.linalg.norm(diffs, axis=1) / 0.5  # dt=0.5s
    avg_speed = np.mean(speeds)
    # 主方向: 起点到终点的角度
    direction = np.arctan2(xy[-1, 1] - xy[0, 1], xy[-1, 0] - xy[0, 0])
    return avg_speed, direction


def render_traffic_flow(scene, color_by, min_track_len, zoom, output_file, dpi):
    """渲染单场景的全局交通流图"""
    fig, ax = plt.subplots(figsize=(16, 14))

    dt = scene.dt
    all_x, all_y = [], []
    vehicle_tracks = []
    ped_tracks = []

    for node in scene.nodes:
        xy = node.data.data[:, 0:2]
        if len(xy) < min_track_len:
            continue
        is_vehicle = node.type.name == 'VEHICLE'
        avg_speed, direction = compute_track_stats(node)
        track_info = {
            'xy': xy,
            'speed': avg_speed,
            'direction': direction,
            'node': node,
            'first_t': node.first_timestep,
        }
        if is_vehicle:
            vehicle_tracks.append(track_info)
        else:
            ped_tracks.append(track_info)
        all_x.extend(xy[:, 0].tolist())
        all_y.extend(xy[:, 1].tolist())

    all_tracks = vehicle_tracks + ped_tracks

    if len(all_tracks) == 0:
        print(f"  场景 {scene.name}: 无有效轨迹, 跳过")
        plt.close(fig)
        return

    # ── 按不同模式着色 ──
    if color_by == "type":
        # 先画行人 (底层), 再画车辆 (上层)
        for track in ped_tracks:
            xy = track['xy']
            ax.plot(xy[:, 0], xy[:, 1], '-', color='#FF9800', alpha=0.35,
                    linewidth=0.8, zorder=2)
            ax.plot(xy[0, 0], xy[0, 1], 'o', color='#FF9800', markersize=2,
                    alpha=0.5, zorder=3)
        for track in vehicle_tracks:
            xy = track['xy']
            ax.plot(xy[:, 0], xy[:, 1], '-', color='#1E88E5', alpha=0.4,
                    linewidth=1.2, zorder=4)
            ax.plot(xy[0, 0], xy[0, 1], 'o', color='#1E88E5', markersize=2,
                    alpha=0.5, zorder=5)
            # 终点箭头
            if len(xy) >= 2:
                dx = xy[-1, 0] - xy[-2, 0]
                dy = xy[-1, 1] - xy[-2, 1]
                norm_d = np.sqrt(dx**2 + dy**2)
                if norm_d > 0.01:
                    ax.annotate('', xy=(xy[-1, 0], xy[-1, 1]),
                                xytext=(xy[-1, 0] - dx/norm_d*2, xy[-1, 1] - dy/norm_d*2),
                                arrowprops=dict(arrowstyle='->', color='#1E88E5',
                                                lw=1.0, mutation_scale=10),
                                zorder=5)

        legend_elements = [
            Line2D([0], [0], color='#1E88E5', linewidth=2, alpha=0.7,
                   label=f'Vehicle ({len(vehicle_tracks)})'),
            Line2D([0], [0], color='#FF9800', linewidth=2, alpha=0.7,
                   label=f'Pedestrian ({len(ped_tracks)})'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=11,
                  framealpha=0.9)

    elif color_by == "speed":
        speeds = np.array([t['speed'] for t in all_tracks])
        norm = Normalize(vmin=0, vmax=min(np.percentile(speeds, 95), 20))
        cmap = cm.get_cmap('RdYlGn_r')  # 红=快, 绿=慢

        for track in all_tracks:
            xy = track['xy']
            # 逐段速度
            if len(xy) >= 2:
                diffs = np.diff(xy, axis=0)
                seg_speeds = np.linalg.norm(diffs, axis=1) / dt
                seg_speeds = np.append(seg_speeds, seg_speeds[-1])
                is_v = track['node'].type.name == 'VEHICLE'
                colored_line(ax, xy[:, 0], xy[:, 1], seg_speeds, cmap, norm,
                             linewidth=1.5 if is_v else 0.8, alpha=0.6)

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('Speed (m/s)', fontsize=11)

    elif color_by == "direction":
        cmap = cm.get_cmap('hsv')
        norm = Normalize(vmin=-np.pi, vmax=np.pi)

        for track in all_tracks:
            xy = track['xy']
            direction = track['direction']
            color = cmap(norm(direction))
            is_v = track['node'].type.name == 'VEHICLE'
            ax.plot(xy[:, 0], xy[:, 1], '-', color=color, alpha=0.5,
                    linewidth=1.5 if is_v else 0.8, zorder=2)
            if is_v and len(xy) >= 2:
                dx = xy[-1, 0] - xy[-2, 0]
                dy = xy[-1, 1] - xy[-2, 1]
                nd = np.sqrt(dx**2 + dy**2)
                if nd > 0.01:
                    ax.annotate('', xy=(xy[-1, 0], xy[-1, 1]),
                                xytext=(xy[-1, 0] - dx/nd*2, xy[-1, 1] - dy/nd*2),
                                arrowprops=dict(arrowstyle='->', color=color,
                                                lw=1.0, mutation_scale=10),
                                zorder=5)

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('Direction (rad)', fontsize=11)
        cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        cbar.set_ticklabels(['-π', '-π/2', '0', 'π/2', 'π'])

    elif color_by == "time":
        max_t = scene.timesteps
        norm = Normalize(vmin=0, vmax=max_t * dt)
        cmap = cm.get_cmap('viridis')

        for track in all_tracks:
            xy = track['xy']
            t_vals = (np.arange(len(xy)) + track['first_t']) * dt
            is_v = track['node'].type.name == 'VEHICLE'
            colored_line(ax, xy[:, 0], xy[:, 1], t_vals, cmap, norm,
                         linewidth=1.5 if is_v else 0.8, alpha=0.6)

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('Time (s)', fontsize=11)

    # ── 布局 ──
    if zoom is not None and len(all_x) > 0:
        cx, cy = np.mean(all_x), np.mean(all_y)
        ax.set_xlim(cx - zoom, cx + zoom)
        ax.set_ylim(cy - zoom, cy + zoom)
    elif len(all_x) > 0:
        margin = 20
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15, linestyle='--')
    ax.set_facecolor('#FAFAFA')

    # 比例尺
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    scale_len = 20
    sx = xlim[0] + (xlim[1] - xlim[0]) * 0.05
    sy = ylim[0] + (ylim[1] - ylim[0]) * 0.05
    ax.plot([sx, sx + scale_len], [sy, sy], 'k-', linewidth=3, zorder=20)
    ax.text(sx + scale_len / 2, sy + 2, f'{scale_len}m', ha='center',
            fontsize=9, fontweight='bold', zorder=20)

    duration = scene.timesteps * dt
    title = (f'Traffic Flow: {scene.name}  |  '
             f'Duration: {duration:.0f}s  |  '
             f'{len(vehicle_tracks)}V + {len(ped_tracks)}P  |  '
             f'Color: {color_by}')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_file}")
    plt.close(fig)


if __name__ == "__main__":
    with open(args.data, 'rb') as f:
        env = dill.load(f, encoding='latin1')

    os.makedirs(args.output_path, exist_ok=True)

    if args.scene_idx is not None:
        scene_indices = [args.scene_idx]
    else:
        scene_indices = list(range(min(args.num_scenes, len(env.scenes))))

    # 如果是单场景, 生成所有 color_by 模式; 否则只用指定模式
    if args.scene_idx is not None:
        color_modes = ["type", "speed", "direction"]
    else:
        color_modes = [args.color_by]

    for si in scene_indices:
        scene = env.scenes[si]
        print(f"\n场景 {si}: {scene.name} "
              f"(timesteps={scene.timesteps}, nodes={len(scene.nodes)})")

        for mode in color_modes:
            output_file = os.path.join(
                args.output_path,
                f'traffic_flow_scene{si}_{scene.name}_{mode}.png')
            render_traffic_flow(scene, mode, args.min_track_len, args.zoom,
                                output_file, args.dpi)

    print(f"\n完成! 图片保存在 {args.output_path}/")
