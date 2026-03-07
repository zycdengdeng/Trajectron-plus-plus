"""
安全指标计算: TTC (Time-to-Collision) 和 PET (Post-Encroachment Time)

基于预测轨迹和真值轨迹，计算所有交互对之间的 TTC 和 PET 安全指标。

用法:
    # 使用模型预测轨迹计算
    python compute_safety_metrics.py \
        --data ../processed/car_road_test_full.pkl \
        --model logs/models_05_Mar_2026_16_25_06_car_road \
        --checkpoint 80 \
        --output_path results

    # 仅基于真值轨迹计算
    python compute_safety_metrics.py \
        --data ../processed/car_road_test_full.pkl \
        --output_path results
"""
import sys
import os
import dill
import json
import argparse
import numpy as np
import pandas as pd

sys.path.append("../../trajectron")

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="../processed/car_road_test_full.pkl")
parser.add_argument("--model", type=str, default=None)
parser.add_argument("--checkpoint", type=int, default=80)
parser.add_argument("--ph", type=int, default=6, help="预测时域")
parser.add_argument("--output_path", type=str, default="results")
parser.add_argument("--collision_dist", type=float, default=2.0,
                    help="碰撞判定距离阈值 (m)")
parser.add_argument("--pet_dist", type=float, default=3.0,
                    help="PET 冲突点判定距离阈值 (m)")
args = parser.parse_args()


def compute_ttc_linear(pos_a, vel_a, pos_b, vel_b, radius=2.0, dt=0.5, max_t=6.0):
    """
    基于线性外推计算 TTC。
    在当前时刻，根据两个 agent 的位置和速度，线性外推找到最近接近时刻。

    返回:
        ttc: 碰撞时间 (秒)，若不碰撞返回 np.inf
        min_dist: 最小接近距离 (m)
    """
    dp = pos_a - pos_b  # 相对位置
    dv = vel_a - vel_b  # 相对速度

    # 相对速度为零 → 不会碰撞
    dv_sq = np.dot(dv, dv)
    if dv_sq < 1e-10:
        dist = np.linalg.norm(dp)
        return (0.0 if dist < radius else np.inf), dist

    # 最近接近时刻: t* = -dp·dv / |dv|²
    t_closest = -np.dot(dp, dv) / dv_sq

    if t_closest < 0:
        # 已经在远离
        return np.inf, np.linalg.norm(dp)

    if t_closest > max_t:
        t_closest = max_t

    closest_dp = dp + dv * t_closest
    min_dist = np.linalg.norm(closest_dp)

    if min_dist < radius:
        # 反推碰撞时刻 (二次方程)
        a = dv_sq
        b = 2 * np.dot(dp, dv)
        c = np.dot(dp, dp) - radius ** 2
        discriminant = b ** 2 - 4 * a * c
        if discriminant >= 0:
            t_col = (-b - np.sqrt(discriminant)) / (2 * a)
            if 0 <= t_col <= max_t:
                return t_col, min_dist
        return t_closest, min_dist

    return np.inf, min_dist


def compute_ttc_trajectory(traj_a, traj_b, radius=2.0, dt=0.5):
    """
    基于轨迹点序列计算 TTC。
    检查两条轨迹在每个时间步的距离，找到第一次碰撞的时刻。

    参数:
        traj_a: shape (T, 2) - agent A 的轨迹
        traj_b: shape (T, 2) - agent B 的轨迹
        radius: 碰撞判定距离
        dt: 时间步长

    返回:
        ttc: 碰撞时间 (秒)，若不碰撞返回 np.inf
        min_dist: 最小接近距离
    """
    T = min(len(traj_a), len(traj_b))
    dists = np.linalg.norm(traj_a[:T] - traj_b[:T], axis=1)
    min_dist = dists.min()
    min_idx = dists.argmin()

    collision_mask = dists < radius
    if collision_mask.any():
        first_col = np.argmax(collision_mask)
        return first_col * dt, min_dist

    return np.inf, min_dist


def compute_pet(traj_a, traj_b, conflict_dist=3.0, dt=0.5):
    """
    计算 PET (Post-Encroachment Time)。

    PET 定义: 当两个 agent 先后经过同一空间区域时，
    后到达者到达该区域时刻 与 先到达者离开该区域时刻 之间的时间差。

    简化实现: 对于 traj_a 的每个点，找到 traj_b 中距离最近的点，
    若最近距离 < conflict_dist，则 PET = |t_a - t_b| * dt。

    返回:
        min_pet: 最小 PET (秒)，若无冲突返回 np.inf
        conflict_point: 冲突点坐标 (x, y) 或 None
    """
    min_pet = np.inf
    conflict_point = None

    T_a = len(traj_a)
    T_b = len(traj_b)

    for i in range(T_a):
        dists = np.linalg.norm(traj_b - traj_a[i], axis=1)
        j = np.argmin(dists)
        if dists[j] < conflict_dist:
            pet = abs(i - j) * dt
            if pet < min_pet and pet > 0:  # PET > 0 排除同时到达
                min_pet = pet
                conflict_point = (traj_a[i] + traj_b[j]) / 2

    return min_pet, conflict_point


def get_active_nodes_at(scene, t, ph):
    """获取在时刻 t 存在且有足够未来步的所有 node"""
    nodes = []
    for node in scene.nodes:
        t_start = node.first_timestep
        t_end = t_start + node.data.data.shape[0] - 1
        if t_start <= t and t_end >= t + ph:
            nodes.append(node)
    return nodes


def get_node_trajectory(node, t_start, t_end):
    """提取 node 在 [t_start, t_end] 的位置轨迹"""
    local_start = t_start - node.first_timestep
    local_end = t_end - node.first_timestep
    return node.data.data[local_start:local_end + 1, 0:2].copy()


def get_node_velocity(node, t):
    """提取 node 在时刻 t 的速度"""
    local_t = t - node.first_timestep
    vx = float(node.data[:, ('velocity', 'x')].flatten()[local_t])
    vy = float(node.data[:, ('velocity', 'y')].flatten()[local_t])
    return np.array([vx, vy])


def get_effective_radius(node_a, node_b):
    """根据 agent 尺寸计算等效碰撞半径"""
    r_a = 1.0  # PEDESTRIAN 默认半径
    r_b = 1.0
    if node_a.type.name == 'VEHICLE' and node_a.length is not None:
        r_a = max(node_a.length, node_a.width) / 2.0
    if node_b.type.name == 'VEHICLE' and node_b.length is not None:
        r_b = max(node_b.length, node_b.width) / 2.0
    return r_a + r_b


if __name__ == "__main__":
    import torch

    with open(args.data, 'rb') as f:
        env = dill.load(f, encoding='latin1')

    # 可选: 加载模型
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
    dt = env.scenes[0].dt
    results = []

    print(f"计算安全指标: TTC / PET")
    print(f"  数据集: {len(env.scenes)} 个场景")
    print(f"  预测时域: {ph} 步 ({ph * dt:.1f} 秒)")
    print(f"  碰撞判定距离: {args.collision_dist} m")
    print(f"  PET 冲突距离: {args.pet_dist} m")

    for si, scene in enumerate(env.scenes):
        print(f"\n场景 {si}: {scene.name} (timesteps={scene.timesteps})")

        # 模型预测 (如果可用)
        predictions = {}
        if eval_stg is not None:
            scene.calculate_scene_graph(env.attention_radius,
                                        hyperparams['edge_addition_filter'],
                                        hyperparams['edge_removal_filter'])

        # 每隔几步采样一个时刻进行分析
        sample_step = max(1, scene.timesteps // 20)
        eval_timesteps = list(range(ph, scene.timesteps - ph, sample_step))

        for t in eval_timesteps:
            active_nodes = get_active_nodes_at(scene, t, ph)
            if len(active_nodes) < 2:
                continue

            # 获取预测轨迹 (使用模型或线性外推)
            pred_trajs = {}
            if eval_stg is not None:
                with torch.no_grad():
                    preds = eval_stg.predict(scene, np.array([t]), ph,
                                             num_samples=1,
                                             min_future_timesteps=ph,
                                             z_mode=True, gmm_mode=True,
                                             full_dist=False)
                if t in preds:
                    for node in active_nodes:
                        if node in preds[t]:
                            # shape: [1, 1, ph, 2] → [ph, 2]
                            pred_trajs[node] = preds[t][node][0, 0]

            # 对所有交互对计算 TTC 和 PET
            for i in range(len(active_nodes)):
                for j in range(i + 1, len(active_nodes)):
                    node_a = active_nodes[i]
                    node_b = active_nodes[j]

                    # 真值轨迹
                    gt_traj_a = get_node_trajectory(node_a, t, t + ph)
                    gt_traj_b = get_node_trajectory(node_b, t, t + ph)

                    radius = get_effective_radius(node_a, node_b)

                    # --- TTC (真值) ---
                    gt_ttc, gt_min_dist = compute_ttc_trajectory(
                        gt_traj_a, gt_traj_b, radius=radius, dt=dt)

                    # --- TTC (线性外推) ---
                    pos_a = gt_traj_a[0]
                    pos_b = gt_traj_b[0]
                    vel_a = get_node_velocity(node_a, t)
                    vel_b = get_node_velocity(node_b, t)
                    lin_ttc, lin_min_dist = compute_ttc_linear(
                        pos_a, vel_a, pos_b, vel_b, radius=radius, dt=dt)

                    # --- TTC (模型预测) ---
                    pred_ttc = np.inf
                    pred_min_dist = np.inf
                    if node_a in pred_trajs and node_b in pred_trajs:
                        pred_ttc, pred_min_dist = compute_ttc_trajectory(
                            pred_trajs[node_a], pred_trajs[node_b],
                            radius=radius, dt=dt)

                    # --- PET (真值) ---
                    gt_pet, conflict_pt = compute_pet(
                        gt_traj_a, gt_traj_b, conflict_dist=args.pet_dist, dt=dt)

                    # --- PET (模型预测) ---
                    pred_pet = np.inf
                    if node_a in pred_trajs and node_b in pred_trajs:
                        pred_pet, _ = compute_pet(
                            pred_trajs[node_a], pred_trajs[node_b],
                            conflict_dist=args.pet_dist, dt=dt)

                    # 只记录有意义的交互 (最小距离 < 20m)
                    if gt_min_dist > 20.0:
                        continue

                    pair_type = f"{node_a.type.name}-{node_b.type.name}"

                    results.append({
                        'scene': scene.name,
                        'timestep': t,
                        'node_a': str(node_a),
                        'node_b': str(node_b),
                        'pair_type': pair_type,
                        'gt_ttc': gt_ttc,
                        'gt_min_dist': gt_min_dist,
                        'linear_ttc': lin_ttc,
                        'linear_min_dist': lin_min_dist,
                        'pred_ttc': pred_ttc if pred_ttc != np.inf else np.nan,
                        'pred_min_dist': pred_min_dist if pred_min_dist != np.inf else np.nan,
                        'gt_pet': gt_pet if gt_pet != np.inf else np.nan,
                        'pred_pet': pred_pet if pred_pet != np.inf else np.nan,
                        'collision_radius': radius,
                    })

    df = pd.DataFrame(results)

    if len(df) == 0:
        print("\n未发现有效交互对")
        sys.exit(0)

    # 保存原始数据
    csv_path = os.path.join(args.output_path, 'safety_metrics.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n已保存: {csv_path} ({len(df)} 条记录)")

    # ── 汇总统计 ──
    print("\n" + "=" * 70)
    print("安全指标汇总")
    print("=" * 70)

    # TTC 统计
    print("\n--- TTC (Time-to-Collision) ---")
    finite_ttc = df[df['gt_ttc'] < np.inf]['gt_ttc']
    print(f"  总交互对数: {len(df)}")
    print(f"  存在碰撞风险 (TTC < ∞): {len(finite_ttc)} ({100*len(finite_ttc)/len(df):.1f}%)")
    if len(finite_ttc) > 0:
        print(f"  TTC (真值) - mean: {finite_ttc.mean():.2f}s, "
              f"median: {finite_ttc.median():.2f}s, min: {finite_ttc.min():.2f}s")

    for threshold in [1.0, 2.0, 3.0]:
        n = (finite_ttc < threshold).sum()
        print(f"  TTC < {threshold}s: {n} 对 ({100*n/len(df):.2f}%)")

    # PET 统计
    print("\n--- PET (Post-Encroachment Time) ---")
    finite_pet = df['gt_pet'].dropna()
    print(f"  存在冲突点 (PET 有效): {len(finite_pet)} ({100*len(finite_pet)/len(df):.1f}%)")
    if len(finite_pet) > 0:
        print(f"  PET (真值) - mean: {finite_pet.mean():.2f}s, "
              f"median: {finite_pet.median():.2f}s, min: {finite_pet.min():.2f}s")
        for threshold in [1.0, 2.0, 3.0]:
            n = (finite_pet < threshold).sum()
            print(f"  PET < {threshold}s: {n} 对 ({100*n/len(finite_pet):.2f}%)")

    # 按交互对类型分组
    print("\n--- 按交互类型统计 ---")
    for pair_type, grp in df.groupby('pair_type'):
        n = len(grp)
        finite = grp[grp['gt_ttc'] < np.inf]['gt_ttc']
        print(f"  {pair_type}: {n} 对, "
              f"碰撞风险 {len(finite)} 对, "
              f"平均最小距离 {grp['gt_min_dist'].mean():.2f}m")

    # 预测 vs 真值对比
    if df['pred_ttc'].notna().any():
        print("\n--- 预测 vs 真值对比 ---")
        both_valid = df[df['pred_ttc'].notna() & (df['gt_ttc'] < np.inf)]
        if len(both_valid) > 0:
            ttc_error = (both_valid['pred_ttc'] - both_valid['gt_ttc']).abs()
            print(f"  TTC 误差 (|pred - gt|) - mean: {ttc_error.mean():.2f}s, "
                  f"median: {ttc_error.median():.2f}s")

        both_pet = df[df['pred_pet'].notna() & df['gt_pet'].notna()]
        if len(both_pet) > 0:
            pet_error = (both_pet['pred_pet'] - both_pet['gt_pet']).abs()
            print(f"  PET 误差 (|pred - gt|) - mean: {pet_error.mean():.2f}s, "
                  f"median: {pet_error.median():.2f}s")

    print(f"\n完成!")
