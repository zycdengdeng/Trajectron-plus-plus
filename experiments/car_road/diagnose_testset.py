"""
诊断 test set 构成，排查样本数异常 (5827 → 18707) 和极端 outlier

检查:
1. test set 场景数、节点数、时间步数
2. 静止 vs 运动车辆比例
3. 预测样本的速度分布
4. 极端误差样本特征

用法:
  python diagnose_testset.py --data ../processed/car_road_test_full.pkl \
      [--results-dir results] [--prediction-horizon 6]
"""

import argparse
import numpy as np
import os
import csv
import sys
import dill

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'trajectron'))
from model.online.online_trajectron import OnlineTrajectron
from model.model_registrar import ModelRegistrar
import evaluation


def load_test_data(data_path):
    """加载 test pickle"""
    with open(data_path, 'rb') as f:
        env = dill.load(f, encoding='latin1')
    return env


def analyze_test_scenes(env):
    """分析 test set 场景构成"""
    print(f"\n{'='*60}")
    print(f"  Test Set 场景构成")
    print(f"{'='*60}")

    scenes = env.scenes
    print(f"  场景数: {len(scenes)}")

    total_vehicle = 0
    total_ped = 0
    total_timesteps = 0
    all_vehicle_nodes = []

    for i, scene in enumerate(scenes):
        vehicles = [n for n in scene.nodes if n.type.name == 'VEHICLE']
        peds = [n for n in scene.nodes if n.type.name == 'PEDESTRIAN']
        total_vehicle += len(vehicles)
        total_ped += len(peds)
        total_timesteps += scene.timesteps
        all_vehicle_nodes.extend(vehicles)
        print(f"  Scene {i}: {scene.timesteps} 步, "
              f"{len(vehicles)} VEHICLE, {len(peds)} PEDESTRIAN")

    print(f"\n  总计: {total_vehicle} VEHICLE, {total_ped} PEDESTRIAN, "
          f"{total_timesteps} 总时间步")

    return scenes, all_vehicle_nodes


def analyze_velocity_distribution(scenes, ph=6):
    """分析可用于预测的样本的速度分布"""
    print(f"\n{'='*60}")
    print(f"  VEHICLE 预测样本速度分布 (ph={ph})")
    print(f"{'='*60}")

    speeds = []
    sample_count = 0
    stationary_count = 0  # speed < 0.5 m/s
    slow_count = 0        # speed < 1.0 m/s

    for scene in scenes:
        timesteps = np.arange(scene.timesteps)
        for node in scene.nodes:
            if node.type.name != 'VEHICLE':
                continue

            # 对每个时间步，检查是否有足够的 future
            for t in timesteps:
                # 检查该节点在 t 时刻是否存在
                if t < node.first_timestep or t >= node.last_timestep:
                    continue
                # 检查是否有足够 future 时间步
                future_end = t + ph
                if future_end > node.last_timestep:
                    continue

                sample_count += 1

                # 获取该时刻的速度
                try:
                    state = node.get(t, {'velocity': ['x', 'y']})
                    if state is not None and len(state) > 0:
                        vx, vy = state[0]
                        speed = np.sqrt(vx**2 + vy**2)
                        speeds.append(speed)
                        if speed < 0.5:
                            stationary_count += 1
                        if speed < 1.0:
                            slow_count += 1
                except:
                    pass

    speeds = np.array(speeds)
    print(f"  可预测样本总数: {sample_count}")
    print(f"  有速度信息的样本: {len(speeds)}")

    if len(speeds) > 0:
        print(f"\n  速度统计 (m/s):")
        print(f"    mean:   {speeds.mean():.3f}")
        print(f"    median: {np.median(speeds):.3f}")
        print(f"    std:    {speeds.std():.3f}")
        print(f"    min:    {speeds.min():.3f}")
        print(f"    max:    {speeds.max():.3f}")
        print(f"    p25:    {np.percentile(speeds, 25):.3f}")
        print(f"    p75:    {np.percentile(speeds, 75):.3f}")

        print(f"\n  速度分布:")
        print(f"    静止 (< 0.5 m/s): {stationary_count} ({stationary_count/len(speeds)*100:.1f}%)")
        print(f"    低速 (< 1.0 m/s): {slow_count} ({slow_count/len(speeds)*100:.1f}%)")
        print(f"    运动 (>= 1.0 m/s): {len(speeds)-slow_count} ({(len(speeds)-slow_count)/len(speeds)*100:.1f}%)")

        # 按速度段统计
        bins = [0, 0.5, 1, 2, 5, 10, 15, 20, 30, 100]
        print(f"\n  速度段分布:")
        for i in range(len(bins)-1):
            count = np.sum((speeds >= bins[i]) & (speeds < bins[i+1]))
            pct = count / len(speeds) * 100
            print(f"    [{bins[i]:>5.1f}, {bins[i+1]:>5.1f}) m/s: {count:>6d} ({pct:>5.1f}%)")

    return speeds, sample_count


def analyze_outliers(results_dir):
    """分析极端误差样本"""
    print(f"\n{'='*60}")
    print(f"  极端误差样本分析")
    print(f"{'='*60}")

    ade_path = os.path.join(results_dir, 'car_road_6_ade_most_likely_z.csv')
    fde_path = os.path.join(results_dir, 'car_road_6_fde_most_likely_z.csv')

    if not os.path.exists(ade_path) or not os.path.exists(fde_path):
        print("  CSV 文件不存在，跳过")
        return

    def read_csv(path):
        values = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                values.append(float(row['value']))
        return np.array(values)

    ade = read_csv(ade_path)
    fde = read_csv(fde_path)

    print(f"  总样本数: {len(ade)}")

    # Top 20 worst ADE
    worst_ade_idx = np.argsort(ade)[-20:][::-1]
    print(f"\n  Top 20 最差 ADE:")
    print(f"  {'排名':>4} {'索引':>8} {'ADE(m)':>10} {'FDE(m)':>10}")
    print(f"  {'-'*4} {'-'*8} {'-'*10} {'-'*10}")
    for rank, idx in enumerate(worst_ade_idx, 1):
        print(f"  {rank:>4} {idx:>8} {ade[idx]:>10.3f} {fde[idx]:>10.3f}")

    # 统计极端误差
    print(f"\n  极端误差统计:")
    for thresh in [5, 10, 20, 30, 50]:
        ade_count = np.sum(ade > thresh)
        fde_count = np.sum(fde > thresh)
        print(f"    > {thresh}m: ADE {ade_count} 个 ({ade_count/len(ade)*100:.2f}%), "
              f"FDE {fde_count} 个 ({fde_count/len(fde)*100:.2f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='../processed/car_road_test_full.pkl')
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--prediction-horizon', type=int, default=6)
    args = parser.parse_args()

    # 1. 加载并分析 test set
    if os.path.exists(args.data):
        print(f"加载 test data: {args.data}")
        env = load_test_data(args.data)
        scenes, vehicle_nodes = analyze_test_scenes(env)
        speeds, sample_count = analyze_velocity_distribution(scenes, args.prediction_horizon)
    else:
        print(f"警告: test data 不存在 ({args.data})，跳过数据分析")
        print(f"请指定正确路径: python diagnose_testset.py --data <path_to_test.pkl>")

    # 2. 分析评估结果中的 outlier
    if os.path.exists(args.results_dir):
        analyze_outliers(args.results_dir)
    else:
        print(f"警告: results 目录不存在 ({args.results_dir})")

    print()


if __name__ == '__main__':
    main()
