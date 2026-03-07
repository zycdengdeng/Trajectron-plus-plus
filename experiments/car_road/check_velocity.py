"""检查 test set 中 VEHICLE 速度分布和样本构成"""
import sys, os, dill
import numpy as np
import csv

sys.path.append("../../trajectron")

# 1. 加载 test data
data_path = sys.argv[1] if len(sys.argv) > 1 else '../processed/car_road_test_full.pkl'
print(f"加载: {data_path}")
with open(data_path, 'rb') as f:
    env = dill.load(f)

ph = 6  # prediction horizon

# 2. 统计每个可预测样本的速度
speeds = []
sample_info = []  # (scene_idx, node_id, local_t, speed)

for si, scene in enumerate(env.scenes):
    for node in scene.nodes:
        if node.type.name != 'VEHICLE':
            continue

        vx = node.data[:, ('velocity', 'x')].flatten()
        vy = node.data[:, ('velocity', 'y')].flatten()
        track_len = node.data.data.shape[0]

        for t in range(track_len - ph):
            speed = np.sqrt(float(vx[t])**2 + float(vy[t])**2)
            speeds.append(speed)
            sample_info.append((si, node.id, t, speed))

speeds = np.array(speeds)
print(f"\n总可预测 VEHICLE 样本: {len(speeds)}")

# 3. 速度统计
print(f"\n速度统计 (m/s):")
print(f"  mean:   {speeds.mean():.3f}")
print(f"  median: {np.median(speeds):.3f}")
print(f"  min:    {speeds.min():.3f}")
print(f"  max:    {speeds.max():.3f}")
print(f"  p25:    {np.percentile(speeds, 25):.3f}")
print(f"  p75:    {np.percentile(speeds, 75):.3f}")

# 4. 静止 vs 运动
stationary = np.sum(speeds < 0.5)
slow = np.sum(speeds < 1.0)
moving = np.sum(speeds >= 1.0)
print(f"\n速度分布:")
print(f"  静止 (< 0.5 m/s): {stationary} ({stationary/len(speeds)*100:.1f}%)")
print(f"  低速 (< 1.0 m/s): {slow} ({slow/len(speeds)*100:.1f}%)")
print(f"  运动 (>= 1.0 m/s): {moving} ({moving/len(speeds)*100:.1f}%)")

bins = [0, 0.5, 1, 2, 5, 10, 15, 20, 30, 100]
print(f"\n速度段分布:")
for i in range(len(bins)-1):
    count = np.sum((speeds >= bins[i]) & (speeds < bins[i+1]))
    pct = count / len(speeds) * 100
    print(f"  [{bins[i]:>5.1f}, {bins[i+1]:>5.1f}) m/s: {count:>6d} ({pct:>5.1f}%)")

# 5. 如果有评估结果，按速度段分析误差
ade_path = 'results/car_road_6_ade_most_likely_z.csv'
if os.path.exists(ade_path):
    ade_vals = []
    with open(ade_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ade_vals.append(float(row['value']))
    ade = np.array(ade_vals)

    print(f"\n按速度段分析 ADE (样本数 {len(ade)} vs 速度样本 {len(speeds)}):")
    if len(ade) == len(speeds):
        for lo, hi in [(0, 0.5), (0.5, 1), (1, 5), (5, 15), (15, 100)]:
            mask = (speeds >= lo) & (speeds < hi)
            if mask.sum() > 0:
                seg_ade = ade[mask]
                print(f"  [{lo:>5.1f}, {hi:>5.1f}) m/s: n={mask.sum():>5d}, "
                      f"ADE mean={seg_ade.mean():.3f}m, median={np.median(seg_ade):.3f}m, "
                      f"max={seg_ade.max():.3f}m")
    else:
        print(f"  样本数不匹配，无法按速度段分析")
        print(f"  (可能评估时有部分样本被跳过)")
