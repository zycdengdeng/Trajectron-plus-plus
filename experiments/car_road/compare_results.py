"""
对比分析: Epoch 20 (修复前) vs Epoch 100 (修复后)

验证三项修复是否生效:
1. PEDESTRIAN standardization (std: 1 → 80/40)
2. 速度/加速度 clipping: 分量级 → norm级 (消除 √2 放大)
3. 训练 epoch: 20 → 100 (KL annealing 充分收敛)

用法:
  python compare_results.py [--results-dir results]

注意: 脚本中硬编码了 Epoch 20 的 baseline 数据 (来自 EXPERIMENT_LOG.md)
"""

import csv
import numpy as np
import os
import sys
import argparse


# =====================================================
# Epoch 20 Baseline (来自 EXPERIMENT_LOG.md Section 5)
# VEHICLE, test set, ph=6, Most Likely Z, 5827 samples
# =====================================================
BASELINE = {
    'ade': {
        'mean': 1.084, 'median': 0.769, 'std': 1.331,
        'min': 0.047, 'max': 37.119,
        'p25': 0.238, 'p75': 1.419, 'p95': 3.068,
        'n_samples': 5827,
        'thresholds': {0.5: 52.6, 1: 39.6, 2: 13.0, 3: 5.3, 5: 1.5, 10: 0.1},
    },
    'fde': {
        'mean': 2.020, 'median': 1.466, 'std': 2.221,
        'min': 0.016, 'max': 37.169,
        'p25': 0.449, 'p75': 2.696, 'p95': 5.905,
        'n_samples': 5827,
        'thresholds': {1: 60.4, 2: 37.0, 5: 7.4, 10: 1.1, 20: 0.1},
    },
    'kde': {
        'mean': 1.685, 'median': 1.973, 'std': 1.956,
        'min': -1.745, 'max': 20.000,
        'p25': -0.129, 'p75': 2.720, 'p95': 4.235,
        'n_samples': 5827,
    },
}


def read_csv_values(path):
    """读取评估 CSV 文件"""
    values = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            values.append(float(row['value']))
    return np.array(values)


def compute_stats(values):
    """计算统计量"""
    return {
        'mean': values.mean(),
        'median': np.median(values),
        'std': values.std(),
        'min': values.min(),
        'max': values.max(),
        'p25': np.percentile(values, 25),
        'p75': np.percentile(values, 75),
        'p95': np.percentile(values, 95),
        'n_samples': len(values),
    }


def compute_thresholds(values, thresholds):
    """计算超过阈值的比例"""
    return {t: (values > t).mean() * 100 for t in thresholds}


def pct_change(old, new):
    """计算变化百分比"""
    if old == 0:
        return float('inf')
    return (new - old) / abs(old) * 100


def print_comparison(metric_name, old_stats, new_stats, old_thresh=None, new_thresh=None):
    """打印对比表格"""
    print(f"\n{'='*72}")
    print(f"  {metric_name.upper()} 对比: Epoch 20 (修复前) → Epoch 100 (修复后)")
    print(f"{'='*72}")
    print(f"  {'指标':<12} {'Epoch 20':>12} {'Epoch 100':>12} {'变化':>12} {'变化%':>10}")
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")

    for key in ['n_samples', 'mean', 'median', 'std', 'min', 'max', 'p25', 'p75', 'p95']:
        old_val = old_stats[key]
        new_val = new_stats[key]
        diff = new_val - old_val
        pct = pct_change(old_val, new_val)

        if key == 'n_samples':
            print(f"  {'样本数':<12} {old_val:>12.0f} {new_val:>12.0f} {diff:>+12.0f} {pct:>+9.1f}%")
        else:
            # 对于误差指标，下降是好事 (绿色); 对于 KDE NLL 下降也是好事
            indicator = '✓' if diff < 0 else '✗' if diff > 0 else '-'
            print(f"  {key:<12} {old_val:>12.4f} {new_val:>12.4f} {diff:>+12.4f} {pct:>+9.1f}% {indicator}")

    if old_thresh and new_thresh:
        print(f"\n  阈值分析:")
        print(f"  {'阈值':<12} {'Epoch 20':>12} {'Epoch 100':>12} {'变化':>12}")
        print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
        for t in sorted(old_thresh.keys()):
            if t in new_thresh:
                old_pct = old_thresh[t]
                new_pct = new_thresh[t]
                diff = new_pct - old_pct
                unit = 'm'
                indicator = '✓' if diff < 0 else '✗' if diff > 0 else '-'
                print(f"  > {t}{unit:<9} {old_pct:>11.1f}% {new_pct:>11.1f}% {diff:>+11.1f}% {indicator}")


def print_fix_verification(new_stats_ade, new_stats_fde, new_stats_kde):
    """验证三项修复是否生效"""
    print(f"\n{'='*72}")
    print(f"  修复效果验证")
    print(f"{'='*72}")

    # 修复 1: PEDESTRIAN standardization
    # 这个需要 PEDESTRIAN 的评估结果，如果没有则标记为待验证
    print(f"\n  [修复 1] PEDESTRIAN Standardization (std: 1 → 80/40)")
    print(f"    状态: 需要 PEDESTRIAN 评估结果来验证 (当前 CSV 仅包含 VEHICLE)")
    print(f"    间接验证: 如果训练日志中 PEDESTRIAN loss 从 ~10 降至 0 附近，则修复成功")

    # 修复 2: Norm-based clipping (消除 √2)
    print(f"\n  [修复 2] Norm-based Clipping (消除 √2 放大)")
    old_max_ade = BASELINE['ade']['max']
    new_max_ade = new_stats_ade['max']
    old_max_fde = BASELINE['fde']['max']
    new_max_fde = new_stats_fde['max']
    print(f"    ADE max: {old_max_ade:.3f}m → {new_max_ade:.3f}m")
    print(f"    FDE max: {old_max_fde:.3f}m → {new_max_fde:.3f}m")
    if new_max_ade < old_max_ade:
        print(f"    ✓ 极端异常值减少 (ADE max 下降 {old_max_ade - new_max_ade:.1f}m)")
    else:
        print(f"    ✗ 极端异常值未减少，可能 clipping 修复未影响 VEHICLE 结果")
    print(f"    注意: 更精确的验证需运行 validate_data.py 检查 vel max ≤ 30 m/s, acc max ≤ 8 m/s²")

    # 修复 3: 增加训练 Epoch
    print(f"\n  [修复 3] 训练 Epoch 增加 (20 → 100)")
    ade_improvement = pct_change(BASELINE['ade']['mean'], new_stats_ade['mean'])
    fde_improvement = pct_change(BASELINE['fde']['mean'], new_stats_fde['mean'])
    kde_improvement = pct_change(BASELINE['kde']['mean'], new_stats_kde['mean'])
    print(f"    ADE mean: {BASELINE['ade']['mean']:.4f} → {new_stats_ade['mean']:.4f} ({ade_improvement:+.1f}%)")
    print(f"    FDE mean: {BASELINE['fde']['mean']:.4f} → {new_stats_fde['mean']:.4f} ({fde_improvement:+.1f}%)")
    print(f"    KDE mean: {BASELINE['kde']['mean']:.4f} → {new_stats_kde['mean']:.4f} ({kde_improvement:+.1f}%)")

    if ade_improvement < 0 and fde_improvement < 0:
        print(f"    ✓ 整体指标显著改善")
    elif ade_improvement < 0 or fde_improvement < 0:
        print(f"    ~ 部分指标改善，部分未改善")
    else:
        print(f"    ✗ 指标未改善，需进一步排查")


def print_summary(new_stats_ade, new_stats_fde, new_stats_kde):
    """打印总结"""
    print(f"\n{'='*72}")
    print(f"  总结")
    print(f"{'='*72}")

    ade_pct = pct_change(BASELINE['ade']['mean'], new_stats_ade['mean'])
    fde_pct = pct_change(BASELINE['fde']['mean'], new_stats_fde['mean'])
    kde_pct = pct_change(BASELINE['kde']['mean'], new_stats_kde['mean'])

    print(f"\n  Epoch 20 → Epoch 100 综合变化:")
    print(f"    ADE mean: {BASELINE['ade']['mean']:.3f}m → {new_stats_ade['mean']:.3f}m ({ade_pct:+.1f}%)")
    print(f"    FDE mean: {BASELINE['fde']['mean']:.3f}m → {new_stats_fde['mean']:.3f}m ({fde_pct:+.1f}%)")
    print(f"    KDE NLL:  {BASELINE['kde']['mean']:.3f} → {new_stats_kde['mean']:.3f} ({kde_pct:+.1f}%)")

    print(f"\n  EXPERIMENT_LOG.md 中的预期 (Section 8):")
    print(f"    - velocity max ≤ 30 m/s (不再出现 42 m/s) → 需 validate_data.py 验证")
    print(f"    - acceleration max ≤ 8 m/s² (不再出现 11.3 m/s²) → 需 validate_data.py 验证")
    print(f"    - VEHICLE ADE/FDE 略有改善 → ADE {ade_pct:+.1f}%, FDE {fde_pct:+.1f}%")
    print(f"    - PEDESTRIAN loss 从 ~10 降至 0 附近 → 需查看训练日志")

    # 下一步建议
    print(f"\n  下一步建议:")
    if new_stats_ade['max'] > 10:
        print(f"    1. ADE max = {new_stats_ade['max']:.1f}m 仍然较高，建议检查这些极端样本:")
        print(f"       - 可能是追踪 ID 切换导致的轨迹跳变")
        print(f"       - 可能是转弯/变道场景预测不准")
        print(f"       - 用 visualize.py 可视化这些 worst-case 场景")
    if new_stats_ade['mean'] > 0.5:
        print(f"    2. 可考虑的进一步优化:")
        print(f"       - 增大网络容量 (enc_rnn_dim, dec_rnn_dim)")
        print(f"       - 增加 GMM components (当前=1)")
        print(f"       - 调整 KL weight / min 参数")
        print(f"       - 数据层面: 过滤低质量轨迹, 检查标注一致性")
    print()


def main():
    parser = argparse.ArgumentParser(description='对比 Epoch 20 vs Epoch 100 评估结果')
    parser.add_argument('--results-dir', default='results', help='结果 CSV 目录')
    args = parser.parse_args()

    results_dir = args.results_dir

    # 查找 most_likely_z 版本的 CSV (与 baseline 对应)
    ade_path = os.path.join(results_dir, 'car_road_6_ade_most_likely_z.csv')
    fde_path = os.path.join(results_dir, 'car_road_6_fde_most_likely_z.csv')
    kde_path = os.path.join(results_dir, 'car_road_6_kde_full.csv')

    missing = []
    for name, path in [('ADE', ade_path), ('FDE', fde_path), ('KDE', kde_path)]:
        if not os.path.exists(path):
            missing.append(f"{name}: {path}")

    if missing:
        print("错误: 缺少以下文件:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    # 读取新结果
    ade_values = read_csv_values(ade_path)
    fde_values = read_csv_values(fde_path)
    kde_values = read_csv_values(kde_path)

    # 计算统计量
    new_ade = compute_stats(ade_values)
    new_fde = compute_stats(fde_values)
    new_kde = compute_stats(kde_values)

    # ADE 阈值
    ade_thresholds = [0.5, 1, 2, 3, 5, 10, 20]
    new_ade_thresh = compute_thresholds(ade_values, ade_thresholds)
    old_ade_thresh = BASELINE['ade'].get('thresholds', {})

    # FDE 阈值
    fde_thresholds = [1, 2, 5, 10, 20]
    new_fde_thresh = compute_thresholds(fde_values, fde_thresholds)
    old_fde_thresh = BASELINE['fde'].get('thresholds', {})

    print(f"\n{'#'*72}")
    print(f"#  Trajectron++ Car-Road: Epoch 20 vs Epoch 100 对比分析")
    print(f"#  Baseline: Epoch 20 (修复前, EXPERIMENT_LOG.md)")
    print(f"#  New:      Epoch 100 (修复后, {results_dir}/)")
    print(f"{'#'*72}")

    # 打印对比
    print_comparison('ADE (Most Likely Z)', BASELINE['ade'], new_ade, old_ade_thresh, new_ade_thresh)
    print_comparison('FDE (Most Likely Z)', BASELINE['fde'], new_fde, old_fde_thresh, new_fde_thresh)
    print_comparison('KDE NLL (Full)', BASELINE['kde'], new_kde)

    # 验证修复效果
    print_fix_verification(new_ade, new_fde, new_kde)

    # 总结
    print_summary(new_ade, new_fde, new_kde)


if __name__ == '__main__':
    main()
