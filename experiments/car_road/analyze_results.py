"""分析评估结果 CSV，输出详细统计信息"""
import csv
import numpy as np
import sys
import os

def read_csv_values(path):
    values = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            values.append(float(row['value']))
    return np.array(values)

def analyze(csv_path, metric_name):
    values = read_csv_values(csv_path)

    print(f"\n{'='*60}")
    print(f"  {metric_name.upper()}  ({len(values)} samples)")
    print(f"{'='*60}")

    print(f"  mean:   {values.mean():.4f}")
    print(f"  median: {np.median(values):.4f}")
    print(f"  std:    {values.std():.4f}")
    print(f"  min:    {values.min():.4f}")
    print(f"  max:    {values.max():.4f}")
    print(f"  p25:    {np.percentile(values, 25):.4f}")
    print(f"  p75:    {np.percentile(values, 75):.4f}")
    print(f"  p95:    {np.percentile(values, 95):.4f}")

    if metric_name in ('ade', 'fde'):
        print(f"\n  阈值分析:")
        for thresh in [0.5, 1, 2, 3, 5, 10, 20]:
            pct = (values > thresh).mean() * 100
            print(f"    > {thresh}m: {pct:.1f}%")

def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else 'results'

    files = {'ade': None, 'fde': None, 'kde': None}

    for f in sorted(os.listdir(results_dir)):
        if not f.endswith('.csv'):
            continue
        path = os.path.join(results_dir, f)
        if 'ade' in f:
            files['ade'] = path
        elif 'fde' in f:
            files['fde'] = path
        elif 'kde' in f:
            files['kde'] = path

    for metric, path in files.items():
        if path:
            analyze(path, metric)

    print()

if __name__ == '__main__':
    main()
