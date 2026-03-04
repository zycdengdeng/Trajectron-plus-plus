"""Quick validation of processed pickle files."""
import sys
import os
import dill
import numpy as np

sys.path.append("../../trajectron")

processed_dir = "../processed"

for split in ['train', 'val', 'test']:
    path = os.path.join(processed_dir, f'car_road_{split}_full.pkl')
    if not os.path.exists(path):
        print(f"  {path} not found, skipping")
        continue
    with open(path, 'rb') as f:
        env = dill.load(f)

    print(f"\n=== {split} ===")
    print(f"  scenes: {len(env.scenes)}")

    total_nodes = {}
    dt_values = []
    track_lengths = []
    pos_x_all = []
    pos_y_all = []
    vel_all = []
    acc_all = []

    for scene in env.scenes:
        dt_values.append(scene.dt)
        for node in scene.nodes:
            nt = str(node.type)
            total_nodes[nt] = total_nodes.get(nt, 0) + 1
            track_lengths.append(node.data.data.shape[0])

            # DoubleHeaderNumpyArray: node.data[:, ('position', 'x')] returns np array
            x = node.data[:, ('position', 'x')]
            y = node.data[:, ('position', 'y')]
            pos_x_all.extend(x.flatten().tolist())
            pos_y_all.extend(y.flatten().tolist())

            vx = node.data[:, ('velocity', 'x')]
            vy = node.data[:, ('velocity', 'y')]
            vel_all.extend(np.sqrt(vx.flatten()**2 + vy.flatten()**2).tolist())

            ax = node.data[:, ('acceleration', 'x')]
            ay = node.data[:, ('acceleration', 'y')]
            acc_all.extend(np.sqrt(ax.flatten()**2 + ay.flatten()**2).tolist())

    for nt, cnt in sorted(total_nodes.items()):
        print(f"  {nt}: {cnt} nodes")
    print(f"  dt values: {np.unique(dt_values)}")
    print(f"  track length: min={np.min(track_lengths)}, "
          f"median={np.median(track_lengths):.0f}, "
          f"max={np.max(track_lengths)}")

    pos_x = np.array(pos_x_all)
    pos_y = np.array(pos_y_all)
    vel = np.array(vel_all)
    acc = np.array(acc_all)

    print(f"  position x: min={pos_x.min():.1f}, max={pos_x.max():.1f}, "
          f"std={pos_x.std():.1f}")
    print(f"  position y: min={pos_y.min():.1f}, max={pos_y.max():.1f}, "
          f"std={pos_y.std():.1f}")
    print(f"  velocity (norm): mean={vel.mean():.2f}, "
          f"p95={np.percentile(vel, 95):.2f}, max={vel.max():.2f} m/s")
    print(f"  acceleration (norm): mean={acc.mean():.2f}, "
          f"p95={np.percentile(acc, 95):.2f}, max={acc.max():.2f} m/s^2")

    # Check: what fraction of position values exceed current std=80?
    frac_x = np.mean(np.abs(pos_x) > 80) * 100
    frac_y = np.mean(np.abs(pos_y) > 80) * 100
    print(f"  |position_x| > 80 (current std): {frac_x:.1f}%")
    print(f"  |position_y| > 80 (current std): {frac_y:.1f}%")
