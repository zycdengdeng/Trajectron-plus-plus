import sys
import os
import dill
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../../trajectron')

# Load test data
with open('../processed/car_road_test_full.pkl', 'rb') as f:
    env = dill.load(f, encoding='latin1')

os.makedirs('results', exist_ok=True)

# Visualize first few scenes
for i, scene in enumerate(env.scenes[:3]):
    print(f"Scene {i}: {scene.name}, timesteps={scene.timesteps}")

    fig, ax = plt.subplots(figsize=(10, 10))

    for node in scene.nodes:
        xy = node.data.data[:, 0:2]
        if node.type.name == 'VEHICLE':
            ax.plot(xy[:, 0], xy[:, 1], 'b-', alpha=0.3, linewidth=1)
            ax.plot(xy[-1, 0], xy[-1, 1], 'bo', markersize=3)
        elif node.type.name == 'PEDESTRIAN':
            ax.plot(xy[:, 0], xy[:, 1], 'r-', alpha=0.3, linewidth=1)
            ax.plot(xy[-1, 0], xy[-1, 1], 'ro', markersize=3)

    ax.set_title(f'Scene: {scene.name}')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='b', label='Vehicle'),
        Line2D([0], [0], color='r', label='Pedestrian'),
    ]
    ax.legend(handles=legend_elements)

    plt.tight_layout()
    plt.savefig(f'results/scene_{i}_trajectories.png', dpi=150)
    print(f"  Saved to results/scene_{i}_trajectories.png")

plt.close('all')
print("Done!")
