"""
Data processing script for car-road dataset -> Trajectron++ format.

Converts roadside 3D bounding box annotations into the Environment/Scene/Node
pickle format expected by Trajectron++.

Usage:
    python process_data.py --data /mnt/car_road_data_fix \
                           --output_path ../processed \
                           --label_type interpolation \
                           --val_split 0.15 \
                           --test_split 0.15

The script reads roadside labels (JSON files with 3D bounding boxes in
the virtual LiDAR coordinate frame), extracts (x, y, yaw) trajectories
per tracked object, and generates Trajectron++-compatible pickle files.
"""

import sys
import os
import glob
import json
import numpy as np
import pandas as pd
import dill
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split

sys.path.append("../../trajectron")
from environment import Environment, Scene, Node, derivative_of

# ---------- label name -> Trajectron++ node type mapping ----------
VEHICLE_LABELS = {
    'Car', 'Suv', 'Bus', 'Truck', 'Van',
    'car', 'suv', 'bus', 'truck', 'van',
    'Engineering_vehicle', 'engineering_vehicle',
    'Fire_engine', 'fire_engine',
    'Trailer', 'trailer',
    'Vehicle_else', 'vehicle_else',
    'Huge_vehicle', 'huge_vehicle',
}
PEDESTRIAN_LABELS = {
    'Pedestrian', 'pedestrian',
    'Pedestrian_else', 'pedestrian_else',
    'Person', 'person',
}
CYCLIST_LABELS = {
    'Cyclist', 'cyclist',
    'Bicycle', 'bicycle',
    'Motorcycle', 'motorcycle',
    'Tricycle', 'tricycle',
    'Non_motor_rider', 'non_motor_rider',
    'Motor_rider', 'motor_rider',
    'Other_rider', 'other_rider',
}

# ---------- data columns (matching nuScenes format) ----------
data_columns_vehicle = pd.MultiIndex.from_product(
    [['position', 'velocity', 'acceleration', 'heading'], ['x', 'y']]
)
data_columns_vehicle = data_columns_vehicle.append(
    pd.MultiIndex.from_tuples([('heading', '°'), ('heading', 'd°')])
)
data_columns_vehicle = data_columns_vehicle.append(
    pd.MultiIndex.from_product([['velocity', 'acceleration'], ['norm']])
)

data_columns_pedestrian = pd.MultiIndex.from_product(
    [['position', 'velocity', 'acceleration'], ['x', 'y']]
)

# ---------- standardization parameters ----------
standardization = {
    'PEDESTRIAN': {
        'position': {
            'x': {'mean': 0, 'std': 80},
            'y': {'mean': 0, 'std': 40}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 2},
            'y': {'mean': 0, 'std': 2}
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        }
    },
    'VEHICLE': {
        'position': {
            'x': {'mean': 0, 'std': 120},
            'y': {'mean': 0, 'std': 60}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 15},
            'y': {'mean': 0, 'std': 15},
            'norm': {'mean': 0, 'std': 15}
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 4},
            'y': {'mean': 0, 'std': 4},
            'norm': {'mean': 0, 'std': 4}
        },
        'heading': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1},
            '°': {'mean': 0, 'std': np.pi},
            'd°': {'mean': 0, 'std': 1}
        }
    }
}

# ---------- trajectory curvature (for frequency weighting) ----------
curv_0_2 = 0
curv_0_1 = 0
total = 0


def trajectory_curvature(t):
    path_distance = np.linalg.norm(t[-1] - t[0])
    lengths = np.sqrt(np.sum(np.diff(t, axis=0) ** 2, axis=1))
    path_length = np.sum(lengths)
    if np.isclose(path_distance, 0.):
        return 0, 0, 0
    return (path_length / path_distance) - 1, path_length, path_distance


def classify_label(label_str):
    """Map a label string to a Trajectron++ node type string."""
    if label_str in VEHICLE_LABELS:
        return 'VEHICLE'
    elif label_str in PEDESTRIAN_LABELS:
        return 'PEDESTRIAN'
    elif label_str in CYCLIST_LABELS:
        # Treat cyclists as pedestrians for trajectory prediction
        return 'PEDESTRIAN'
    else:
        return None


def load_scene_labels(scene_dir, label_type='interpolation'):
    """
    Load all label JSON files for a scene from road_labels/.

    Parameters
    ----------
    scene_dir : str
        Path to the scene folder (e.g., /mnt/car_road_data_fix/001_car0325_road0327_t1)
    label_type : str
        'interpolation' to use interpolation_labels, 'ori' to use ori_labels

    Returns
    -------
    list of dict
        Each dict is one frame's label file contents, sorted by timestamp.
    """
    if label_type == 'interpolation':
        label_dir = os.path.join(scene_dir, 'road_labels', 'interpolation_labels')
    else:
        label_dir = os.path.join(scene_dir, 'road_labels', 'ori_labels')

    if not os.path.isdir(label_dir):
        return []

    json_files = sorted(glob.glob(os.path.join(label_dir, '*.json')))
    if not json_files:
        return []

    frames = []
    for jf in json_files:
        with open(jf, 'r') as f:
            try:
                data = json.load(f)
                frames.append(data)
            except json.JSONDecodeError:
                continue

    # Sort by timestamp
    frames.sort(key=lambda x: int(x.get('timestamp', 0)))
    return frames


def estimate_dt(frames):
    """Estimate the time step (dt) from consecutive timestamps."""
    if len(frames) < 2:
        return 0.1  # default fallback

    timestamps = [int(f['timestamp']) for f in frames]
    diffs = np.diff(timestamps)
    # timestamps are in milliseconds
    median_diff_ms = np.median(diffs)
    dt = median_diff_ms / 1000.0  # convert to seconds

    if dt <= 0:
        dt = 0.1

    return dt


def augment_scene(scene, angle):
    """Create a rotated copy of the scene for data augmentation."""
    def rotate_pc(pc, alpha):
        M = np.array([[np.cos(alpha), -np.sin(alpha)],
                      [np.sin(alpha), np.cos(alpha)]])
        return M @ pc

    scene_aug = Scene(timesteps=scene.timesteps, dt=scene.dt,
                      name=scene.name, non_aug_scene=scene)
    alpha = angle * np.pi / 180

    for node in scene.nodes:
        if node.type == 'PEDESTRIAN':
            x = node.data.position.x.copy()
            y = node.data.position.y.copy()
            x, y = rotate_pc(np.array([x, y]), alpha)

            vx = derivative_of(x, scene.dt)
            vy = derivative_of(y, scene.dt)
            ax = derivative_of(vx, scene.dt)
            ay = derivative_of(vy, scene.dt)

            data_dict = {('position', 'x'): x,
                         ('position', 'y'): y,
                         ('velocity', 'x'): vx,
                         ('velocity', 'y'): vy,
                         ('acceleration', 'x'): ax,
                         ('acceleration', 'y'): ay}
            node_data = pd.DataFrame(data_dict, columns=data_columns_pedestrian)
            new_node = Node(node_type=node.type, node_id=node.id,
                            data=node_data, first_timestep=node.first_timestep)
        elif node.type == 'VEHICLE':
            x = node.data.position.x.copy()
            y = node.data.position.y.copy()
            heading = getattr(node.data.heading, '°').copy()
            heading += alpha
            heading = (heading + np.pi) % (2.0 * np.pi) - np.pi

            x, y = rotate_pc(np.array([x, y]), alpha)

            vx = derivative_of(x, scene.dt)
            vy = derivative_of(y, scene.dt)
            ax = derivative_of(vx, scene.dt)
            ay = derivative_of(vy, scene.dt)

            v = np.stack((vx, vy), axis=-1)
            v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
            heading_v = np.divide(v, v_norm, out=np.zeros_like(v),
                                  where=(v_norm > 1.))
            heading_x = heading_v[:, 0]
            heading_y = heading_v[:, 1]

            data_dict = {('position', 'x'): x,
                         ('position', 'y'): y,
                         ('velocity', 'x'): vx,
                         ('velocity', 'y'): vy,
                         ('velocity', 'norm'): np.linalg.norm(v, axis=-1),
                         ('acceleration', 'x'): ax,
                         ('acceleration', 'y'): ay,
                         ('acceleration', 'norm'): np.linalg.norm(
                             np.stack((ax, ay), axis=-1), axis=-1),
                         ('heading', 'x'): heading_x,
                         ('heading', 'y'): heading_y,
                         ('heading', '°'): heading,
                         ('heading', 'd°'): derivative_of(heading, scene.dt,
                                                          radian=True)}
            node_data = pd.DataFrame(data_dict, columns=data_columns_vehicle)
            new_node = Node(node_type=node.type, node_id=node.id,
                            data=node_data, first_timestep=node.first_timestep,
                            non_aug_node=node)
        else:
            new_node = node

        scene_aug.nodes.append(new_node)
    return scene_aug


def augment(scene):
    """Random augmentation function used at training time."""
    scene_aug = np.random.choice(scene.augmented)
    scene_aug.temporal_scene_graph = scene.temporal_scene_graph
    scene_aug.map = scene.map
    return scene_aug


def process_scene(scene_dir, env, label_type='interpolation',
                  min_track_length=2, resample_dt=None):
    """
    Process a single scene folder into a Trajectron++ Scene object.

    Parameters
    ----------
    scene_dir : str
        Path to the scene folder.
    env : Environment
        The Trajectron++ environment (provides NodeType enum).
    label_type : str
        Which label folder to use.
    min_track_length : int
        Minimum number of frames for a track to be included.
    resample_dt : float or None
        If set, resample trajectories to this time step.
        If None, use the raw dt from timestamps.

    Returns
    -------
    Scene or None
    """
    global total, curv_0_2, curv_0_1

    scene_name = os.path.basename(scene_dir)
    frames = load_scene_labels(scene_dir, label_type)

    if len(frames) < 2:
        print(f'  Skipping {scene_name}: too few frames ({len(frames)})')
        return None

    raw_dt = estimate_dt(frames)
    dt = resample_dt if resample_dt is not None else raw_dt

    # If resampling is needed, subsample frames
    if resample_dt is not None and resample_dt > raw_dt * 1.5:
        step = max(1, int(round(resample_dt / raw_dt)))
        frames = frames[::step]
        if len(frames) < 2:
            print(f'  Skipping {scene_name}: too few frames after resample')
            return None

    # Build a DataFrame: frame_id, node_id, type, x, y, z, heading, l, w, h
    records = []
    for frame_idx, frame in enumerate(frames):
        objects = frame.get('object', [])
        for obj in objects:
            obj_label = obj.get('label', '')
            node_type_str = classify_label(obj_label)
            if node_type_str is None:
                continue

            records.append({
                'frame_id': frame_idx,
                'node_id': str(obj['id']),
                'type': node_type_str,
                'x': float(obj['x']),
                'y': float(obj['y']),
                'z': float(obj['z']),
                'heading': float(obj.get('yaw', 0.0)),
                'length': float(obj.get('length', 4.0)),
                'width': float(obj.get('width', 2.0)),
                'height': float(obj.get('height', 1.5)),
            })

    if len(records) == 0:
        print(f'  Skipping {scene_name}: no valid objects')
        return None

    data = pd.DataFrame(records)
    data.sort_values('frame_id', inplace=True)
    max_timesteps = int(data['frame_id'].max())

    # Center positions around scene mean to improve numerical stability
    x_mean = data['x'].mean()
    y_mean = data['y'].mean()
    data['x'] = data['x'] - x_mean
    data['y'] = data['y'] - y_mean

    scene = Scene(timesteps=max_timesteps + 1, dt=dt, name=scene_name,
                  aug_func=augment)

    for node_id in pd.unique(data['node_id']):
        node_df = data[data['node_id'] == node_id].sort_values('frame_id')

        if len(node_df) < min_track_length:
            continue

        # Check for continuous frames (no gaps)
        frame_ids = node_df['frame_id'].values
        if not np.all(np.diff(frame_ids) == 1):
            # Split into continuous segments and use the longest one
            diffs = np.diff(frame_ids)
            split_points = np.where(diffs != 1)[0] + 1
            segments = np.split(np.arange(len(node_df)), split_points)
            # Find longest continuous segment
            best_seg = max(segments, key=len)
            if len(best_seg) < min_track_length:
                continue
            node_df = node_df.iloc[best_seg]
            frame_ids = node_df['frame_id'].values

        node_type_str = node_df.iloc[0]['type']

        x = node_df['x'].values.astype(np.float64)
        y = node_df['y'].values.astype(np.float64)
        heading = node_df['heading'].values.astype(np.float64)

        node_frequency_multiplier = 1

        if node_type_str == 'VEHICLE':
            # Trajectory curvature for frequency weighting
            curvature, pl, _ = trajectory_curvature(
                np.stack((x, y), axis=-1))
            if pl < 1.0:
                # Vehicle is essentially stationary
                x = np.full(max_timesteps + 1, x[0])
                y = np.full(max_timesteps + 1, y[0])
                heading = np.full(max_timesteps + 1, heading[0])

            total += 1
            if pl > 1.0:
                if curvature > 0.2:
                    curv_0_2 += 1
                    node_frequency_multiplier = 3 * int(
                        np.floor(total / max(curv_0_2, 1)))
                elif curvature > 0.1:
                    curv_0_1 += 1
                    node_frequency_multiplier = 3 * int(
                        np.floor(total / max(curv_0_1, 1)))

        # Compute derivatives
        vx = derivative_of(x, dt)
        vy = derivative_of(y, dt)
        ax = derivative_of(vx, dt)
        ay = derivative_of(vy, dt)

        # Clip extreme velocity/acceleration by norm (not per-component)
        # to avoid sqrt(2) factor on diagonal. Limits: 30 m/s, 8 m/s²
        max_vel = 30.0 if node_type_str == 'VEHICLE' else 10.0
        max_acc = 8.0 if node_type_str == 'VEHICLE' else 5.0
        v_norm = np.sqrt(vx**2 + vy**2)
        v_scale = np.where(v_norm > max_vel, max_vel / np.maximum(v_norm, 1e-8), 1.0)
        vx = vx * v_scale
        vy = vy * v_scale
        a_norm = np.sqrt(ax**2 + ay**2)
        a_scale = np.where(a_norm > max_acc, max_acc / np.maximum(a_norm, 1e-8), 1.0)
        ax = ax * a_scale
        ay = ay * a_scale

        if node_type_str == 'VEHICLE':
            v = np.stack((vx, vy), axis=-1)
            v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
            heading_v = np.divide(v, v_norm, out=np.zeros_like(v),
                                  where=(v_norm > 1.))
            heading_x = heading_v[:, 0]
            heading_y = heading_v[:, 1]

            d_heading = derivative_of(heading, dt, radian=True)
            d_heading = np.clip(d_heading, -np.pi, np.pi)

            data_dict = {
                ('position', 'x'): x,
                ('position', 'y'): y,
                ('velocity', 'x'): vx,
                ('velocity', 'y'): vy,
                ('velocity', 'norm'): np.linalg.norm(v, axis=-1),
                ('acceleration', 'x'): ax,
                ('acceleration', 'y'): ay,
                ('acceleration', 'norm'): np.linalg.norm(
                    np.stack((ax, ay), axis=-1), axis=-1),
                ('heading', 'x'): heading_x,
                ('heading', 'y'): heading_y,
                ('heading', '°'): heading,
                ('heading', 'd°'): d_heading
            }
            node_data = pd.DataFrame(data_dict,
                                     columns=data_columns_vehicle)
            node = Node(
                node_type=env.NodeType.VEHICLE,
                node_id=node_id,
                data=node_data,
                length=node_df.iloc[0]['length'],
                width=node_df.iloc[0]['width'],
                height=node_df.iloc[0]['height'],
                frequency_multiplier=node_frequency_multiplier
            )
        else:
            data_dict = {
                ('position', 'x'): x,
                ('position', 'y'): y,
                ('velocity', 'x'): vx,
                ('velocity', 'y'): vy,
                ('acceleration', 'x'): ax,
                ('acceleration', 'y'): ay
            }
            node_data = pd.DataFrame(data_dict,
                                     columns=data_columns_pedestrian)
            node = Node(
                node_type=env.NodeType.PEDESTRIAN,
                node_id=node_id,
                data=node_data,
                frequency_multiplier=node_frequency_multiplier
            )

        node.first_timestep = int(frame_ids[0])
        scene.nodes.append(node)

    if len(scene.nodes) == 0:
        print(f'  Skipping {scene_name}: no valid nodes')
        return None

    return scene


def process_data(data_path, output_path, label_type, val_split, test_split,
                 resample_dt, min_track_length, node_types):
    """Main processing entry point."""
    global total, curv_0_2, curv_0_1

    os.makedirs(output_path, exist_ok=True)

    # Discover all scene folders
    scene_dirs = sorted([
        os.path.join(data_path, d)
        for d in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, d))
    ])

    if not scene_dirs:
        print(f'No scene directories found in {data_path}')
        return

    print(f'Found {len(scene_dirs)} scene directories')

    # Split into train / val / test
    if val_split + test_split >= 1.0:
        raise ValueError('val_split + test_split must be < 1.0')

    train_dirs, test_dirs = train_test_split(
        scene_dirs, test_size=test_split, random_state=42
    )
    if val_split > 0:
        adjusted_val = val_split / (1.0 - test_split)
        train_dirs, val_dirs = train_test_split(
            train_dirs, test_size=adjusted_val, random_state=42
        )
    else:
        val_dirs = []

    split_dirs = {
        'train': train_dirs,
        'val': val_dirs,
        'test': test_dirs
    }

    print(f'Split: train={len(train_dirs)}, val={len(val_dirs)}, '
          f'test={len(test_dirs)}')

    for data_class in ['train', 'val', 'test']:
        if not split_dirs[data_class]:
            print(f'No scenes for {data_class}, skipping.')
            continue

        env = Environment(node_type_list=node_types,
                          standardization=standardization)
        attention_radius = dict()
        if 'PEDESTRIAN' in node_types and 'VEHICLE' in node_types:
            attention_radius[
                (env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 10.0
            attention_radius[
                (env.NodeType.PEDESTRIAN, env.NodeType.VEHICLE)] = 20.0
            attention_radius[
                (env.NodeType.VEHICLE, env.NodeType.PEDESTRIAN)] = 20.0
            attention_radius[
                (env.NodeType.VEHICLE, env.NodeType.VEHICLE)] = 30.0
        elif 'VEHICLE' in node_types:
            attention_radius[
                (env.NodeType.VEHICLE, env.NodeType.VEHICLE)] = 30.0
        elif 'PEDESTRIAN' in node_types:
            attention_radius[
                (env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 10.0

        env.attention_radius = attention_radius

        scenes = []
        for scene_dir in tqdm(split_dirs[data_class],
                              desc=f'Processing {data_class}'):
            scene = process_scene(
                scene_dir, env,
                label_type=label_type,
                min_track_length=min_track_length,
                resample_dt=resample_dt
            )
            if scene is not None:
                if data_class == 'train':
                    scene.augmented = list()
                    angles = np.arange(0, 360, 15)
                    for angle in angles:
                        scene.augmented.append(augment_scene(scene, angle))
                scenes.append(scene)

        print(f'Processed {len(scenes)} scenes for {data_class}')

        env.scenes = scenes

        if len(scenes) > 0:
            pkl_name = f'car_road_{data_class}_full.pkl'
            data_dict_path = os.path.join(output_path, pkl_name)
            with open(data_dict_path, 'wb') as f:
                dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)
            print(f'Saved {data_dict_path}')

        print(f"  Total vehicle nodes: {total}")
        print(f"  Curvature > 0.1: {curv_0_1}")
        print(f"  Curvature > 0.2: {curv_0_2}")
        total = 0
        curv_0_1 = 0
        curv_0_2 = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process car-road dataset for Trajectron++')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to car_road_data_fix root directory')
    parser.add_argument('--output_path', type=str, default='../processed',
                        help='Output directory for pickle files')
    parser.add_argument('--label_type', type=str, default='interpolation',
                        choices=['interpolation', 'ori'],
                        help='Which label folder to use')
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='Fraction of scenes for validation')
    parser.add_argument('--test_split', type=float, default=0.15,
                        help='Fraction of scenes for testing')
    parser.add_argument('--resample_dt', type=float, default=None,
                        help='Resample to this time step (seconds). '
                             'If not set, uses raw dt from timestamps.')
    parser.add_argument('--min_track_length', type=int, default=2,
                        help='Minimum number of frames for a valid track')
    parser.add_argument('--node_types', nargs='+',
                        default=['VEHICLE', 'PEDESTRIAN'],
                        help='Node types to include')
    args = parser.parse_args()

    process_data(
        data_path=args.data,
        output_path=args.output_path,
        label_type=args.label_type,
        val_split=args.val_split,
        test_split=args.test_split,
        resample_dt=args.resample_dt,
        min_track_length=args.min_track_length,
        node_types=args.node_types
    )
