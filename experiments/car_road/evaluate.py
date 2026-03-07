"""
评估脚本: 在 car-road 测试集上计算 ADE / FDE 指标

用法:
    python evaluate.py \
        --model logs/models_xxx_car_road \
        --checkpoint 100 \
        --data ../processed/car_road_test_full.pkl \
        --output_path results \
        --output_tag car_road \
        --node_type VEHICLE \
        --prediction_horizon 6
"""
import sys
import os
import dill
import json
import argparse
import torch
import numpy as np
import pandas as pd

sys.path.append("../../trajectron")
from tqdm import tqdm
from model.model_registrar import ModelRegistrar
from model.trajectron import Trajectron
import evaluation
from utils import prediction_output_to_trajectories

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="模型目录路径", type=str)
parser.add_argument("--checkpoint", help="要评估的 checkpoint 编号", type=int)
parser.add_argument("--data", help="测试集 pickle 文件路径", type=str)
parser.add_argument("--output_path", help="输出 csv 文件的目录", type=str)
parser.add_argument("--output_tag", help="输出文件名标签", type=str)
parser.add_argument("--node_type", help="要评估的节点类型", type=str)
parser.add_argument("--prediction_horizon", nargs='+',
                    help="预测时间步数", type=int, default=None)
args = parser.parse_args()


def load_model(model_dir, env, ts=100):
    model_registrar = ModelRegistrar(model_dir, 'cpu')
    model_registrar.load_models(ts)
    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)

    trajectron = Trajectron(model_registrar, hyperparams, None, 'cpu')
    trajectron.set_environment(env)
    trajectron.set_annealing_params()
    return trajectron, hyperparams


def extract_speeds_from_predictions(prediction_output_dict, dt, max_hl, ph, node_type_str):
    """按照 compute_batch_statistics 的迭代顺序提取每个预测样本对应的速度"""
    (prediction_dict, _, _) = prediction_output_to_trajectories(
        prediction_output_dict, dt, max_hl, ph, prune_ph_to_future=False)

    speeds = []
    for t in prediction_dict.keys():
        for node in prediction_dict[t].keys():
            if node.type.name != node_type_str:
                continue
            # 获取该时刻的速度
            vx = node.data[:, ('velocity', 'x')].flatten()
            vy = node.data[:, ('velocity', 'y')].flatten()
            local_t = t - node.first_timestep
            if 0 <= local_t < len(vx):
                speed = np.sqrt(float(vx[local_t])**2 + float(vy[local_t])**2)
            else:
                speed = -1.0  # 标记异常
            speeds.append(speed)
    return speeds


if __name__ == "__main__":
    with open(args.data, 'rb') as f:
        env = dill.load(f, encoding='latin1')

    eval_stg, hyperparams = load_model(args.model, env, ts=args.checkpoint)

    if 'override_attention_radius' in hyperparams:
        for attention_radius_override in hyperparams['override_attention_radius']:
            node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
            env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

    scenes = env.scenes

    os.makedirs(args.output_path, exist_ok=True)

    print("-- 构建场景图")
    for scene in tqdm(scenes):
        scene.calculate_scene_graph(env.attention_radius,
                                    hyperparams['edge_addition_filter'],
                                    hyperparams['edge_removal_filter'])

    for ph in args.prediction_horizon:
        print(f"预测时域: {ph}")
        max_hl = hyperparams['maximum_history_length']

        with torch.no_grad():
            # ---------- Most Likely Z ----------
            eval_ade_batch_errors = np.array([])
            eval_fde_batch_errors = np.array([])
            eval_speeds = []

            print("-- 评估 GMM Z Mode (Most Likely)")
            for scene in tqdm(scenes):
                timesteps = np.arange(scene.timesteps)
                predictions = eval_stg.predict(scene,
                                               timesteps,
                                               ph,
                                               num_samples=1,
                                               min_future_timesteps=ph,
                                               z_mode=True,
                                               gmm_mode=True,
                                               full_dist=False)

                if not predictions:
                    continue

                batch_error_dict = evaluation.compute_batch_statistics(
                    predictions,
                    scene.dt,
                    max_hl=max_hl,
                    ph=ph,
                    node_type_enum=env.NodeType,
                    map=None,
                    prune_ph_to_future=False,
                    kde=False)

                # 提取速度 (与 ADE/FDE 精确对齐)
                speeds = extract_speeds_from_predictions(
                    predictions, scene.dt, max_hl, ph, args.node_type)
                eval_speeds.extend(speeds)

                if args.node_type in batch_error_dict:
                    eval_ade_batch_errors = np.hstack(
                        (eval_ade_batch_errors,
                         batch_error_dict[args.node_type]['ade']))
                    eval_fde_batch_errors = np.hstack(
                        (eval_fde_batch_errors,
                         batch_error_dict[args.node_type]['fde']))

            eval_speeds = np.array(eval_speeds)

            if len(eval_ade_batch_errors) > 0:
                print(f"  ML ADE: {np.mean(eval_ade_batch_errors):.4f}")
                print(f"  ML FDE: {np.mean(eval_fde_batch_errors):.4f}")

                # 按速度段统计
                if len(eval_speeds) == len(eval_ade_batch_errors):
                    print(f"\n  按速度段统计 (Most Likely Z):")
                    print(f"  {'速度段':<20} {'样本数':>8} {'ADE mean':>10} {'ADE med':>10} {'FDE mean':>10} {'FDE med':>10}")
                    for lo, hi, label in [(0, 0.5, '静止(<0.5)'),
                                          (0.5, 2, '低速(0.5-2)'),
                                          (2, 5, '中速(2-5)'),
                                          (5, 15, '正常(5-15)'),
                                          (15, 100, '高速(>15)')]:
                        mask = (eval_speeds >= lo) & (eval_speeds < hi)
                        n = mask.sum()
                        if n > 0:
                            a = eval_ade_batch_errors[mask]
                            f = eval_fde_batch_errors[mask]
                            print(f"  {label:<20} {n:>8} {a.mean():>10.3f} {np.median(a):>10.3f} {f.mean():>10.3f} {np.median(f):>10.3f}")
                    # 总体 (仅运动)
                    moving = eval_speeds >= 1.0
                    if moving.sum() > 0:
                        a = eval_ade_batch_errors[moving]
                        f = eval_fde_batch_errors[moving]
                        print(f"  {'运动(>=1.0)合计':<20} {moving.sum():>8} {a.mean():>10.3f} {np.median(a):>10.3f} {f.mean():>10.3f} {np.median(f):>10.3f}")

            pd.DataFrame(
                {'value': eval_ade_batch_errors, 'metric': 'ade', 'type': 'ml',
                 'speed': eval_speeds if len(eval_speeds) == len(eval_ade_batch_errors) else np.nan}
            ).to_csv(os.path.join(
                args.output_path,
                f"{args.output_tag}_{ph}_ade_most_likely_z.csv"))
            pd.DataFrame(
                {'value': eval_fde_batch_errors, 'metric': 'fde', 'type': 'ml',
                 'speed': eval_speeds if len(eval_speeds) == len(eval_fde_batch_errors) else np.nan}
            ).to_csv(os.path.join(
                args.output_path,
                f"{args.output_tag}_{ph}_fde_most_likely_z.csv"))

            # ---------- Full (多采样) ----------
            eval_ade_batch_errors = np.array([])
            eval_fde_batch_errors = np.array([])
            eval_kde_nll = np.array([])

            print("-- 评估 Full (2000 samples)")
            for scene in tqdm(scenes):
                timesteps = np.arange(scene.timesteps)
                predictions = eval_stg.predict(scene,
                                               timesteps,
                                               ph,
                                               num_samples=2000,
                                               min_future_timesteps=ph,
                                               z_mode=False,
                                               gmm_mode=False,
                                               full_dist=False)

                if not predictions:
                    continue

                batch_error_dict = evaluation.compute_batch_statistics(
                    predictions,
                    scene.dt,
                    max_hl=max_hl,
                    ph=ph,
                    node_type_enum=env.NodeType,
                    map=None,
                    prune_ph_to_future=False)

                if args.node_type in batch_error_dict:
                    eval_ade_batch_errors = np.hstack(
                        (eval_ade_batch_errors,
                         batch_error_dict[args.node_type]['ade']))
                    eval_fde_batch_errors = np.hstack(
                        (eval_fde_batch_errors,
                         batch_error_dict[args.node_type]['fde']))
                    eval_kde_nll = np.hstack(
                        (eval_kde_nll,
                         batch_error_dict[args.node_type]['kde']))

            if len(eval_ade_batch_errors) > 0:
                print(f"  Full ADE: {np.mean(eval_ade_batch_errors):.4f}")
                print(f"  Full FDE: {np.mean(eval_fde_batch_errors):.4f}")
                print(f"  KDE NLL:  {np.mean(eval_kde_nll):.4f}")

            pd.DataFrame(
                {'value': eval_ade_batch_errors, 'metric': 'ade', 'type': 'full'}
            ).to_csv(os.path.join(
                args.output_path,
                f"{args.output_tag}_{ph}_ade_full.csv"))
            pd.DataFrame(
                {'value': eval_fde_batch_errors, 'metric': 'fde', 'type': 'full'}
            ).to_csv(os.path.join(
                args.output_path,
                f"{args.output_tag}_{ph}_fde_full.csv"))
            pd.DataFrame(
                {'value': eval_kde_nll, 'metric': 'kde', 'type': 'full'}
            ).to_csv(os.path.join(
                args.output_path,
                f"{args.output_tag}_{ph}_kde_full.csv"))
