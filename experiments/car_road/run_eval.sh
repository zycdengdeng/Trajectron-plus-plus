#!/bin/bash
# Trajectron++ 评估脚本 (car-road 数据集)
#
# 用法:
#   bash run_eval.sh <model_dir>
#
# 示例:
#   bash run_eval.sh logs/models_04_Mar_2026_10_00_00_car_road

MODEL_DIR=${1:?"请指定模型目录, 例如: logs/models_04_Mar_2026_10_00_00_car_road"}

mkdir -p results

python evaluate.py \
    --model "${MODEL_DIR}" \
    --checkpoint 100 \
    --data ../processed/car_road_test_full.pkl \
    --output_path results \
    --output_tag "car_road" \
    --node_type VEHICLE \
    --prediction_horizon 6
