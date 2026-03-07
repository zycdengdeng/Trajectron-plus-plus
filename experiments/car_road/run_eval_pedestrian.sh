#!/bin/bash
# Trajectron++ PEDESTRIAN 评估脚本 (car-road 数据集)
#
# 用法:
#   bash run_eval_pedestrian.sh
#
# 评估模型: models_05_Mar_2026_16_25_06_car_road, checkpoint 80
# 节点类型: PEDESTRIAN

MODEL_DIR="logs/models_05_Mar_2026_16_25_06_car_road"
CHECKPOINT=80
DATA="../processed/car_road_test_full.pkl"

mkdir -p results

echo "===== 评估 PEDESTRIAN (checkpoint ${CHECKPOINT}) ====="

python evaluate.py \
    --model "${MODEL_DIR}" \
    --checkpoint ${CHECKPOINT} \
    --data "${DATA}" \
    --output_path results \
    --output_tag "car_road_ped" \
    --node_type PEDESTRIAN \
    --prediction_horizon 6
