#!/bin/bash
# Trajectron++ 训练脚本 (car-road 数据集)
#
# 用法:
#   bash run_train.sh
#
# 训练之前请先运行 run_preprocess.sh 生成 pickle 数据文件

cd ../../trajectron

python train.py \
    --conf ../config/car_road.json \
    --data_dir ../experiments/processed \
    --train_data_dict car_road_train_full.pkl \
    --eval_data_dict car_road_val_full.pkl \
    --log_dir ../experiments/car_road/logs \
    --log_tag "_car_road" \
    --train_epochs 20 \
    --batch_size 4096 \
    --preprocess_workers 16 \
    --eval_every 10 \
    --save_every 10 \
    --device cuda:0 \
    --offline_scene_graph yes \
    --dynamic_edges yes \
    --edge_state_combine_method sum \
    --edge_influence_combine_method attention \
    --augment \
    --node_freq_mult_train
