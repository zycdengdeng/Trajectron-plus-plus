#!/bin/bash
# 数据预处理脚本
# 将 car-road 数据集转换为 Trajectron++ 所需的 pickle 格式
#
# 用法:
#   bash run_preprocess.sh
#
# 可调参数说明:
#   --data            数据集根目录
#   --output_path     输出 pickle 文件的目录
#   --label_type      使用哪种标注 (interpolation / ori)
#   --val_split       验证集比例
#   --test_split      测试集比例
#   --resample_dt     重采样时间步长(秒), 不设则使用原始时间戳间隔
#                     路侧 LiDAR 约 7-8Hz, 可以设 0.5 与 nuScenes 对齐
#   --min_track_length 最少连续帧数(低于此值的轨迹会被过滤)

DATA_ROOT="/mnt/car_road_data_fix"
OUTPUT_DIR="../processed"

python process_data.py \
    --data "${DATA_ROOT}" \
    --output_path "${OUTPUT_DIR}" \
    --label_type interpolation \
    --val_split 0.15 \
    --test_split 0.15 \
    --resample_dt 0.5 \
    --min_track_length 4 \
    --node_types VEHICLE PEDESTRIAN
