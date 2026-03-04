#!/bin/bash
# ============================================================
# Trajectron++ 部署脚本 (car-road 数据集)
#
# 用法:
#   cd /mnt/zyc_wzh
#   bash deploy.sh
#
# 或者按脚本内的步骤手动逐步执行
# ============================================================

set -e

DEPLOY_DIR="/mnt/zyc_wzh"
CONDA_ENV="trajectronpp"
PYTHON_VER="3.9"

echo "========================================="
echo " Trajectron++ 部署 (car-road 数据集)"
echo "========================================="

# ----- Step 1: 克隆仓库 -----
echo ""
echo "[Step 1/5] 克隆仓库..."
cd "${DEPLOY_DIR}"

if [ -d "Trajectron-plus-plus" ]; then
    echo "  仓库已存在, 拉取最新代码..."
    cd Trajectron-plus-plus
    git fetch origin claude/reproduce-car-road-dataset-EJALZ
    git checkout claude/reproduce-car-road-dataset-EJALZ
    git pull origin claude/reproduce-car-road-dataset-EJALZ
else
    git clone --recurse-submodules https://github.com/zycdengdeng/Trajectron-plus-plus.git
    cd Trajectron-plus-plus
    git checkout claude/reproduce-car-road-dataset-EJALZ
fi

# ----- Step 2: 创建 conda 环境 -----
echo ""
echo "[Step 2/5] 创建 conda 环境: ${CONDA_ENV} (Python ${PYTHON_VER})..."

# 检查环境是否已存在
if conda env list | grep -q "${CONDA_ENV}"; then
    echo "  环境 ${CONDA_ENV} 已存在, 跳过创建"
else
    conda create -n "${CONDA_ENV}" python="${PYTHON_VER}" -y
fi

# 激活环境
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"
echo "  当前 Python: $(python --version)"
echo "  当前环境: ${CONDA_ENV}"

# ----- Step 3: 安装依赖 -----
echo ""
echo "[Step 3/5] 安装 PyTorch (CUDA 12.1)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "[Step 3/5] 安装其他 Python 依赖..."
pip install \
    dill==0.3.7 \
    tqdm \
    pandas \
    numpy \
    scipy \
    scikit-learn \
    matplotlib \
    seaborn \
    tensorboardX \
    pyquaternion \
    orjson \
    ncls \
    opencv-python \
    notebook

# ----- Step 4: 创建输出目录 -----
echo ""
echo "[Step 4/5] 创建输出目录..."
cd "${DEPLOY_DIR}/Trajectron-plus-plus"
mkdir -p experiments/processed
mkdir -p experiments/car_road/logs
mkdir -p experiments/car_road/results

# ----- Step 5: 验证 -----
echo ""
echo "[Step 5/5] 验证安装..."
python -c "
import sys
sys.path.append('trajectron')
from environment import Environment, Scene, Node, derivative_of
import dill
import torch
import numpy as np
import pandas as pd
print('  所有核心依赖导入成功')
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
else:
    print('  警告: CUDA 不可用, 请检查 PyTorch 安装')
"

echo ""
echo "========================================="
echo " 部署完成!"
echo "========================================="
echo ""
echo " 后续使用时先激活环境:"
echo "    conda activate ${CONDA_ENV}"
echo ""
echo " 然后按顺序执行:"
echo ""
echo " 1. 数据预处理:"
echo "    cd ${DEPLOY_DIR}/Trajectron-plus-plus/experiments/car_road"
echo "    bash run_preprocess.sh"
echo ""
echo " 2. 训练模型:"
echo "    bash run_train.sh"
echo ""
echo " 3. 评估模型:"
echo "    bash run_eval.sh logs/models_xxx_car_road"
echo ""
