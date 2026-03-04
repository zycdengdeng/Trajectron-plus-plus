#!/bin/bash
# ============================================================
# Trajectron++ 部署脚本 (car-road 数据集)
#
# 在服务器上一键部署 Trajectron++ 并准备 car-road 数据管线
#
# 用法:
#   # 1. 先 cd 到你想要部署的目录
#   cd /mnt/zyc_wzh
#
#   # 2. 运行此脚本 (需要先把这个脚本拷贝过去, 或者直接按下面步骤手动执行)
#   bash deploy.sh
# ============================================================

set -e

DEPLOY_DIR="/mnt/zyc_wzh"
CONDA_ENV="da3"  # 你已有的 conda 环境

echo "========================================="
echo " Trajectron++ 部署 (car-road 数据集)"
echo "========================================="

# ----- Step 1: 克隆仓库 -----
echo ""
echo "[Step 1/4] 克隆仓库..."
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

# ----- Step 2: 安装依赖 -----
echo ""
echo "[Step 2/4] 安装 Python 依赖..."
echo "  当前 conda 环境: ${CONDA_ENV}"
echo ""
echo "  请确保你已激活 conda 环境: conda activate ${CONDA_ENV}"
echo "  以下依赖将通过 pip 安装:"
echo ""

pip install dill tqdm pandas numpy scipy scikit-learn \
    tensorboardX pyquaternion orjson ncls opencv-python

# PyTorch: 如果你的环境已经有 PyTorch 就跳过
python -c "import torch; print(f'  PyTorch 已安装: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null || {
    echo "  警告: 未检测到 PyTorch, 请手动安装适合你 CUDA 版本的 PyTorch"
    echo "  例如: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
}

# ----- Step 3: 创建输出目录 -----
echo ""
echo "[Step 3/4] 创建输出目录..."
mkdir -p experiments/processed
mkdir -p experiments/car_road/logs
mkdir -p experiments/car_road/results

# ----- Step 4: 验证 -----
echo ""
echo "[Step 4/4] 验证安装..."
cd "${DEPLOY_DIR}/Trajectron-plus-plus"
python -c "
import sys
sys.path.append('trajectron')
from environment import Environment, Scene, Node, derivative_of
import dill
import torch
import numpy as np
import pandas as pd
print('  ✓ 所有核心依赖导入成功')
print(f'  ✓ PyTorch: {torch.__version__}')
print(f'  ✓ CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  ✓ GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "========================================="
echo " 部署完成!"
echo "========================================="
echo ""
echo " 接下来的步骤:"
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
