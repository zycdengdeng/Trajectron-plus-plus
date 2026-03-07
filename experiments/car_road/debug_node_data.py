"""调试: 检查 node.data 的实际结构"""
import sys, os, dill
import numpy as np
sys.path.append("../../trajectron")

with open('../processed/car_road_test_full.pkl', 'rb') as f:
    env = dill.load(f)

scene = env.scenes[0]
node = [n for n in scene.nodes if n.type.name == 'VEHICLE'][0]

print(f"Node type: {node.type}")
print(f"Node data type: {type(node.data)}")
print(f"Node data.data shape: {node.data.data.shape}")
print(f"Node data.data dtype: {node.data.data.dtype}")

# Check headers
if hasattr(node.data, 'header'):
    print(f"Headers: {node.data.header}")
if hasattr(node.data, 'columns'):
    print(f"Columns: {node.data.columns}")

# Try different access methods
print("\n--- Trying access methods ---")
try:
    val = node.data[:, ('velocity', 'x')]
    print(f"node.data[:, ('velocity', 'x')] = {val[:3]} (shape {val.shape})")
except Exception as e:
    print(f"node.data[:, ('velocity', 'x')] FAILED: {e}")

try:
    val = node.data[:, ('position', 'x')]
    print(f"node.data[:, ('position', 'x')] = {val[:3]} (shape {val.shape})")
except Exception as e:
    print(f"node.data[:, ('position', 'x')] FAILED: {e}")

# Try raw numpy access
print(f"\nRaw data first row: {node.data.data[0]}")
print(f"Raw data shape: {node.data.data.shape}")

# Check all available attributes
print(f"\nDir of node.data: {[a for a in dir(node.data) if not a.startswith('_')]}")
