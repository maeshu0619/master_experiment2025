import os
import numpy as np
import open3d as o3d
from pathlib import Path

def voxel_index(points: np.ndarray, origin: np.ndarray, voxel_size: float) -> np.ndarray:
    return np.floor((points - origin) / voxel_size).astype(np.int32)

PCD_PATH = r"F:\Experiments\MasterEx\pcd-dataset\PU-GAN\complex\AncientTurtl_aligned.pcd"
VOXEL_SIZE = 0.05
DS_RATIO = 1.0 / 10.0
OUT_DIR = Path("down_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

pcd = o3d.io.read_point_cloud(PCD_PATH)
if pcd.is_empty():
    raise RuntimeError(f"点群が読み込めない: {PCD_PATH}")
pts = np.asarray(pcd.points, dtype=np.float64)

pmin = np.min(pts, axis=0)
v_indices = voxel_index(pts, pmin, VOXEL_SIZE)

for i in range(20):
    #print(f"p:{pmin}, v:{v_indices[i]}")
    print(f'{np.floor((2.4 - (-1.0)) / 0.5).astype(np.int32)}')

print(f"{np.random.default_rng()}\n {list(voxel_to_indices.keys())}
select_k = min(3, len(occupied_keys))
chosen_keys = rng.choice(occupied_keys, size=select_k, replace=False)")