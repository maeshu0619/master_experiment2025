import os
import numpy as np
import open3d as o3d
from pathlib import Path

from DownSample.partial import numpy_to_pcd, voxel_index, fps_indices

# ---------- 設定 ----------
PCD_PATH = r"F:\Experiments\MasterEx\pcd-dataset\PU-GAN\complex\AncientTurtl_aligned.pcd"
VOXEL_SIZE = 0.05
DS_RATIO1 = 1.0 / 10.0
DS_RATIO2 = 1.0 / 5.0
NOISE_STD = 0.02
OUT_DIR = Path("down_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 点群読み込み ----------
pcd = o3d.io.read_point_cloud(PCD_PATH)
if pcd.is_empty():
    raise RuntimeError(f"点群が読み込めない: {PCD_PATH}")
pts = np.asarray(pcd.points, dtype=np.float64)

# ---------- Voxel分割 ----------
pmin = np.min(pts, axis=0)
v_indices = voxel_index(pts, pmin, VOXEL_SIZE)

# ---------- occupied voxel（点が存在するボクセル）を収集 ----------
voxel_to_indices = {}
for i, idx in enumerate(v_indices):
    key = tuple(idx)
    voxel_to_indices.setdefault(key, []).append(i)

# ---------- occupied voxel からランダムに複数個選択 ----------
rng = np.random.default_rng()
occupied_keys = list(voxel_to_indices.keys())
select_k = min(3, len(occupied_keys))
chosen_keys = rng.choice(occupied_keys, size=select_k, replace=False)

# ---------- FPS＋ノイズ（選ばれたボクセルのみ） ----------
kept_points = []
noisy_points = []
original_points = []

for key, indices in voxel_to_indices.items():
    voxel_pts = pts[indices]
    if key in chosen_keys:
        k = max(3, int(len(voxel_pts) * DS_RATIO1))
        if len(voxel_pts) < k:
            continue  # 小さすぎる場合はスキップ
        idx = fps_indices(voxel_pts, k, seed=rng.integers(0, 10000))
        selected = voxel_pts[idx]
        noise = rng.normal(loc=0.0, scale=NOISE_STD, size=selected.shape)
        noisy_points.append(selected + noise)
        original_points.append(selected)
    else:
        """
        k = max(3, int(len(voxel_pts) * DS_RATIO2))
        if len(voxel_pts) < k:
            continue  # 小さすぎる場合はスキップ
        idx = fps_indices(voxel_pts, k, seed=rng.integers(0, 10000))
        selected = voxel_pts[idx]
        """
        kept_points.append(voxel_pts)

# ---------- 統合 ----------
kept = np.concatenate(kept_points, axis=0)
original = np.concatenate(original_points, axis=0) if original_points else np.empty((0, 3))
noisy = np.concatenate(noisy_points, axis=0) if noisy_points else np.empty((0, 3))

pcd_kept = numpy_to_pcd(kept, color=(0.5, 0.5, 0.5))       # 通常点群（灰）
pcd_orig = numpy_to_pcd(original, color=(0.2, 0.8, 0.2))   # 元のFPS点（緑）
pcd_noisy = numpy_to_pcd(noisy, color=(1.0, 0.0, 0.0))     # ノイズ付き点（赤）

# ---------- 保存（ノイズ込みのみ） ----------
pcd_all = numpy_to_pcd(np.concatenate([kept, noisy], axis=0), color=(0.8, 0.5, 0.3))
out_path = OUT_DIR / "partial_voxel_fps_noise_fixed.pcd"
o3d.io.write_point_cloud(str(out_path), pcd_all, write_ascii=True)
print(f"保存しました: {out_path}")

# ---------- 表示 ----------
o3d.visualization.draw_geometries(
    [pcd_kept, pcd_orig, pcd_noisy],
    window_name="Fixed: Voxel-wise FPS+Noise",
    width=1024,
    height=768,
)
