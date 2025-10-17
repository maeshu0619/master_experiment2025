"""
RepKPU内にあるPU-GANデータセットの点群の分布を非一様にする
"""

import os
import numpy as np
import open3d as o3d
from RepKPU.dataset.utils import nonuniform_sampling


def make_nonuniform_dataset(input_dir='./RepKPU/result/pugan_2x/xyz',
                            save_dir='./RepKPU/data/PU-GAN/test/pugan_4x/nu_same_num',
                            num_points=2048):
    """
    点群を非一様サンプリングして、局所的に密度の異なる分布にする。
    utils.nonuniform_sampling() を用いて非一様な分布を再現。
    """

    os.makedirs(save_dir, exist_ok=True)

    xyz_files = [f for f in os.listdir(input_dir) if f.endswith('.xyz')]
    print(f"[INFO] Found {len(xyz_files)} files in {input_dir}")

    for fname in xyz_files:
        # 点群の読み込み
        path = os.path.join(input_dir, fname)
        pcd = o3d.io.read_point_cloud(path)
        points = np.asarray(pcd.points)

        total_points = points.shape[0]

        # --- 変更点ここ ---
        # 入力点群数の半分をサンプリング数とする
        num_points = total_points // 2
        # ------------------

        # 非一様サンプリング（PUDatasetと同じ手法）
        sample_idx = nonuniform_sampling(points.shape[0], sample_num=num_points)
        sampled_points = points[sample_idx, :]

        # 保存
        save_path = os.path.join(save_dir, fname)
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(sampled_points)
        o3d.io.write_point_cloud(save_path, new_pcd)
        print(f"[SAVED] Nonuniform point cloud -> {save_path}")

    print(f"\n[FINISHED] Nonuniform dataset saved in: {save_dir}")


if __name__ == "__main__":
    make_nonuniform_dataset()
