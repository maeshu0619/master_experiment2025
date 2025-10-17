"""
RepKPUディレクトリ内のPU-GANデータセットを拡張する
拡張方法は、RepKPU内に元からある拡張コード
ジッタ、回転、スケーリングを行う
"""


import os
import numpy as np
import torch
import open3d as o3d
from RepKPU.dataset.utils import jitter_perturbation_point_cloud, rotate_point_cloud_and_gt, random_scale_point_cloud_and_gt

def expand_test_dataset(input_dir='./RepKPU/data/PU-GAN/test/pugan_4x/input',
                        save_dir='./RepKPU/data/PU-GAN/test/pugan_4x/expanded',
                        jitter_sigma=0.005,
                        jitter_max=0.02,
                        scale_low=0.8,
                        scale_high=1.2,
                        num_augmentations=3):
    """
    テスト用データセットにPUDatasetクラス同等のデータ拡張を適用し、
    ./expandedディレクトリ以下に保存する。

    各ファイルごとに複数の拡張バリエーションを作成する。
    """

    os.makedirs(save_dir, exist_ok=True)

    # 入力ディレクトリ内の全.xyzファイルを取得
    xyz_files = [f for f in os.listdir(input_dir) if f.endswith('.xyz')]
    print(f"[INFO] Found {len(xyz_files)} files in {input_dir}")

    for fname in xyz_files:
        # 点群を読み込み
        path = os.path.join(input_dir, fname)
        pcd = o3d.io.read_point_cloud(path)
        points = np.asarray(pcd.points)
        gt = np.copy(points)  # gtは同一で回転・スケーリングを合わせるため

        # ファイル名拡張用ディレクトリ
        base_name = os.path.splitext(fname)[0]
        # target_dir = os.path.join(save_dir, base_name)
        target_dir = os.path.join(save_dir)
        os.makedirs(target_dir, exist_ok=True)

        for i in range(num_augmentations):
            # jitter
            jittered = jitter_perturbation_point_cloud(points, sigma=jitter_sigma, clip=jitter_max)

            # 回転
            rotated_input, rotated_gt = rotate_point_cloud_and_gt(jittered, gt)

            # スケーリング
            scaled_input, scaled_gt, _ = random_scale_point_cloud_and_gt(rotated_input, rotated_gt,
                                                                         scale_low=scale_low,
                                                                         scale_high=scale_high)

            # Open3D形式に変換して保存
            new_pcd = o3d.geometry.PointCloud()
            new_pcd.points = o3d.utility.Vector3dVector(scaled_input)

            # save_path = os.path.join(target_dir, f"{base_name}_aug{i+1}.xyz")
            save_path = os.path.join(target_dir, f"{base_name}.xyz")
            o3d.io.write_point_cloud(save_path, new_pcd)
            print(f"[SAVED] {save_path}")

    print(f"\n[FINISHED] All augmented files are saved under {save_dir}")


if __name__ == "__main__":
    expand_test_dataset()
