import os
import numpy as np
import open3d as o3d


def make_downsampled_dataset(input_dir='./RepKPU/data/PU-GAN/test/pugan_4x/input',
                             save_dir='./RepKPU/data/PU-GAN/test/pugan_4x/downsampled',
                             downsample_ratio=0.25):
    """
    点群を一様にダウンサンプリングして密度を減らす。
    Open3Dのuniform_down_sample()を使用。
    downsample_ratio=0.25 → 点数を1/4に削減。
    """

    os.makedirs(save_dir, exist_ok=True)

    xyz_files = [f for f in os.listdir(input_dir) if f.endswith('.xyz')]
    print(f"[INFO] Found {len(xyz_files)} files in {input_dir}")

    for fname in xyz_files:
        path = os.path.join(input_dir, fname)
        pcd = o3d.io.read_point_cloud(path)

        # 元点群の点数
        n_points = np.asarray(pcd.points).shape[0]
        every_k = int(1 / downsample_ratio)

        # 一様ダウンサンプリング
        down_pcd = pcd.uniform_down_sample(every_k_points=every_k)

        # 保存
        save_path = os.path.join(save_dir, fname)
        o3d.io.write_point_cloud(save_path, down_pcd)
        print(f"[SAVED] Downsampled ({n_points} -> {len(down_pcd.points)}) -> {save_path}")

    print(f"\n[FINISHED] Downsampled dataset saved in: {save_dir}")


if __name__ == "__main__":
    make_downsampled_dataset()
