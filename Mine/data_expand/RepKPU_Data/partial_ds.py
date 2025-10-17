import os
import numpy as np
import open3d as o3d

def make_partial_downsample_dataset(input_dir='./RepKPU/data/PU-GAN/test/pugan_4x/input',
                                    save_dir='./RepKPU/data/PU-GAN/test/pugan_4x/partial_downsampled',
                                    downsample_ratio=0.3,
                                    region_ratio=0.4):
    """
    点群の一部領域だけをダウンサンプリングして「部分的に疎な」点群を生成する。

    Parameters:
    ----------
    input_dir : str
        入力点群(.xyz)が格納されたディレクトリ
    save_dir : str
        出力先ディレクトリ
    downsample_ratio : float
        部分的に削除する点の割合（例: 0.3 → 該当地域の点を30%残す）
    region_ratio : float
        ダウンサンプリングを適用する空間領域の比率（例: 0.4 → 全体の40%の範囲に適用）
    """

    os.makedirs(save_dir, exist_ok=True)
    xyz_files = [f for f in os.listdir(input_dir) if f.endswith('.xyz')]
    print(f"[INFO] Found {len(xyz_files)} files in {input_dir}")

    for fname in xyz_files:
        path = os.path.join(input_dir, fname)
        pcd = o3d.io.read_point_cloud(path)
        points = np.asarray(pcd.points)
        n_points = points.shape[0]

        # --- 1. 点群全体のバウンディングボックス ---
        min_bounds = np.min(points, axis=0)
        max_bounds = np.max(points, axis=0)
        box_size = max_bounds - min_bounds

        # --- 2. ダウンサンプリングする部分領域をランダム選択 ---
        region_center = min_bounds + np.random.rand(3) * box_size
        region_extent = box_size * region_ratio / 2.0

        x_min, y_min, z_min = region_center - region_extent
        x_max, y_max, z_max = region_center + region_extent

        # --- 3. 領域内の点を抽出 ---
        inside_mask = (
            (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
            (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
            (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        )
        inside_idx = np.where(inside_mask)[0]
        outside_idx = np.where(~inside_mask)[0]

        # --- 4. 部分領域内をダウンサンプリング ---
        num_keep_inside = int(len(inside_idx) * downsample_ratio)
        if num_keep_inside > 0:
            keep_inside = np.random.choice(inside_idx, num_keep_inside, replace=False)
        else:
            keep_inside = np.array([], dtype=int)

        # --- 5. 外部領域はすべて保持 ---
        keep_indices = np.concatenate([keep_inside, outside_idx])

        # --- 6. 新しい点群を生成 ---
        partial_points = points[keep_indices, :]

        # --- 7. 保存 ---
        save_path = os.path.join(save_dir, fname)
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(partial_points)
        o3d.io.write_point_cloud(save_path, new_pcd)

        print(f"[SAVED] {fname} ({n_points} -> {partial_points.shape[0]} points)")

    print(f"\n[FINISHED] Partial downsampled dataset saved under {save_dir}")


if __name__ == "__main__":
    make_partial_downsample_dataset()
