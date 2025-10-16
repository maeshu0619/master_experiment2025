import numpy as np
import open3d as o3d

# 非一様サンプリング（参考コードより）
def nonuniform_sampling(num, sample_num):
    sample = set()
    loc = np.random.rand() * 0.8 + 0.1
    while len(sample) < sample_num:
        a = int(np.random.normal(loc=loc, scale=0.3) * num)
        if 0 <= a < num:
            sample.add(a)
    return list(sample)

# ==== ボクセル化して一部を疎にする処理 ====
def voxel_downsample_partial(pcd, voxel_size=0.1, downsample_ratio=0.3):
    """
    pcd: open3d.geometry.PointCloud
    voxel_size: ボクセルサイズ
    downsample_ratio: 部分的に残す割合（小さいほど疎）
    """

    # 点群をnumpy化
    pts = np.asarray(pcd.points)

    # 各点をボクセルインデックスに割り当て
    min_bound = pts.min(axis=0)
    voxel_indices = np.floor((pts - min_bound) / voxel_size).astype(int)

    # ボクセルごとの点のインデックスを収集
    from collections import defaultdict
    voxel_to_indices = defaultdict(list)
    for idx, v in enumerate(voxel_indices):
        voxel_to_indices[tuple(v)].append(idx)

    # 1つだけランダムにボクセルを選択
    chosen_voxel = list(voxel_to_indices.keys())[np.random.randint(len(voxel_to_indices))]
    chosen_indices = voxel_to_indices[chosen_voxel]

    # 部分的にダウンサンプリング（nonuniform_samplingで残す）
    keep_num = max(1, int(len(chosen_indices) * downsample_ratio))
    sampled_idx = nonuniform_sampling(len(chosen_indices), keep_num)
    keep_indices_in_voxel = [chosen_indices[i] for i in sampled_idx]

    # 色をつける（赤=疎にしたボクセル、灰色=それ以外）
    colors = np.tile(np.array([[0.5, 0.5, 0.5]]), (pts.shape[0], 1))  # 全体を灰色
    colors[keep_indices_in_voxel] = np.array([1.0, 0.0, 0.0])  # 残った疎ボクセルは赤

    # 最終的に疎ボクセルは残した点だけ、それ以外は全部そのまま保持
    keep_indices_total = set(range(len(pts))) - set(chosen_indices)  # 他のボクセルは全保持
    keep_indices_total = list(keep_indices_total) + keep_indices_in_voxel

    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(pts[keep_indices_total])
    new_pcd.colors = o3d.utility.Vector3dVector(colors[keep_indices_total])

    return new_pcd

# ==== 実行部分 ====
pcd_path = "./pcd-dataset/PU-GAN/medium/10014_dolphin_v2_max2011_it2.pcd"
pcd = o3d.io.read_point_cloud(pcd_path)
print(f"Original points: {len(pcd.points)}")

# 部分的に密度を下げる
aug_pcd = voxel_downsample_partial(pcd, voxel_size=0.05, downsample_ratio=0.2)

# 保存（表示はしない）
o3d.io.write_point_cloud("partial_downsampled.pcd", aug_pcd)
print("partial_downsampled.pcd として保存しました。")
