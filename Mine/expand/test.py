import numpy as np
import open3d as o3d

# 拡張関数群
def nonuniform_sampling(num, sample_num):
    sample = set()
    loc = np.random.rand() * 0.8 + 0.1
    while len(sample) < sample_num:
        a = int(np.random.normal(loc=loc, scale=0.3) * num)
        if 0 <= a < num:
            sample.add(a)
    return list(sample)

def jitter_perturbation_point_cloud(input, sigma=0.005, clip=0.02):
    N, C = input.shape
    jittered = np.clip(sigma * np.random.randn(N, C), -clip, clip)
    jittered += input
    return jittered

def rotate_point_cloud(input):
    angles = np.random.uniform(size=(3)) * 2 * np.pi
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]),  np.cos(angles[0])]])
    Ry = np.array([[ np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[ np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [ np.sin(angles[2]),  np.cos(angles[2]), 0],
                   [0, 0, 1]])
    R = Rz @ (Ry @ Rx)
    return input @ R.T

def random_scale_point_cloud(input, scale_low=0.8, scale_high=1.2):
    scale = np.random.uniform(scale_low, scale_high)
    return input * scale

def augment_point_cloud(points, num_sample=1024):
    idx = nonuniform_sampling(points.shape[0], min(num_sample, points.shape[0]))
    points = points[idx, :]
    points = jitter_perturbation_point_cloud(points, sigma=0.01, clip=0.03)
    points = rotate_point_cloud(points)
    points = random_scale_point_cloud(points, 0.8, 1.2)
    return points


# ==== 実行部 ====
pcd_path = "/home/labliu/maejima/MasterEx2025/pcd-dataset/PU-GAN/medium/10014_dolphin_v2_max2011_it2.pcd"
pcd = o3d.io.read_point_cloud(pcd_path)
print(f"Original points: {len(pcd.points)}")

# 拡張
aug_points = augment_point_cloud(np.asarray(pcd.points), num_sample=2048)
aug_pcd = o3d.geometry.PointCloud()
aug_pcd.points = o3d.utility.Vector3dVector(aug_points)

# 保存（PCDのまま）
out_path = "/home/labliu/maejima/MasterEx2025/augmented_dolphin.pcd"
o3d.io.write_point_cloud(out_path, aug_pcd)
print(f"拡張後の点群を {out_path} に保存しました。")
