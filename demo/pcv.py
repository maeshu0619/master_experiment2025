import open3d as o3d

# PCDファイル（ASCII形式でもバイナリでもOK）
input_file = r"F:\Experiments\MasterEx\demo\AncientTurtl_pointcloud_ascii.pcd"

# 点群データを読み込む
pcd = o3d.io.read_point_cloud(input_file)

# 読み込んだ情報を確認
print(pcd)
print(f"点数: {len(pcd.points)}")

# 表示
o3d.visualization.draw_geometries([pcd])
