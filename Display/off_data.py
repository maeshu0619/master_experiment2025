import os
import numpy as np
import open3d as o3d

# データセットディレクトリ
dataset_dir = r"F:\Dataset\PU-GAN\simple"

# OFFファイルの拡張子を対象
valid_ext = ['.off']

# ディレクトリからOFFファイルを探索
files = [f for f in os.listdir(dataset_dir) if os.path.splitext(f)[1].lower() in valid_ext]

if not files:
    print("OFFファイルが見つかりません。")
    exit()

# 最初のファイルを使用
file_name = files[0]
file_path = os.path.join(dataset_dir, file_name)
print(f"読み込み中: {file_path}")

# メッシュを読み込み
mesh = o3d.io.read_triangle_mesh(file_path)
print(mesh)

# メッシュ情報を出力
print('Vertices:')
print(np.asarray(mesh.vertices))
print(np.asarray(mesh.vertices).shape)

print('Triangles:')
print(np.asarray(mesh.triangles))
print(np.asarray(mesh.triangles).shape)

# メッシュを点群に変換（4096点サンプリング）
pcd = mesh.sample_points_uniformly(number_of_points=4096)

# 3D表示（メッシュと点群を同時表示）
print(f"メッシュと点群を表示中: {file_name}")
o3d.visualization.draw_geometries([mesh, pcd])
