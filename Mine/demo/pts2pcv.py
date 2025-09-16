import os
import numpy as np
import open3d as o3d

from data_expand.file_road import get_files, get_single_file_name

# 入力と出力ファイル
input_file_path = r"..\..\Dataset\PartAnnotation\03636649\points"
input_files = get_files(input_file_path)
output_file = r"F:\Experiments\MasterEx\demo\pts"

def convert_one_pts_to_pcd(input_file, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)  # 出力フォルダがなければ作成

    if input_file.lower().endswith('.pts'):  # .ptsファイルのみ処理
        # PTSファイルを読み込み（各行: x y z）
        with open(input_file, 'r') as f:
            lines = f.readlines()

        # 数値データに変換
        vertices = np.array([list(map(float, line.strip().split())) for line in lines])

        print(f"変換中: {input_file}, 頂点数: {len(vertices)}")

        # 点群をOpen3Dオブジェクトに変換
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)

        # 出力ファイル名を決定
        base_name = get_single_file_name(input_file, with_extension=False)
        output_pcd_file = os.path.join(output_path, base_name + '.pcd')

        # PCDをASCII形式で保存
        o3d.io.write_point_cloud(output_pcd_file, pcd, write_ascii=True)
        print(f"✅ PCDファイルを保存しました: {output_pcd_file}")

        return output_pcd_file  # 保存したファイルのパスを返す
    else:
        print(f"スキップ: {input_file} はPTS形式ではありません")
        return None

# --- 実行部分 ---
# 1つ目のファイルを変換
first_file = input_files[0]
pcd_path = convert_one_pts_to_pcd(first_file, output_file)

# 変換したPCDファイルを読み込んで表示
if pcd_path:
    pcv = o3d.io.read_point_cloud(pcd_path)
    o3d.visualization.draw_geometries([pcv])
