import os
import numpy as np
import open3d as o3d

from data_expand.file_road import get_files, get_single_file_name

# 入力と出力ファイル
input_file_path = r"..\..\Dataset\ModelNet40\cone\train"
input_files = get_files(input_file_path)
output_file = r"F:\Experiments\MasterEx\demo\ModelNet40"

def convert_off_to_pcd(input_files, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    first_converted_file = None

    for idx, file in enumerate(input_files):
        if file.lower().endswith('.off'):
            with open(file, 'r') as f:
                lines = f.readlines()

            # BOM除去
            header_line = lines[0].strip().replace('\ufeff', '')

            # OFF判定 + 頂点数・面数取得
            if header_line == "OFF":
                n_vertices, n_faces, _ = map(int, lines[1].strip().split())
                vertex_start_index = 2
            elif header_line.startswith("OFF"):
                parts = header_line[3:].strip().split()
                n_vertices, n_faces, _ = map(int, parts)
                vertex_start_index = 1
            else:
                print(f"スキップ: {file} はOFF形式ではありません (先頭行: {header_line})")
                continue

            # 頂点座標の読み込み
            vertices = np.array([
                list(map(float, line.strip().split()))
                for line in lines[vertex_start_index:vertex_start_index + n_vertices]
            ])

            print(f"変換中: {file}, 頂点数: {n_vertices}")

            # Open3Dの点群に変換
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vertices)

            # 出力ファイル名
            base_name = get_single_file_name(file, with_extension=False)
            output_pcd_file = os.path.join(output_path, base_name + '.pcd')

            # 保存
            o3d.io.write_point_cloud(output_pcd_file, pcd, write_ascii=True)
            print(f"PCDファイルを保存しました: {output_pcd_file}")

            if idx == 0:
                first_converted_file = output_pcd_file

    return first_converted_file

# --- 実行部分 ---
# 1つ目のファイルを変換
first_file = []
first_file.append(input_files[0])
pcd_path = convert_off_to_pcd(first_file, output_file)

# 変換した1つ目のファイルを表示
if pcd_path:
    pcv = o3d.io.read_point_cloud(pcd_path)
    o3d.visualization.draw_geometries([pcv])
