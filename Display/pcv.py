import os
import time
import argparse
import numpy as np
import open3d as o3d

from assist.time import TimeTracker


class DisplayPCV:
    def __init__(self, input_path):
        self.input_path = input_path

    def display(self):
        """入力パスのファイルを読み込み、点群を表示"""
        if os.path.isdir(self.input_path):
            print(f"ディレクトリが指定されました: {self.input_path}")
            files = [f for f in os.listdir(self.input_path)]
            print("ディレクトリ内のファイル:", files)
            return

        if not os.path.isfile(self.input_path):
            print("指定されたパスがファイルではありません。")
            return

        TimeTracker().start() # 計測開始
        
        ext = os.path.splitext(self.input_path)[1].lower()
        print(f"読み込み中: {self.input_path} (拡張子: {ext})")

        if ext == ".off":
            # OFFファイル → メッシュとして読み込み → 点群に変換
            mesh = o3d.io.read_triangle_mesh(self.input_path)
            if mesh.is_empty():
                print("メッシュデータの読み込みに失敗しました。")
                return

            print(mesh)
            print(f"Vertices数: {len(mesh.vertices)}, Triangles数: {len(mesh.triangles)}")

            # 点群サンプリング
            pcd = mesh.sample_points_uniformly(number_of_points=4096)
            print("メッシュを点群に変換しました。")

            TimeTracker().stop() # 計測終了
            
            o3d.visualization.draw_geometries([mesh, pcd]) # 表示

        elif ext in [".ply", ".pcd", ".xyz"]:
            # 点群ファイルとして読み込み
            pcd = o3d.io.read_point_cloud(self.input_path)
            if pcd.is_empty():
                print("点群データの読み込みに失敗しました。")
                return

            print(pcd)
            print(f"点数: {len(pcd.points)}")

            TimeTracker().stop() # 計測終了
            
            o3d.visualization.draw_geometries([pcd]) # 表示

        else:
            print(f"サポートされていないファイル形式: {ext}")


if __name__ == "__main__":
    st = time.time()

    parser = argparse.ArgumentParser(description="PCV Viewer: メッシュ/点群を表示")
    parser.add_argument(
        "--data",
        type=str,
        default=r"F:\Dataset\PU-GAN\simple\armadillo.off",
        help="ファイルパスを指定してください（例: F:\\Dataset\\PU-GAN\\simple\\armadillo.off）"
    )

    args = parser.parse_args()

    if not args.data or not os.path.exists(args.data):
        print("指定されたパスが存在しません。")
        exit()

    viewer = DisplayPCV(args.data)
    viewer.display()

    TimeTracker().print_elapsed()  # 経過時間を表示
