import os
import time
import argparse
import numpy as np
import open3d as o3d

from assist.time import TimeTracker


class DisplayPCV:
    def __init__(self, input_path):
        self.input_path = input_path
        self.time_tracker = TimeTracker()

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

        self.time_tracker.start() # 計測開始
        
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
            pcd = mesh.sample_points_uniformly(number_of_points=len(mesh.triangles))
            print("メッシュを点群に変換しました。")

            self.time_tracker.stop() # 計測終了
            self.time_tracker.print_elapsed(str = "読み込み/変換時間")  # 経過時間を表示
            
            o3d.visualization.draw_geometries([pcd]) # 表示

        elif ext in [".ply", ".pcd", ".xyz"]:
            # 点群ファイルとして読み込み
            pcd = o3d.io.read_point_cloud(self.input_path)
            if pcd.is_empty():
                print("点群データの読み込みに失敗しました。")
                return

            print(pcd)
            print(f"点数: {len(pcd.points)}")

            self.time_tracker.stop() # 計測終了
            self.time_tracker.print_elapsed(str = "読み込み/変換時間")  # 経過時間を表示
            
            o3d.visualization.draw_geometries([pcd]) # 表示

        elif ext == ".pts":
            points = np.loadtxt(self.input_path)
            if points.ndim != 2 or points.shape[1] < 3:
                print("不正な .pts ファイル形式です。x y z 座標が必要です。")
                return
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            print(f".pts 読み込み完了。点数: {points.shape[0]}")

            seg_file = self.input_path.replace(".pts", ".seg")
            if os.path.exists(seg_file):
                labels = np.loadtxt(seg_file).astype(int)
                if len(labels) == points.shape[0]:
                    num_labels = labels.max() + 1
                    colors = np.random.rand(num_labels, 3)
                    pcd.colors = o3d.utility.Vector3dVector(colors[labels])
                    print(f".seg 読み込み完了。ラベル数: {num_labels}")
                else:
                    print(".seg の点数が一致しません。カラー付与をスキップ。")

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