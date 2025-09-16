import glob
import open3d as o3d
import numpy as np
import os

input_path = r"F:\Experiments\MasterEx\down_data\partial_voxel_fps.pcd"

class DisplayPCD:
    def __init__(self, input_path):
        self.input_path = glob.glob(input_path)
    
    def display(self):
        """指定されたPCDファイルを読み込み、点群を表示"""
        if not self.input_path:
            print("指定されたパスにPCDファイルが見つかりません。")
            return

        print(f"読み込み中: {self.input_path[0]}")

        pcd = o3d.io.read_point_cloud(self.input_path[0])
        if pcd.is_empty():
            print("点群データの読み込みに失敗しました。")
            return
        
        print("候補:", self.input_path[:5])
        assert self.input_path, "PCDファイルが見つからない。変換コードの出力先とファイル名規則を確認せよ。"

        
        print("表示ファイル:", self.input_path[0], "点数:", len(pcd.points))
        o3d.visualization.draw_geometries([pcd])


app = DisplayPCD(input_path)
app.display()