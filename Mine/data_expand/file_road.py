"""
データセットのファイル内の各データのアドレス名を取得するコード
"""
import os

# ファイル内のデータを全て取得
def get_files(folder_path):
    """
    指定フォルダ内の全ファイルパス（拡張子問わず）を昇順で取得する関数
    """
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    files.sort()  # 名前順でソート
    return files

#　引数のデータのファイル名を取得
def get_single_file_name(file_path, with_extension=True):
    if not os.path.isfile(file_path):
        raise ValueError(f"指定したパスはファイルではありません: {file_path}")
    
    if with_extension:
        return os.path.basename(file_path)
    else:
        return os.path.splitext(os.path.basename(file_path))[0]


"""
file_path = r"..\..\Dataset\PU-GAN\simple"
files = get_files(file_path)   
for file in files:
    print(get_single_file_name(file, with_extension=True))
"""