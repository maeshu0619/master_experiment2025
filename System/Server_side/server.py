import numpy as np

from InternalSystem.file_setup import file_setup, extract_bitrate_and_resolution

class Server:
    def __init__(self, input_video, fps, encode_rate):
        self.fps = fps # 動画フレームレート
        #self.encoded_video = []

        self.max_seg = 60 # 最大セグメント数
        self.remain_seg = self.max_seg # 残りセグメント
        self.streamed_seg = 0 # 既に送信したセグメント数
        self.rebuffering_cnt = 0 # リバッファリング回数

        self.stream_seg_cnt = 3 # 一度に送信するセグメントの最大数

        self.bitrate_resolution = { # 解像度と動画サイズの関係
            250: (375, 666),
            500: (540, 960),
            1000: (750, 1333),
            2000: (1080, 1920)
        }

    def bitrate_resolution_list(self):
        self.bitrate_list, self.resolution_list = extract_bitrate_and_resolution(self.bitrate_resolution) # 解像度リスト
        return self.bitrate_list, self.resolution_list
    
    def PreparingStreaming():
        pass

    def reset():
        pass
    