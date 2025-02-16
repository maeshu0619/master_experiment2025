import numpy as np
from DQN.agent import Actor

from InternalSystem.rate_simu import simulate_transmission_rate
from InternalSystem.action_generate import focas_combination, ours_combination
from InternalSystem.gaze_prediction import gaze_data

class Client:
    def __init__(self, mode, max_steps_per_episode, 
                 mu, sigma_ratio, base_band):
        self.mode = mode
        self.max_steps_per_episode = max_steps_per_episode
        self.mu = mu
        self.sigma_ratio = sigma_ratio
        self.base_band = base_band

        act = Actor(self.mode)

        self.buffer_size = 3
        self.buffer = []

        self.bitrate_index = 0
        self.size_fovea_index = 0
        self.size_blend_index = 0
        self.bitrate_index = 0
        self.bitrate_index = 0

        self.action_comb = ours_combination() # 行動範囲の組み合わせ

        self.min_dis = 150 # 送受信アンテナ間距離の最小値
        self.max_dis = 1000 # 送受信アンテナ間距離の最大値
        self.distance = np.random.uniform(self.min_dis, self.max_dis) # 送受信アンテナ間距離の決定

        self.depth_fovea_list = [6,7,8,9,10] # フォビア深度リスト
        self.depth_blend_list = [3,4,5,6,7] # ブレンド深度リスト
        self.depth_peri_list = [1,2,3,4,5] # 周辺深度リスト
        self.size_list = [x / 12 for x in [0,1,2,3,4,5]] # サイズリスト

        # 帯域幅シミュレート
        self.transmission_list = simulate_transmission_rate(self.distance, self.max_steps_per_episode, self.mu, self.sigma_ratio, self.base_band)
        # 視線情報の取得
        self.gazedata_path = "UD_UHD_EyeTrakcing_Videos/Gaze_Data/HD"
        self.gaze_coordinates = gaze_data(self.gazedata_path, self.max_steps_per_episode, video_center=(960, 540))
        
        self.play_stop = -1 # 動画再生フラグ
            

    def get_simulation_data(self): # シミュレーションデータ取得
        self.transmission_list = simulate_transmission_rate(self.distance, self.max_steps_per_episode, self.mu, self.sigma_ratio, self.base_band)
        self.gaze_coordinates = gaze_data(self.gazedata_path, self.max_steps_per_episode, video_center=(960, 540))
        return self.transmission_list, self.gaze_coordinates

    def buffer_add(self, bitrate): # セグメントをバッファに保存
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(bitrate)
    
    def seg_play(self): # セグメントから1つ取り出して動画を再生
        if len(self.buffer) != 0:
            self.buffer.pop()
        else:
            self.play_stop = 1

    def reset(self):
        action_invalid_judge = False

        self.distance = np.random.uniform(self.min_dis, self.max_dis) # 送受信アンテナ間距離の決定

        self.get_simulation_data() # シミュレーションデータ取得


