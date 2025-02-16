import numpy as np
import os
from gym import Env
from gym.spaces import Discrete, Box
from tqdm import tqdm
import datetime
from InternalSystem.qoe_cal import qoe_cal, ave_cal
from InternalSystem.file_setup import file_setup, extract_bitrate_and_resolution
from InternalSystem.gaze_prediction import gaze_data
from InternalSystem.graph_plot import generate_cdf_plot, generate_training_plot
from InternalSystem.Client.action_generate import focas_combination, ours_combination
from InternalSystem.rate_simu import simulate_transmission_rate
from InternalSystem.load_trace import transmissionSimulator
from DQN.agent import Actor
from System.Server_side.server import Server
from System.Client_side.client import Client

class Environment(Env):
    def __init__(self, mode, train_or_test, latency_file, network_file, 
                 total_timesteps, max_steps_per_episode, latency_constraint, fps, 
                 mu, sigma_ratio, base_band):
        super(Environment, self).__init__()
        self.server = Server # サーバクラス
        self.client = Client # クライアントクラス
        
        # 行動
        self.actor = Actor(mode)

        ###################################################################################################
        """引数の再定義"""
        self.mode = mode
        self.train_or_test = train_or_test
        self.latency_file = latency_file
        self.network_file = network_file
        self.total_timesteps = total_timesteps
        self.max_steps_per_episode = max_steps_per_episode
        self.latency_constraint = latency_constraint
        self.mu = mu
        self.sigma_ratio = sigma_ratio
        self.base_band = base_band
        ###################################################################################################
        
        # トレーニングかテストか
        if self.train_or_test == 0:
            output_file="graph_train"
        elif self.train_or_test == 1:
            output_file="graph_test"

        # 解像度と動画サイズのリスト
        self.bitrate_list, self.resolution_list = self.server.bitrate_resolution_list()

        # グラフ保存のためのアドレス生成
        mode_name = "Ours"
        self.current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # 現在の時刻を取得    
        self.graph_file_path = os.path.join(output_file, f"{mode_name}/{self.latency_file}/{self.network_file}/{self.current_time}") # 出力フォルダの作成

        self.transmission_list, self.gaze_coordinates = self.client.get_simulation_data() # シミュレーションデータの取得

        # 遅延制約超過のエラー比率計算
        self.error_late = 0
        self.error_late_per = []
        # 伝送レート超過のエラー比率計算
        self.error_buffer = 0
        self.error_buffer_per = []

        ###################################################################################################
        """超解像におけるサイズ、深さ、処理時間の定義"""
        self.depth_fovea_list = [6,7,8,9,10] # フォビア深度リスト
        self.depth_blend_list = [3,4,5,6,7] # ブレンド深度リスト
        self.depth_peri_list = [1,2,3,4,5] # 周辺深度リスト
        self.size_list = [x / 12 for x in [0,1,2,3,4,5]] # サイズリスト

        # ResBlock1ピクセル当たり一層通過させるのに必要な計算時間と、それによって何倍解像度が向上するかの情報
        self.resblock_time = 2.525720165e-8
        self.other_time = 0
        self.resblock_quality = 1.148698354997035
        self.resblock_info = (self.resblock_time, self.other_time, self.resblock_quality)
        ###################################################################################################

        ###################################################################################################
        """履歴用リストの定義"""
        self.quality_vc_history = [] # 報酬の動画品質の履歴
        self.jitter_t_history = [] # 報酬の時間ジッタの履歴
        self.jitter_s_history = [] # 報酬の空間ジッタの履歴
        self.rebuffer_history = [] # 報酬のリバッファリングペナルティ

        self.bitrate_history = [] # 選択された品質の履歴
        self.resolution_history = [] # 選択された動画サイズの履歴
        self.size_history = [] # 選択された領域サイズの履歴
        self.depth_history = [] # 選択された深さの履歴
        self.transmission_history = [] # シミュレートした帯域幅の履歴
        self.latency_history = [] # 遅延の履歴
        self.latency_ave = 0

        self.q_values = []  # Q値の履歴を保持
        self.action_history = [] # 行動の履歴
        self.reward_history = [] # 報酬の履歴
        self.reward_ave_history = [] # 各エピソードの報酬の平均の履歴
        self.transmission_ave_history = [] # 各エピソードの伝送レートの平均の履歴
        ###################################################################################################
        

        self.action_space = Discrete(4800)
        self.observation_space = Box(
            low=0,
            high=np.inf,
            shape=(4,), # 現在の帯域幅，選択された品質，現在の視線座標
            dtype=np.float32
        )

        # ログファイルのセットアップ
        self.log_file, self.debug_log, self.logger = file_setup(self.mode, self.train_or_test, self.current_time, self.latency_file, self.network_file)

        self.time_in_training = 0 # 初期化しない
        self.steps_per_episode = 0 # エピソード終了時に初期化

    """初期化関数"""
    def reset(self):
        self.client.reset()
        self.server.reset()

        self.bitrate_history = [] # 選択されたビットレート履歴の初期化
        self.latency_ave = 0 # 遅延制約の初期化
        
        if self.steps_per_episode != 0:
            # 遅延制約超過記録
            self.error_late_per.append(100*self.error_late/self.steps_per_episode)
            self.error_late = 0
            # 帯域幅超過記録
            self.error_buffer_per.append(100*self.error_buffer/self.steps_per_episode)
            self.error_buffer = 0

        self.steps_per_episode = 0

        self.bitrate_history.append(self.bitrate_list[0])  # 初期ビットレートを設定
        state = np.array([self.transmission_list[0], self.bitrate_history[0], self.gaze_coordinates[0][0], self.gaze_coordinates[0][1]], dtype=np.float32) # 状態の初期化

        self.bitrate_legacy.append(self.bitrate_list[0])  # 初期ビットレートを設定
        self.gaze_coordinates = gaze_data(self.directory_path, self.max_steps_per_episode, video_center=(960, 540))        
        self.distance = np.random.uniform(self.min_dis, self.max_dis)
        self.bandwidth_list = simulate_transmission_rate(self.distance, self.max_steps_per_episode, self.mu, self.sigma_ratio, self.base_band)
        state = np.array([self.bandwidth_list[0], self.bitrate_legacy[0], self.gaze_coordinates[0][0], self.gaze_coordinates[0][1]], dtype=np.float32) # 状態の初期化

        action_invalid_judge = False
        done = False
            
        self.log_file.write(f"--- state reset ---\n")
        self.debug_log.write(f"--- state reset ---\n")
        return state

    """ステップごとに環境を更新するための関数"""
    def step(self, action, goal_reward):
        self.debug_log.write(f'current step is {self.time_in_training+1} / {self.total_timesteps+1}\n')
        self.debug_log.write(f'action: {action}\n')

        reward = 0 # 報酬の初期化

        # 伝送レートの変化をシミュレーション
        self.transmission_history.append(self.transmission_list[self.steps_per_episode])

        # 行動の数値を各情報に割り当てる
        index = self.client.assign_action(action) # 行動の割り当て

        self.bitrate_history.append(self.bitrate_list[index[0]]) # 解像度履歴の記録
        self.resolution_history.append(self.resolution_list[index[0]]) # 動画サイズ履歴の記録
        self.size_history.append([int(self.size_list[index[1][0]] * self.resolution_list[index[0]][0]),  # サイズ履歴の記録
                        int(self.size_list[index[1][1]] * self.resolution_list[index[0]][0])])
        self.depth_history.append([self.depth_fovea_list[index[2][0]], # 深さ履歴の記録
                        self.depth_blend_list[index[2][1]], 
                        self.depth_peri_list[index[2][2]]])

        # QoE計算 # 報酬の各項、報酬、処理時間、行動の有効無効フラグ
        qoe_paragraph, reward, all_cal_time, action_invalid_judge= qoe_cal(self.mode, self.steps_per_episode, self.time_in_training, self.bitrate_history, self.resolution_history, 
                                                                self.bitrate_list, self.resolution_list, self.quality_vc_history, 
                                                                self.transmission_history, self.resblock_info, self.gaze_coordinates, self.buffer, self.max_buffer, self.segment_length, 
                                                                self.size_history, self.depth_history, self.latency_constraint, self.debug_log, self.train_or_test)
        
        # 無効な行動を記録
        if action_invalid_judge:
            self.error_late += 1 # 遅延制約違反回数を記録
            self.actor.add_invalid_action(action)

        self.latency_ave += all_cal_time # 処理時間の和

        if qoe_paragraph[3] > 0: # 再生中断ペナルティが発生している場合
            self.error_buffer += 1 # ペナルティの発生回数を1つ追加

        ###################################################################################################
        """記録の保存"""
        # 報酬の内訳の保存
        self.quality_vc_history.append(qoe_paragraph[0]) # 平均品質の履歴記録
        self.jitter_t_history.append(qoe_paragraph[1]) # 時間ジッタの履歴記録
        self.jitter_s_history.append(qoe_paragraph[2]) # 空間ジッタの履歴記録
        self.rebuffer_history.append(qoe_paragraph[3]) # 再生中断ペナルティの履歴記録
        # 報酬と行動の保存
        self.action_history.append(action) # 行動の履歴記録
        self.reward_history.append(reward) # 報酬の履歴記録
        # TensorBoard用のカスタム記録
        self.logger.record("Reward", reward)
        self.logger.record("Selected Bitrate", self.bitrate_history[self.steps_per_episode])
        self.logger.record("Transmission Rate", self.transmission_history[self.steps_per_episode])
        self.logger.dump(self.time_in_training)
        # ログに記録
        self.log_file.write(
            f"Step {self.time_in_training+1}({self.steps_per_episode+1}/{self.max_steps_per_episode}): Action:{action}, Reward:{reward:.2f}\n"
            f"     Transmission Rate: {self.transmission_history[self.steps_per_episode]:.2f}, Selected Bitrate: {self.bitrate_history[self.steps_per_episode]}, Average Quality: {qoe_paragraph[0]}, Time Jitter: {qoe_paragraph[1]}, Scale Jitter: {qoe_paragraph[2]}, Rebuffering Penalty: {qoe_paragraph[3]}\n"
        )
        # 内部情報の記録
        self.debug_log.write(
            f'Bitrate -> {index[0]}\n'
            f'Size -> fovea:{index[1][0]}, blend:{index[1][1]}\n'
            f'Depth -> fovea:{index[2][0]}, blend:{index[2][1]}, peri:{index[2][2]}\n'
            f'Transmission Rate: {self.transmission_list[self.steps_per_episode][0]}, {self.transmission_list[self.steps_per_episode][1]})\n'
            f'Gaze Coordinate(y,x): ({self.gaze_coordinates[self.steps_per_episode][0]}, {self.gaze_coordinates[self.steps_per_episode][1]})\n'
        )
        ###################################################################################################
        
        # 終了フラグ
        done = (np.mean(self.reward_history) >= goal_reward)  # 目標報酬に達したら終了

        # 状態を更新
        state = np.array([self.transmission_history[-1], # 伝送レート
                            self.bitrate_history[-1], # 要求したビットレート
                            self.gaze_coordinates[self.steps_per_episode][0], # 焦点のy座標
                            self.gaze_coordinates[self.steps_per_episode][1]], # 焦点のx座標
                            dtype=np.float32)

        self.steps_per_episode += 1 # 各エピソードごとのステップ数の進行
        self.time_in_training += 1 # 各トレーニングごとのステップ数の進行


        ###################################################################################################
        """エピソードの最後のステップの場合、各パラメータの平均値等を記録"""
        if self.steps_per_episode == self.max_steps_per_episode:
            self.reward_ave_history.append(ave_cal(self.reward_history, self.max_steps_per_episode)) # 報酬の平均
            self.transmission_ave_history.append(ave_cal(self.transmission_history, self.max_steps_per_episode)) # 伝送レートの平均
            self.latency_history.append(self.latency_ave/self.max_steps_per_episode) # 遅延制約の平均
            self.debug_log.write(
                f'Average per Erisode| Reword:{self.reward_ave_history[-1]}, Transmission Rate: {self.transmission_ave_history[-1]}\n'
            )

        """トレー二ングの最後のステップの場合、図を出力"""
        if self.time_in_training == self.total_timesteps:
            generate_training_plot(self.mode, self.graph_file_path, 
                                   self.latency_file, self.network_file, 
                                   self.reward_ave_history, self.error_late_per, self.error_buffer_per, self.transmission_ave_history, self.latency_history)
            generate_cdf_plot(self.mode, self.graph_file_path, 
                                   self.latency_file, self.network_file, 
                                   self.reward_history, self.quality_vc_history, self.jitter_t_history, self.jitter_s_history, self.rebuffer_history)
            
            error_late_ave = sum(self.error_late_per) / len(self.error_late_per)
            error_buffer_ave = sum(self.error_buffer_per) / len(self.error_buffer_per)

            print(f"Average Percentage| Latency Constraint:{error_late_ave:2f}, Rebuffering Penalty:{error_buffer_ave:2f}\n")
        ###################################################################################################

        self.debug_log.write(f"\n")
        return state, reward, all_cal_time, done # 次の状態、報酬、総処理時間、終了フラグ
    
