import numpy as np
import os
import datetime

from gym import Env
from gym.spaces import Discrete, Box

from InternalSystem.file_setup import file_setup
from InternalSystem.gaze_prediction import gaze_data
from InternalSystem.graph_plot import generate_cdf_plot, generate_training_plot
from InternalSystem.action_generate import focas_combination, a_focas_combination, assign_action
from InternalSystem.rate_simu import simulate_transmission_rate
from InternalSystem.other_cal import ave_cal

from InternalSystem.qoe_cal import QoE
from DQN.agent import Actor
from System.Server_side.server import Server
from System.Client_side.client import Client

class Environment(Env):
    """初期定義"""
    def __init__(self, mode, train_or_test, latency_file, network_file, 
                 total_timesteps, max_steps_per_episode, latency_constraint, fps, 
                 mu, sigma_ratio, base_band):
        super(Environment, self).__init__()
        self.server = Server
        self.client = Client
        
        # 行動 # 無効な行動の記録の為に必要
        self.actor = Actor(mode)

        self.time_in_training = 0 # トレーニング全体における現在のステップ数 # 初期化しない
        self.steps_per_episode = 0 # 各エピソードごとの現在のステップ数 # エピソード終了時に初期化

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


        ###################################################################################################
        """各手法の行動、状態範囲の定義"""
        if mode == 0: # 比較手法ABR
            mode_name = "ABR"
            print(f'action length is 4')
            self.action_space = Discrete(4)
            self.observation_space = Box(
                low=0,
                high=np.inf,
                shape=(2,), # 現在の伝送レート，選択された品質
                dtype=np.float32
            )
        elif mode == 1: # 比較手法FOCAS
            mode_name = "FOCAS"
            self.action_comb = focas_combination() # 行動範囲の組み合わせ
            self.focas_bitrate_index = 1 # FOCASの入力動画の解像度は固定
            print(f'action length is {len(self.action_comb)}, focas video resolution index: {self.focas_bitrate_index}')
            self.action_space = Discrete(1200)
            self.observation_space = Box(
                low=0,
                high=np.inf,
                shape=(2,), # 選択された品質、視線座標
                dtype=np.float32
            )
        elif mode == 2: # 提案手法
            mode_name = "Ours"
            self.action_comb = a_focas_combination() # 行動範囲の組み合わせ
            print(f'action length is {len(self.action_comb)}')
            self.action_space = Discrete(4800)
            self.observation_space = Box(
                low=0,
                high=np.inf,
                shape=(4,), # 現在の伝送レート，選択された品質，現在の視線座標
                dtype=np.float32
            )

        if self.train_or_test == 0:
            output_file="graph_train"
        elif self.train_or_test == 1:
            output_file="graph_test"
        ###################################################################################################


        ###################################################################################################
        """シミュレーションデータの取得"""
        # 伝送レートシミュレート
        self.min_dis = 150
        self.max_dis = 1000
        self.distance = np.random.uniform(self.min_dis, self.max_dis)
        self.transmissionrate_list = simulate_transmission_rate(self.distance, self.max_steps_per_episode, self.mu, self.sigma_ratio, self.base_band)

        # 視線情報の取得
        self.directory_path = "UD_UHD_EyeTrakcing_Videos/Gaze_Data/HD"
        self.gaze_coordinates = gaze_data(self.directory_path, max_steps_per_episode, video_center=(960, 540))
        print("Gaze Cordinate cathed\n")
        ###################################################################################################


        ###################################################################################################
        """超解像におけるさまざまなパラメータの定義"""
        # 解像度と動画サイズの関係定義
        self.bitrate_resolution = {
            [250, [375, 666]],
            [500, [540, 960]],
            [1000, [750, 1333]],
            [2000, [1080, 1920]],
            #[4000, [1500, 2666]],
            #[8000, [2160, 3840]]
        }

        self.segment_length = 2 # 2ステップ分再生される

        # ResBlock1ピクセル当たり一層通過させるのに必要な計算時間と、それによって何倍解像度が向上するかの情報
        self.resblock_time = 2.525720165e-8 #2.525720165e-8 # 1.078679591e-9 # 4.487906e-8
        self.other_time = 0 #0.04784066667 # 0.0167315
        self.resblock_quality = 1.148698354997035
        self.resblock_info = (self.resblock_time, self.other_time, self.resblock_quality)

        self.depth_fovea_list = [6,7,8,9,10] # フォビア深度リスト
        self.depth_blend_list = [3,4,5,6,7] # ブレンド深度リスト
        self.depth_peri_list = [1,2,3,4,5] # 周辺深度リスト
        self.size_list = [x / 12 for x in [0,1,2,3,4,5]] # サイズリスト
        ###################################################################################################

        ###################################################################################################
        """履歴記録のためのリストの定義"""
        # グラフ生成アドレスの定義
        self.current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # 現在の時刻を取得    
        self.graph_file_path = os.path.join(output_file, f"{mode_name}/{self.latency_file}/{self.network_file}/{self.current_time}") # 出力フォルダの作成

        # ログファイルのセットアップ
        self.log_file, self.debug_log, self.logger = file_setup(self.mode, self.train_or_test, self.current_time, self.latency_file, self.network_file)

        # 遅延制約超過のエラー比率計算
        self.error_late = 0
        self.error_late_per = []
        # 伝送レート超過のエラー比率計算
        self.error_buffer = 0
        self.error_buffer_per = []

        self.history # 履歴

        self.ave_quality_history = [] # 報酬の平均品質の履歴
        self.wei_quality_history = [] # 報酬の重みづけされた平均品質の履歴
        self.jitter_t_history = [] # 報酬の時間ジッタの履歴
        self.jitter_s_history = [] # 報酬の空間ジッタの履歴
        self.rebuffer_history = [] # 報酬のリバッファリングペナルティ

        self.bitrate_history = [] # 選択された品質の履歴
        self.resolution_history = [] # 選択された動画サイズの履歴
        self.size_history = [] # 選択された領域サイズの履歴
        self.depth_history = [] # 選択された深さの履歴
        self.transmissionrate_history = [] # シミュレートした伝送レートの履歴

        self.latency_ave = 0
        self.latency_history = [] # 遅延の履歴
        

        self.q_values = []  # Q値の履歴を保持
        self.action_history = [] # 行動の履歴
        self.reward_history = [] # 報酬の履歴
        self.reward_ave_history = [] # 各エピソードの報酬の平均の履歴
        self.transmissionrate_ave_history = [] # 各エピソードの伝送レートの平均の履歴
        ###################################################################################################

        self.qoe = QoE(self.bitrate_resolution, self.debug_log, self.train_or_test, self.resblock_info, 
                       self.seg_length, self.latency_constraint) # 報酬計算クラスの定義

    """初期化"""
    def reset(self):
        if self.steps_per_episode != 0: # トレーニングの一番初めでない場合
            # 遅延制約超過記録
            self.error_late_per.append(100*self.error_late/self.steps_per_episode)
            self.error_late = 0
            # 伝送レート超過記録
            self.error_buffer_per.append(100*self.error_buffer/self.steps_per_episode)
            self.error_buffer = 0

        if self.mode == 0: # 比較手法ABR
            self.bitrate_history.append(self.bitrate_list[0])  # 初期ビットレートを設定
            self.gaze_coordinates = gaze_data(self.directory_path, self.max_steps_per_episode, video_center=(960, 540))
            self.distance = np.random.uniform(self.min_dis, self.max_dis)
            self.transmissionrate_list = simulate_transmission_rate(self.distance, self.max_steps_per_episode, self.mu, self.sigma_ratio, self.base_band)
            state = np.array([self.transmissionrate_list[0], self.bitrate_history[0]], dtype=np.float32) # 状態の初期化
        elif self.mode == 1: # 比較手法FOCAS
            self.bitrate_history.append(self.bitrate_list[self.focas_bitrate_index])  # 初期ビットレートを設定
            self.gaze_coordinates = gaze_data(self.directory_path, self.max_steps_per_episode, video_center=(960, 540))  
            self.distance = np.random.uniform(self.min_dis, self.max_dis)
            self.transmissionrate_list = simulate_transmission_rate(self.distance, self.max_steps_per_episode, self.mu, self.sigma_ratio, self.base_band)
            state = np.array([self.gaze_coordinates[0][0], self.gaze_coordinates[0][1]], dtype=np.float32) # 状態の初期化
        elif self.mode == 2: # 提案手法
            self.bitrate_history.append(self.bitrate_list[0])  # 初期ビットレートを設定
            self.gaze_coordinates = gaze_data(self.directory_path, self.max_steps_per_episode, video_center=(960, 540))        
            self.distance = np.random.uniform(self.min_dis, self.max_dis)
            self.transmissionrate_list = simulate_transmission_rate(self.distance, self.max_steps_per_episode, self.mu, self.sigma_ratio, self.base_band)
            state = np.array([self.transmissionrate_list[0], self.bitrate_history[0], self.gaze_coordinates[0][0], self.gaze_coordinates[0][1]], dtype=np.float32) # 状態の初期化

        self.bitrate_history = [] # 選択されたビットレート履歴の初期化
        self.steps_per_episode = 0
        action_invalid_judge = False
        done = False
            
        self.log_file.write(f"--- state reset ---\n")
        self.debug_log.write(f"--- state reset ---\n")
        return state

    """各ステップで行う操作"""
    def step(self, action, goal_reward):
        self.debug_log.write(f'current step is {self.time_in_training+1} / {self.total_timesteps+1}\n')
        reward = 0 # 報酬の初期化

        ###################################################################################################
        """各手法の行動の割り当てと記録、次の状態の更新"""
        if self.mode == 0: # 比較手法ABR
            self.debug_log.write(f'action: {action}\n')

            # 伝送レートの変化をシミュレーション
            
            self.history = [
                self.transmissionrate_history.append(self.transmissionrate_list[self.steps_per_episode]),
                self.bitrate_resolution_history.append([self.bitrate_resolution[action][0], self.bitrate_resolution[action][1]]) # 解像度、動画サイズ履歴の記録
            ]

            # 次の状態の記録   
            next_state = np.array([self.transmissionrate_history[-1], # 伝送レート
                              self.bitrate_history[-1]], # 要求したビットレート
                              dtype=np.float32)
            
        elif self.mode == 1: # 比較手法FOCAS
            self.debug_log.write(f'action: {action}\n')

            # 行動の数値を各情報に割り当てる
            index = assign_action(self.mode, action) # 行動の割り当て

            self.history = [
                self.transmissionrate_history.append(self.transmissionrate_list[self.steps_per_episode]), # 伝送レートの変化をシミュレーション
                self.ave_quality_history, # 平均品質
                self.bitrate_resolution_history.append([self.bitrate_resolution[self.focas_bitrate_index][0], self.bitrate_resolution[self.focas_bitrate_index][1]]), # 解像度、動画サイズ履歴の記録
                self.size_history.append([int(self.size_list[index[1][0]] * self.bitrate_resolution[self.focas_bitrate_index][1][0]),  # サイズ履歴の記録
                                int(self.size_list[index[1][1]] * self.bitrate_resolution[self.focas_bitrate_index][1][0])]),
                self.depth_history.append([self.depth_fovea_list[index[2][0]], # 深さ履歴の記録
                                self.depth_blend_list[index[2][1]], 
                                self.depth_peri_list[index[2][2]]])
            ]

            # 次の状態の記録
            next_state = np.array([self.gaze_coordinates[self.steps_per_episode][0], # 焦点のy座標
                              self.gaze_coordinates[self.steps_per_episode][1]], # 焦点のx座標
                              dtype=np.float32)
            
        elif self.mode == 2: # 提案手法
            self.debug_log.write(f'action: {action}\n')

            # 行動の数値を各情報に割り当てる
            index = assign_action(self.mode, action) # 行動の割り当て

            self.history = [
                self.transmissionrate_history.append(self.transmissionrate_list[self.steps_per_episode]), # 伝送レートの変化をシミュレーション
                self.ave_quality_history, # 平均品質
                self.bitrate_resolution_history.append([self.bitrate_resolution[index[0]][0], self.bitrate_resolution[index[0]][1]]), # 解像度、動画サイズ履歴の記録
                self.size_history.append([int(self.size_list[index[1][0]] * self.bitrate_resolution[index[0]][1][0]), # サイズ履歴の記録
                                int(self.size_list[index[1][1]] * self.bitrate_resolution[index[0]][1][0])]), 
                self.depth_history.append([self.depth_fovea_list[index[2][0]], # 深さ履歴の記録
                                self.depth_blend_list[index[2][1]], 
                                self.depth_peri_list[index[2][2]]])
            ]
            
            # 次の状態の記録
            next_state = np.array([self.transmissionrate_history[-1], # 伝送レート
                              self.bitrate_history[-1], # 要求したビットレート
                              self.gaze_coordinates[self.steps_per_episode][0], # 焦点のy座標
                              self.gaze_coordinates[self.steps_per_episode][1]], # 焦点のx座標
                              dtype=np.float32)
        ###################################################################################################

        # QoE計算
        qoe_paragraph, reward, all_cal_time, action_invalid_judge = self.qoe.qoe_cal(self.mode, self.steps_per_episode, self.time_in_training, self.history, self.gaze_coordinates)
        
        # 無効な行動を記録 # 遅延制約を超過する行動の記録
        if action_invalid_judge:
            self.error_late += 1 # 遅延制約違反回数を記録
            self.actor.add_invalid_action(action)

        self.latency_ave += all_cal_time # 処理時間の和

        if qoe_paragraph[3] > 0: # 再生中断ペナルティが発生している場合
            self.error_buffer += 1 # ペナルティの発生回数を1つ追加

        ###################################################################################################
        """履歴の記録"""
        # 報酬の内訳の保存
        self.ave_quality_history.append(qoe_paragraph[0]) # 平均品質の履歴記録
        self.wei_quality_history.append(qoe_paragraph[1]) # 重みづけされた平均品質の履歴記録
        self.jitter_t_history.append(qoe_paragraph[2]) # 時間ジッタの履歴記録
        self.jitter_s_history.append(qoe_paragraph[3]) # 空間ジッタの履歴記録
        self.rebuffer_history.append(qoe_paragraph[4]) # 再生中断ペナルティの履歴記録
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
        if self.mode == 0:
            pass
        elif self.mode == 1:
            pass
        else:
            self.debug_log.write(
                f'Bitrate -> {index[0]}\n'
                f'Size -> fovea:{index[1][0]}, blend:{index[1][1]}\n'
                f'Depth -> fovea:{index[2][0]}, blend:{index[2][1]}, peri:{index[2][2]}\n'
                f'Transmission Rate: {self.transmission_list[self.steps_per_episode][0]}, {self.transmission_list[self.steps_per_episode][1]})\n'
                f'Gaze Coordinate(y,x): ({self.gaze_coordinates[self.steps_per_episode][0]}, {self.gaze_coordinates[self.steps_per_episode][1]})\n'
            )
        ###################################################################################################
        
        # 終了フラグの判定
        done = (np.mean(self.reward_history) >= goal_reward)  # 目標報酬に達したら終了

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
                                   self.reward_history, self.wei_quality_history, self.jitter_t_history, self.jitter_s_history, self.rebuffer_history)
            
            error_late_ave = sum(self.error_late_per) / len(self.error_late_per)
            error_buffer_ave = sum(self.error_buffer_per) / len(self.error_buffer_per)

            print(f"Average Percentage| Latency Constraint:{error_late_ave:2f}, Rebuffering Penalty:{error_buffer_ave:2f}\n")
        ###################################################################################################


        self.debug_log.write(f"\n")
        return next_state, reward, all_cal_time, done # 次の状態、報酬、総処理時間、終了フラグ
    
