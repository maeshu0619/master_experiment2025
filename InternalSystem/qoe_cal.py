import numpy as np
import colorama

from InternalSystem.gaussian_weight import calculate_video_quality

from System.Client_side.client import Client

class QoE():
    def __init__(self, bitrate_resolution, debug_log, train_or_test, resblock_info,
                 seg_length, latency_constraint):
        self.client = Client

        self.colorama.init(autoreset=True) # 遅延制約超過の場合は文字を赤くする

        ###################################################################################################
        """引数の再定義"""
        self.bitrate_list, self.resolution_list = bitrate_resolution[0], bitrate_resolution[1]
        self.debug_log = debug_log
        self.train_or_test = train_or_test
        self.resblock_info = resblock_info
        self.seg_length = seg_length
        self.latency_constraint = latency_constraint
        ###################################################################################################

        self.alpha = 10 # 動画品質の重み
        self.beta = 2 # 時間ジッタの重み
        self.gamma = 2 # 空間ジッタの重み
        self.sigma = 4.3 * 10 # リバッファリングペナルティの重み。

        self.sigma_h, self.sigma_w = 64, 64 # ガウス分布による重み係数の計算における垂直方向と水平方向の標準偏差。FOCASに基づく。
        
        self.rebuffer = 0

        self.all_cal_time = 0 # 計算量

        self.now_rate = -1 # 現在の解像度
        self.now_resolution = -1 # 現在の動画サイズ
        self.pre_rate = -1 # 1ステップ前の解像度
        
        self.action_invalid_judge = False
        
        self.buffer = 0

    #　QoEの計算関数
    def qoe_cal(self, mode, steps_per_episode, time_in_training, history, gaze_coordinates):


        if mode == 0: # 比較手法ABR
            ###################################################################################################
            """履歴の采配"""
            self.transmissionrate_history = history[0]
            self.bitrate_resolution_history = history[1]
            now_transmissionrate = self.transmissionrate_history[time_in_training] # 現在の伝送レート
            ###################################################################################################
    
            ###################################################################################################
            """各内部情報、状態の取得"""
            self.now_rate = self.bitrate_resolution_history[steps_per_episode][0] # 現在の解像度
            
            self.now_resolution = self.resolution_history[time_in_training] # 現在の動画サイズ
            gaze_yx  = gaze_coordinates[steps_per_episode] # 視線情報取得
            ###################################################################################################

            ###################################################################################################
            """報酬の各項の計算"""
            # 重みづけされた各領域の品質を計算
            wei_quality = calculate_video_quality(gaze_yx, 0, 0, 
                                            0, 0, self.utility(self.now_rate), 
                                            self.sigma_h, self.sigma_w)
            
            # リバッファリングペナルティ
            if time_in_training == 0:
                pass
            else:
                self.rebuffer, zbuffer = self.client.buffering(now_transmissionrate, self.now_rate, buffer)

            # 時間ジッタの計算
            if steps_per_episode == 0:
                jitter_t = 0
            else:
                jitter_t = abs(self.utility(self.now_rate) - self.utility(self.pre_rate))

            # 空間ジッタの計算
            jitter_s = 0 # 空間ジッタは0
            ###################################################################################################


        elif mode == 1: # 比較手法FOCAS
            ###################################################################################################
            """履歴の采配"""
            self.transmissionrate_history = history[0]
            self.ave_quality_history = history[1]
            self.bitrate_resolution_history = history[2]
            self.size_history = history[3]
            self.depth_history = history[4]
            ###################################################################################################
    
            ###################################################################################################
            """各内部情報、状態の取得"""
            self.now_resolution = self.resolution_history[time_in_training] # 現在の動画サイズ（一定）
            resblock_time = self.resblock_info[0] # 1ピクセルがResBlock一層通過する時間
            other_time = self.resblock_info[1] # 畳み込み層のResblock以外の処理時間
            resblock_quality = self.resblock_info[2] # 1ピクセルがResBlock一層通過して向上する品質の倍率
            gaze_yx  = gaze_coordinates[steps_per_episode] # 視線情報取得
            size_fovea = self.size_history[time_in_training][0] # フォビア領域サイズ
            size_blend = self.size_history[time_in_training][1] # ブレンド領域サイズ
            depth_fovea = self.depth_history[time_in_training][0] # フォビア領域深度
            depth_blend = self.depth_history[time_in_training][1] # ブレンド領域深度
            depth_peri = self.depth_history[time_in_training][2] # 周辺領域深度
            ###################################################################################################

            ###################################################################################################
            """各領域のサイズから総処理時間を求め、それが遅延制約を超過するか否か確認する"""
            fovea_area = self.size_via_resolution(gaze_yx, size_fovea, depth_fovea, resblock_time) # フォビア領域のピクセル数
            fovea_time = fovea_area * (depth_fovea - depth_blend) * resblock_time # フォビア領域の計算時間
            blend_area = self.size_via_resolution(gaze_yx, size_blend, depth_blend, resblock_time) # ブレンド領域のピクセル数
            blend_time = blend_area * (depth_blend - depth_peri) * resblock_time # ブレンド領域の計算時間
            peri_time = self.now_resolution[0] * self.now_resolution[1] * depth_peri * resblock_time # 周辺領域の計算時間
            self.all_cal_time = fovea_time + blend_time + peri_time # 総処理時間

            if self.all_cal_time + other_time > self.latency_constraint: # レイテンシ制約超過をした場合、この行動をもう選択しないようにする
                self.action_invalid_judge = True
                self.debug_log.write(f'{colorama.Fore.RED}latency constraint exceedance{colorama.Style.RESET_ALL}\n')
            ###################################################################################################

            ###################################################################################################
            """各領域の解像度を近似的に求める"""
            # 各領域の動画品質サイズを予測計算
            resolution_fovea  = [self.now_resolution[0]*resblock_quality**depth_fovea,
                            self.now_resolution[1]*resblock_quality**depth_fovea]
            resolution_blend  = [self.now_resolution[0]*resblock_quality**depth_blend,
                            self.now_resolution[1]*resblock_quality**depth_blend]
            resolution_peri  = [self.now_resolution[0]*resblock_quality**depth_peri,
                            self.now_resolution[1]*resblock_quality**depth_peri]
            
            # 各領域の動画品質サイズから解像度を計算
            quality_fovea = self.resolution_to_quality(resolution_fovea)
            quality_blend = self.resolution_to_quality(resolution_blend)
            quality_peri = self.resolution_to_quality(resolution_peri)
            ###################################################################################################

            ###################################################################################################
            """各領域のが占めるフレーム内の割り合いからそのフレームの平均品質を求める"""
            # 各領域の動画全体の何％を占めるか計算
            ratio_fovea = fovea_area / (self.now_resolution[0]*self.now_resolution[1])
            ratio_blend = (blend_area-fovea_area) / (self.now_resolution[0]*self.now_resolution[1])

            # 平均品質を求める
            ave_quality = quality_fovea*ratio_fovea + quality_blend*(ratio_blend - ratio_fovea) + quality_peri*(1-ratio_blend)     
            ###################################################################################################
 
            ###################################################################################################
            """報酬の各項の計算"""
            # 重みづけされた各領域の品質を計算
            wei_quality = calculate_video_quality(gaze_yx, size_fovea, size_blend, 
                                            self.utility(quality_fovea), self.utility(quality_blend), self.utility(quality_peri), 
                                            self.sigma_h, self.sigma_w)
            
            # リバッファリングペナルティ
            if now_transmissionrate < self.now_rate: 
                if self.train_or_test == 0:
                    self.rebuffer = 0 # FOCASは通信環境を考慮しないトレーニングを行う
                else:
                    self.rebuffer, buffer = self.calculate_rebuffering_penalty(now_transmissionrate, self.now_rate, buffer)

            # 時間ジッタ
            if steps_per_episode == 0:
                jitter_t = 0
            else:
                jitter_t = abs(ave_quality - self.ave_quality_history[time_in_training-1])

            # 空間ジッタ
            jitter_s = (self.utility(quality_fovea) - self.utility(ave_quality))**2*ratio_fovea + (self.utility(quality_blend) - self.utility(ave_quality))**2*(ratio_blend-ratio_fovea) + (self.utility(quality_peri) - self.utility(ave_quality))**2*(1-ratio_blend)
            ###################################################################################################
 

        elif mode == 2: # 提案手法
            ###################################################################################################
            """履歴の采配"""
            self.transmissionrate_history = history[0]
            self.ave_quality_history = history[1]
            self.bitrate_resolution_history = history[2]
            self.size_history = history[3]
            self.depth_history = history[4]
            ###################################################################################################
    
            ###################################################################################################
            """各内部情報、状態の取得"""
            self.now_resolution = self.resolution_history[time_in_training] # 現在の動画品質サイズ
            resblock_time = self.resblock_info[0] # 1ピクセルがResBlock一層通過する時間
            other_time = self.resblock_info[1] # 畳み込み層のResblock以外の処理時間
            resblock_quality = self.resblock_info[2] # 1ピクセルがResBlock一層通過して向上する品質の倍率
            gaze_yx  = gaze_coordinates[steps_per_episode] # 現在の視線座標
            size_fovea = self.size_history[time_in_training][0] # フォビア領域サイズ
            size_blend = self.size_history[time_in_training][1] # ブレンド領域サイズ
            depth_fovea = self.depth_history[time_in_training][0] # フォビア領域深度
            depth_blend = self.depth_history[time_in_training][1] # ブレンド領域深度
            depth_peri = self.depth_history[time_in_training][2] # 周辺領域深度
            ###################################################################################################

            ###################################################################################################
            """各領域のサイズから総処理時間を求め、それが遅延制約を超過するか否か確認する"""
            fovea_area = self.size_via_resolution(gaze_yx, self.now_resolution, size_fovea, depth_fovea, resblock_time, self.debug_log) # フォビア領域のピクセル数
            fovea_time = fovea_area * (depth_fovea - depth_blend) * resblock_time # フォビア領域の計算時間
            blend_area = self.size_via_resolution(gaze_yx, self.now_resolution, size_blend, depth_blend, resblock_time, self.debug_log) # ブレンド領域のピクセル数
            blend_time = blend_area * (depth_blend - depth_peri) * resblock_time # ブレンド領域の計算時間
            peri_time = self.now_resolution[0] * self.now_resolution[1] * depth_peri * resblock_time # 周辺領域の計算時間
            self.all_cal_time = fovea_time + blend_time + peri_time # 総処理時間
            
            if self.all_cal_time + other_time > self.latency_constraint: # レイテンシ制約超過をした場合、この行動をもう選択しないようにする
                self.action_invalid_judge = True
                self.debug_log.write(f'{colorama.Fore.RED}latency constraint exceedance{colorama.Style.RESET_ALL}\n')
            ###################################################################################################


            ###################################################################################################
            """各領域の解像度を近似的に求める"""
            # 各領域の動画品質サイズを予測計算
            resolution_fovea  = [self.now_resolution[0]*resblock_quality**depth_fovea,
                            self.now_resolution[1]*resblock_quality**depth_fovea]
            resolution_blend  = [self.now_resolution[0]*resblock_quality**depth_blend,
                            self.now_resolution[1]*resblock_quality**depth_blend]
            resolution_peri  = [self.now_resolution[0]*resblock_quality**depth_peri,
                            self.now_resolution[1]*resblock_quality**depth_peri]
            
            # 各領域の動画品質サイズから解像度を計算
            quality_fovea = self.resolution_to_quality(self.bitrate_list, self.resolution_list, resolution_fovea)
            quality_blend = self.resolution_to_quality(self.bitrate_list, self.resolution_list, resolution_blend)
            quality_peri = self.resolution_to_quality(self.bitrate_list, self.resolution_list, resolution_peri)
            ###################################################################################################


            ###################################################################################################
            """各領域のが占めるフレーム内の割り合いからそのフレームの平均品質を求める"""
            # 各領域の動画全体の何％を占めるか計算
            ratio_fovea = fovea_area / (self.now_resolution[0]*self.now_resolution[1])
            ratio_blend = (blend_area-fovea_area) / (self.now_resolution[0]*self.now_resolution[1])

            ave_quality = quality_fovea*ratio_fovea + quality_blend*(ratio_blend - ratio_fovea) + quality_peri*(1-ratio_blend)
            ###################################################################################################

            
            ###################################################################################################
            """報酬の各項の計算"""
            # 重みづけされた各領域の品質を計算
            wei_quality = calculate_video_quality(gaze_yx, size_fovea, size_blend, 
                                            self.utility(quality_fovea), self.utility(quality_blend), self.utility(quality_peri), 
                                            self.sigma_h, self.sigma_w)
            # リバッファリングペナルティ
            if now_transmissionrate < self.now_rate:
                self.rebuffer, buffer = self.calculate_rebuffering_penalty(now_transmissionrate, self.seg_length, buffer)
            
            # 時間ジッタ
            if steps_per_episode == 0:
                jitter_t = 0
            else:
                jitter_t = abs(ave_quality - self.ave_quality_history[time_in_training-1])

            # 空間ジッタ
            jitter_s = (self.utility(quality_fovea) - self.utility(ave_quality))**2*ratio_fovea + (self.utility(quality_blend) - self.utility(ave_quality))**2*(ratio_blend-ratio_fovea) + (self.utility(quality_peri) - self.utility(ave_quality))**2*(1-ratio_blend)
            ###################################################################################################

        

        # 報酬の計算
        reward = self.alpha * wei_quality - self.beta * jitter_t - self.gamma * jitter_s - self.sigma * self.rebuffer

        ###################################################################################################
        """履歴の記録"""
        self.debug_log.write(f'~~~~~~~~~ QoE calculation ~~~~~~~~~\n')

        if mode == 0:
            self.debug_log.write(
                f'Calculation Time: {self.all_cal_time}\n'
                f'Quality(Bitrate)-> {self.now_rate}\n'
                f'Resolution-> {self.now_resolution}\n'
                f'Utility-> {self.utility(self.now_resolution)}\n'
                f'Reward: {reward}, Weighted Quality: {self.utility(self.now_resolution)}, Time Jitter: {jitter_t}, Scale Jitter: {jitter_s}, Rebuffering Penalty: {self.rebuffer}\n'
            )
        elif mode == 1:
            self.debug_log.write(
                f'Calculation Time: {self.all_cal_time}\n'
                f'Quality(Bitrate)-> Fovea: {quality_fovea}, Blend: {quality_blend}, Peripheral: {quality_peri}\n'
                f'Resolution-> Fovea: {resolution_fovea}, Blend: {resolution_blend}, Peripheral: {resolution_peri}\n'
                f'Ratio(/Video Size)-> Fovea: {ratio_fovea}, Blend: {ratio_blend}, Peripheral: 1\n'
                f'Average Quality: {ave_quality}\n'
                f'Utility-> Fovea: {self.utility(quality_fovea)}, Blend: {self.utility(quality_blend)}, Peripheral: {self.utility(quality_peri)}\n'
                f'Reward: {reward}, Weighted Quality: {wei_quality}, Time Jitter: {jitter_t}, Scale Jitter: {jitter_s}, Rebuffering Penalty: {self.rebuffer}\n'
            )
        else:
            self.debug_log.write(
                f'Calculation Time: {self.all_cal_time}\n'
                f'Quality(Bitrate)-> Fovea: {quality_fovea}, Blend: {quality_blend}, Peripheral: {quality_peri}\n'
                f'Resolution-> Fovea: {resolution_fovea}, Blend: {resolution_blend}, Peripheral: {resolution_peri}\n'
                f'Ratio(/Video Size)-> Fovea: {ratio_fovea}, Blend: {ratio_blend}, Peripheral: 1\n'
                f'Average Quality: {ave_quality}\n'
                f'Utility-> Fovea: {self.utility(quality_fovea)}, Blend: {self.utility(quality_blend)}, Peripheral: {self.utility(quality_peri)}\n'
                f'Reward: {reward}, Weighted Quality: {wei_quality}, Time Jitter: {jitter_t}, Scale Jitter: {jitter_s}, Rebuffering Penalty: {self.rebuffer}\n'
            )
            
        self.debug_log.write(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
        ###################################################################################################
        
        # 報酬計算の各項のリスト作成
        qoe_paragraph = [ave_quality, wei_quality, jitter_t, jitter_s, self.rebuffer]

        self.reset()

        return qoe_paragraph, reward, self.all_cal_time, self.action_invalid_judge

    # https://github.com/godka/ABR-DQN.git
    ### Stick: A Harmonious Fusion of Buffer-based and Learning-based Approach for Adaptive Streaming
    # D-DASH: A Deep Q-Learning Framework for DASH Video Streaming
    # リバッファリングペナルティの計算
    def calculate_rebuffering_penalty(self, now_transmissionrate, buffer, max_buffer):
        segment_size = self.now_rate * self.seg_length  # セグメントのサイズ（ビット単位）
        download_time = segment_size / now_transmissionrate  # ダウンロード時間（秒）

        # バッファを更新
        if buffer >= download_time:
            buffer -= download_time  # バッファで補完可能
            self.rebuffer = 0
        else:
            self.rebuffer = download_time - buffer  # 不足分がリバッファリング時間
            buffer = 0  # バッファは空になる

        # バッファを上限までリチャージ
        buffer = min(buffer + self.seg_length, max_buffer)

        return self.rebuffer, buffer


    # 各領域のサイズ（ピクセル数）
    def size_via_resolution(self, gaze_yx, size, depth, resblock_time):
        video_height = self.now_resolution[0]
        video_width = self.now_resolution[1]
        y = int(gaze_yx[0] * video_height / 1080)
        x = int(gaze_yx[1] * video_width / 1920)
        r = size
        # 動画外に視線座標が出ないように矯正
        x = max(0, min(video_width - 1, x))
        y = max(0, min(video_height - 1, y))

        self.debug_log.write(f'Gaze Coordinates in Video-> x: {x}, y: {y}, r: {r}, the Width of Video: {video_width}, the Height of Video: {video_height}\n')

        if r != 0:
            if x - r <= 0:
                if y - r <= 0:
                    area = (x+r)*(y+r)
                elif y - r > 0 and y + r < video_height:
                    area = r*(x+r)
                elif y + r >= video_height:
                    area = (video_height+r-y)*(x+r)
            elif x - r > 0 and x + r < video_width:
                if y - r <= 0:
                    area = r*(y+r)                
                elif y - r > 0 and y + r < video_height:
                    area = r**2
                elif y + r >= video_height:
                    area = (video_height+r-y)*r
            elif x + r >= video_width:
                if y - r <= 0:
                    area = (video_width+r-x)*(y+r)
                elif y - r > 0 and y + r < video_height:
                    area = r*(video_width+r-x)
                elif y + r >= video_height:
                    area = (video_height+r-y)*(video_width+r-x)
        else:
            area = 0

        return area

    # 動画サイズ間の比率から解像度を予測して計算
    def resolution_to_quality(self, bitrate_list):
        rate_index = 0
        over_quality = 0
        ratio = 0  # ratio を初期化しておく
        quality = 0  # デフォルトの品質値を設定しておく
        
        for i in range(len(self.resolution_list)):
            if self.now_resolution[0] < self.resolution_list[i][0]:
                rate_index = i
                break
            rate_index += 1

        if rate_index != len(self.resolution_list):
            if rate_index == 0:
                ratio = self.now_resolution[0] / self.resolution_list[rate_index][0]
                quality = self.bitrate_list[rate_index] * (ratio)
            else:
                ratio = (self.now_resolution[0] - self.resolution_list[rate_index-1][0]) / (self.resolution_list[rate_index][0] - self.resolution_list[rate_index-1][0])
                quality = self.bitrate_list[rate_index-1] * (1 + ratio)
        else:
            ratio = self.now_resolution[0] / self.resolution_list[len(self.resolution_list)-1][0]
            quality = self.bitrate_list[len(self.resolution_list)-1] * ratio
        
        return quality

    # 各領域が動画サイズ内の何％を占めるかを計算
    def area_percentage(self, size):
        ratio = size**2 / (self.now_resolution[0] * self.now_resolution[1])
        return ratio

    # eを底に持った対数計算における効用関数
    def utility(self, bitrate):
        log = np.log(1 + bitrate)
        return log
    
    def reset(self):
        self.action_invalid_judge = False
        
        self.rebuffer = 0 # 再生中断ペナルティの初期化
        self.all_cal_time = 0 # 総処理時間

        self.pre_rate = self.now_rate # 現在の解像度を保存
