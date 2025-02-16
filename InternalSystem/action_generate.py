bitrate_list = [0,1,2,3]
depth_fovea_list = [6,7,8,9,10] # フォビア深度リスト
depth_blend_list = [3,4,5,6,7] # ブレンド深度リスト
depth_peri_list = [1,2,3,4,5] # 周辺深度リスト
size_list = [x / 12 for x in [0,1,2,3,4,5]] # サイズリスト


def focas_combination():
    all_combi = []
    combi = []
    for size_fovea in range(5):
        combi.append(size_fovea)
        for size_blend in range(size_fovea, 5):
            combi.append(size_blend)

            for depth_fovea in range(5):
                for depth_blend in range(5):
                    for depth_peri in range(5):
                        if depth_fovea_list[depth_fovea] > depth_blend_list[depth_blend] > depth_peri_list[depth_peri]:
                            combi.append(depth_fovea)
                            combi.append(depth_blend)
                            combi.append(depth_peri)

                            all_combi.append(combi)
                            combi = combi[:-3]

            combi = combi[:-1]
        combi = combi[:-1]

    return all_combi

def ours_combination():
    all_combi = []
    combi = []
    for bitrate in range(4):
        combi.append(bitrate)
        for size_fovea in range(5):
            combi.append(size_fovea)
            for size_blend in range(size_fovea, 5):
                combi.append(size_blend)

                for depth_fovea in range(5):
                    for depth_blend in range(5):
                        for depth_peri in range(5):
                            if depth_fovea_list[depth_fovea] > depth_blend_list[depth_blend] > depth_peri_list[depth_peri]:
                                combi.append(depth_fovea)
                                combi.append(depth_blend)
                                combi.append(depth_peri)

                                all_combi.append(combi)
                                combi = combi[:-3]

                combi = combi[:-1]
            combi = combi[:-1]
        combi = combi[:-1]

    return all_combi

def assign_action(self, mode, action): # 行動の割り当て

    if mode == 0:
        size_index = [ # サイズインデックス
            self.action_comb[action][0], # 中心窩領域
            self.action_comb[action][1] # 中間領域
        ]

        depth_index = [ # 深さインデックス
            self.action_comb[action][2], # 中心窩領域
            self.action_comb[action][3], # 中間領域
            self.action_comb[action][4] # 周辺領域
        ]

        index = [ # 総インデックス
            size_index,
            depth_index
        ]
        
    elif mode == 1:
        size_index = [ # サイズインデックス
            self.action_comb[action][0], # 中心窩領域
            self.action_comb[action][1] # 中間領域
        ]

        depth_index = [ # 深さインデックス
            self.action_comb[action][2], # 中心窩領域
            self.action_comb[action][3], # 中間領域
            self.action_comb[action][4] # 周辺領域
        ]

        index = [ # 総インデックス
            size_index,
            depth_index
        ]

    else:
        bitrate_index = self.action_comb[action][0] # 選択された解像度

        size_index = [ # サイズインデックス
            self.action_comb[action][1], # 中心窩領域
            self.action_comb[action][2] # 中間領域
        ]

        depth_index = [ # 深さインデックス
            self.action_comb[action][3], # 中心窩領域
            self.action_comb[action][4], # 中間領域
            self.action_comb[action][5] # 周辺領域
        ]

        index = [ # 総インデックス
            bitrate_index,
            size_index,
            depth_index
        ]

    return index