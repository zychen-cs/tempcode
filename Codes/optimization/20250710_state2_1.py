import matplotlib.pyplot as plt
import numpy as np

# 数据准备
magnet_types = ['8×2', '8×8']
algorithm_status = ['w/', 'w/o']
purple = '#5D3A9B'
light_green = '#E69F00'

# 原始数据结构
data = {
    '8×2': {
        'w/o': [
            [0.93, 0.65, 0.53, 3.58, 2.49, 2.14],  # motion1_pos
            [0.64, 0.48, 0.51, 3.13, 2.29, 2.02],  # motion2_pos
            [5.49, 3.18, 3.46, 12.35, 9.35, 7.47], # motion1_ori
            [8.98, 4.58, 4.36, 17.72, 18.46, 11.26]# motion2_ori
        ],
        'w/': [
            [0.07, 0.07, 0.14, 0.19, 0.32, 0.26],
            [0.12, 0.05, 0.19, 0.22, 0.07, 0.39],
            [4.63, 4.78, 5.72, 4.53, 5.26, 6.70],
            [4.89, 4.92, 6.29, 3.90, 5.32, 6.58]
        ]
    },
    '8×8': {
        'w/o': [
            [3.20, 1.84, 1.70, 6.82, 5.06, 4.46],
            [3.56, 1.57, 1.59, 7.16, 5.31, 4.44],
            [30.33, 15.18, 10.33, 31.25, 14.53, 5.36],
            [74.3, 25.11, 13.23, 25.51, 19.04, 9.49]
        ],
        'w/': [
            [0.28, 0.15, 0.27, 0.15, 0.19, 0.14],
            [0.29, 0.06, 0.29, 0.39, 0.33, 0.17],
            [4.58, 5.68, 5.53, 5.66, 6.88, 7.43],
            [4.82, 5.53, 6.19, 6.06, 7.03, 6.58]
        ]
    }
}

# 改：每对 motion 数据求均值
def averaged_data(original_data):
    result = {}
    for magnet in magnet_types:
        result[magnet] = {}
        for algo in algorithm_status:
            pos = np.mean([original_data[magnet][algo][0], original_data[magnet][algo][1]], axis=0)
            ori = np.mean([original_data[magnet][algo][2], original_data[magnet][algo][3]], axis=0)
            result[magnet][algo] = {'pos': pos, 'ori': ori}
    return result

avg_data = averaged_data(data)

# 绘图函数（合并 motion1 和 motion2）
def plot_merged_comparison(magnet, metric, ylabel):
    bar_width = 0.35
    index = np.arange(6)
    
    fig, ax = plt.subplots(figsize=(8, 4))

    for i, algo in enumerate(algorithm_status):
        values = avg_data[magnet][algo][metric]
        offset = i * bar_width
        color = purple if algo == 'w/' else light_green
        ax.bar(index + offset, values, bar_width, label=algo, color=color, alpha=0.7)

    ax.set_ylabel(ylabel, fontsize=22)
    ax.tick_params(axis='both', labelsize=22)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels([f'Pos{i+1}' for i in range(6)], fontsize=22)
    ax.legend(fontsize=20, loc='upper right', framealpha=0.5)
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

# 画图：每个磁铁分别画 position + orientation
for magnet in magnet_types:
    plot_merged_comparison(magnet, 'pos', 'Position Error (cm)')
    plot_merged_comparison(magnet, 'ori', 'Orientation Error (°)')
