import matplotlib.pyplot as plt
import numpy as np

# 数据准备同上
labels = ['motion1_pos', 'motion2_pos', 'motion1_ori', 'motion2_ori']
magnet_types = ['8×2', '8×8']
algorithm_status = ['w/','w/o']
motion1_ori = [4.63,4.78,5.72,  4.53,5.26,6.70]
motion2_ori = [4.89,4.92,6.29,   3.90,5.32,6.58]
purple = '#5D3A9B'     # Matplotlib's tab:purple
light_green = '#E69F00'  # From ColorBrewer's pastel set
data = {
    '8×2': {
        'w/o': [
            [0.93, 0.65, 0.53, 3.58, 2.49, 2.14],
            [0.64, 0.48, 0.51, 3.13, 2.29, 2.02],
            [5.49, 3.18, 3.46, 12.35, 9.35, 7.47],
            [8.98, 4.58, 4.36, 17.72, 18.46, 11.26]
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

# 子图函数（只比较 w/o 和 w/）
def plot_comparison_by_magnet(magnet, metric_index, ylabel):
    bar_width = 0.35
    index = np.arange(6)

    fig, ax = plt.subplots(figsize=(8, 4))

    for i, algo in enumerate(algorithm_status):
        values = data[magnet][algo][metric_index]
        offset = i * bar_width
        if(algo=="w/"):
            ax.bar(index + offset, values, bar_width, label=f'{algo}',color=purple,alpha=0.7)
        if(algo=="w/o"):
            ax.bar(index + offset, values, bar_width, label=f'{algo}',color=light_green,alpha=0.7)

    ax.set_ylabel(ylabel,fontsize=22)
    ax.tick_params(axis='both', labelsize=22)

    ax.set_xticks(index + bar_width / 2,fontsize=22)
    ax.set_xticklabels([f'Pos{i+1}' for i in range(6)])
    ax.legend(fontsize=20, loc='upper right',framealpha=0.5)
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

# 绘制每种磁铁尺寸下的每项指标（motion1_pos, motion2_pos, motion1_ori, motion2_ori）
for magnet in magnet_types:
    plot_comparison_by_magnet(magnet, 0, 'Position Error (cm)')  # motion1_pos
    plot_comparison_by_magnet(magnet, 1, 'Position Error (cm)')  # motion2_pos
    plot_comparison_by_magnet(magnet, 2, 'Orientation Error (°)')  # motion1_ori
    plot_comparison_by_magnet(magnet, 3, 'Orientation Error (°)')  # motion2_ori
