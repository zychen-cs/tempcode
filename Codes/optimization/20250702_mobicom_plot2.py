import matplotlib.pyplot as plt
import numpy as np

axes = ['X', 'Y', 'Z']
x = np.arange(len(axes))
bar_width = 0.35

# Near 的误差数据：索引 = 2
#8*8
pos_w = [0.90, 0.70, 0.75]
pos_wo = [315.97, 61.95, 201.61]

ori_w = [1.96, 1.80, 1.84]
ori_wo = [38.27, 25.07, 28.49]


#8*2
pos_w = [0.81, 0.75, 0.87]

pos_wo = [42.43, 1.35, 1.51]

ori_w = [2.02, 1.64, 1.55]

ori_wo = [18.02, 4.43, 7.97]

# ---------- 图 1：Position Error ----------
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - bar_width/2, pos_w, width=bar_width, label='w/', color='#5D3A9B',alpha=0.7)
ax.bar(x + bar_width/2, pos_wo, width=bar_width, label='w/o', color='#E69F00',alpha=0.7)

# ax.set_xlabel('Axis', fontsize=22)
ax.set_ylabel('Position Error (cm)', fontsize=22)
ax.tick_params(axis='x', labelsize=22)   # X-axis tick font size
ax.tick_params(axis='y', labelsize=22)   # Y-axis tick font size
# ax.set_title('Position Error (Near Distance)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(axes)
ax.legend(fontsize=18,framealpha=0.5,loc="upper right")
# ax.legend(title='Distance',fontsize=16,title_fontsize=18,loc="upper right",framealpha=0.5)
ax.grid(axis='y', linestyle='--', alpha=0.4)

# 添加误差标签
for i in range(len(axes)):
    ax.text(x[i] - bar_width/2, pos_w[i] + 1, f'{pos_w[i]:.2f}', ha='center', fontsize=18)
    ax.text(x[i] + bar_width/2, pos_wo[i] + 1, f'{pos_wo[i]:.2f}', ha='center', fontsize=18)

plt.tight_layout()
plt.savefig("5_3_2_pos82_compare.pdf")
plt.show()

# ---------- 图 2：Orientation Error ----------
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - bar_width/2, ori_w, width=bar_width, label='w/', color='#5D3A9B',alpha=0.7)
ax.bar(x + bar_width/2, ori_wo, width=bar_width, label='w/o', color='#E69F00',alpha=0.7)

# ax.set_xlabel('Axis', fontsize=22)
ax.set_ylabel('Orientation Error (°)', fontsize=22)
ax.tick_params(axis='x', labelsize=22)   # X-axis tick font size
ax.tick_params(axis='y', labelsize=22)   # Y-axis tick font size
# ax.set_title('Orientation Error (Near Distance)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(axes)
ax.legend(fontsize=18,framealpha=0.5,loc="upper right")
# ax.legend(title='Distance',fontsize=16,title_fontsize=18,loc="upper right",framealpha=0.5)
ax.grid(axis='y', linestyle='--', alpha=0.4)

for i in range(len(axes)):
    ax.text(x[i] - bar_width/2, ori_w[i] + 0.2, f'{ori_w[i]:.2f}', ha='center', fontsize=18)
    ax.text(x[i] + bar_width/2, ori_wo[i] + 0.2, f'{ori_wo[i]:.2f}', ha='center', fontsize=18)

plt.tight_layout()
plt.savefig("5_3_2_ori82_compare.pdf")
plt.show()
