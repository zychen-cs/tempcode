import matplotlib.pyplot as plt
import numpy as np

# 区域数据
# 我们的方法
#15*10*2 LM
area1=[0.78,1.04]
area2=[0.33,0.32]
area3=[0.60,0.61]
#8*8 LM
area1=[1.57,1.46]
area2=[0.55,0.51]
area3=[0.83,0.92]

#15*10*2
area1=[0.42,0.40]
area2=[0.25,0.30]
area3=[0.29,0.25]
#8*8
area1=[0.36,0.39]
area2=[0.22,0.23]
area3=[0.23,0.23]

lm_15102 = [[0.78, 1.04], [0.33, 0.32], [0.60, 0.61]]
lm_88 = [[1.57, 1.46], [0.55, 0.51], [0.83, 0.92]]

# LM方法
ours_15102 = [[0.42, 0.40], [0.25, 0.30], [0.29, 0.25]]
ours_88 = [[0.36, 0.39], [0.22, 0.23], [0.23, 0.23]]

import matplotlib.pyplot as plt
import numpy as np

labels = [
    '15×10×2_M1', '15×10×2_M2',
    '8×8_M1', '8×8_M2'
]

# 提取数据：每个列表是对应 motion 的值
ours_values = [
    ours_15102[2][0],  # area1_motion1
    ours_15102[2][1],  # area1_motion2
    ours_88[2][0],
    ours_88[2][1]
]

lm_values = [
    lm_15102[2][0],
    lm_15102[2][1],
    lm_88[2][0],
    lm_88[2][1]
]

x = np.arange(len(labels))
width = 0.35
purple = '#5D3A9B'     # Matplotlib's tab:purple
light_green = '#E69F00'  # From ColorBrewer's pastel set
plt.figure(figsize=(8, 5))
plt.bar(x - width/2, ours_values, width, label='w/', color=purple,alpha=0.7)
plt.bar(x + width/2, lm_values, width, label='w/o', color=light_green,alpha=0.7)
# plt.xticks()
plt.yticks(fontsize=20)
plt.xticks(x, labels,fontsize=20,rotation=30)
# plt.ylabel('Value')
# plt.xlabel("Noise type",fontsize=25)
plt.ylabel("Error(cm)",fontsize=25)
plt.legend(fontsize=16,loc="upper right",framealpha=0.5)

plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.savefig("5_3_4_Region3.pdf")
plt.show()
