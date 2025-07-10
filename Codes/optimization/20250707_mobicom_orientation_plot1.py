import matplotlib.pyplot as plt
import numpy as np

# 区域数据
# 我们的方法
#ear LM
[2.26,1.57]
#8*4 LM
[2.33,3.55]

#ear
res =[0.25,0.30]

#8*4
res =[0.26,0.23]


lm_ear = [[2.26, 1.57]]
lm_84 = [[2.33,3.55]]

# LM方法
ours_ear = [[0.25,0.30]]
ours_84 = [[0.26,0.23]]

import matplotlib.pyplot as plt
import numpy as np

labels = [
    'Earbud case_M1', 'Earbud case_M2',
    '8×4_M1', '8×4_M2'
]

# 提取数据：每个列表是对应 motion 的值
ours_values = [
    ours_ear[0][0],  # area1_motion1
    ours_ear[0][1],  # area1_motion2
    ours_84[0][0],
    ours_84[0][1]
]

lm_values = [
    lm_ear[0][0],
    lm_ear[0][1],
    lm_84[0][0],
    lm_84[0][1]
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
plt.savefig("5_3_5_orientation.pdf")
plt.show()
