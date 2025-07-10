import matplotlib.pyplot as plt
import numpy as np

# 区域数据
# 我们的方法
#ear LM
[2.20,2.03]
#8*4 LM
[2.24,1.41]

#555 LM
[1.51,1.34]

#15102 LM
[2.97,2.71]

#scissor LM
[0.72,0.89]

#watch LM
[0.81,0.60]



#8*4
res =[0.27,0.20]

#5*5*5
res =[0.19,0.22]

#15102
res =[0.16,0.21]

#ear
res =[0.22,0.21]

#scissor
res=[0.31,0.26]

#watch
res=[0.35,0.29]



lm_84 = [[2.24,1.41]]
lm_555 = [[1.51,1.34]]
lm_15102 = [[2.97,2.71]]

lm_ear = [[2.20,2.03]]
lm_scissor = [[0.72,0.89]]
lm_watch = [[0.81,0.60]]

ours_84 = [[0.27,0.20]]
ours_555 = [[0.19,0.22]]
ours_15102 = [[0.16,0.21]]
ours_ear = [[0.22,0.21]]
ours_scissor = [[0.31,0.26]]
ours_watch = [[0.35,0.29]]

import matplotlib.pyplot as plt
import numpy as np

labels = [
    '8×4_M1', '8×4_M2',
    '5×5×5_M1', '5×5×5_M2',
    '15×10×2_M1', '15×10×2_M2',
    'Earbud case_M1', 'Earbud case_M2',
    'Scissor_M1','Scissor_M2',
    'Smartwatch_M1','Smartwatch_M2'
]

# 提取数据：每个列表是对应 motion 的值
ours_values = [
    ours_84[0][0],  # area1_motion1
    ours_84[0][1],  # area1_motion2
    ours_555[0][0],
    ours_555[0][1],
    ours_15102[0][0],  # area1_motion1
    ours_15102[0][1],  # area1_motion2
    ours_ear[0][0],
    ours_ear[0][1],
    ours_scissor[0][0],
    ours_scissor[0][1],
    ours_watch[0][0],  # area1_motion1
    ours_watch[0][1],  # area1_motion2
    
]

lm_values = [
    lm_84[0][0],
    lm_84[0][1],
    lm_555[0][0],
    lm_555[0][1],
    lm_15102[0][0],
    lm_15102[0][1],
    lm_ear[0][0],
    lm_ear[0][1],
    lm_scissor[0][0],
    lm_scissor[0][1],
    lm_watch[0][0],
    lm_watch[0][1],
    
]

x = np.arange(len(labels))
width = 0.35
purple = '#5D3A9B'     # Matplotlib's tab:purple
light_green = '#E69F00'  # From ColorBrewer's pastel set
plt.figure(figsize=(16, 6))
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
plt.savefig("5_3_3_unseennoise.pdf")
plt.show()
