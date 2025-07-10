import matplotlib.pyplot as plt
import numpy as np

# 数据
res1 = [0.29, 0.30]  # 8*2
res2 = [0.23, 0.27]  # 8*8

labels = ['motion1', 'motion2']
x = np.arange(len(labels))  # [0, 1]
width = 0.35  # 柱子的宽度

# 绘图
plt.figure(figsize=(8, 6))
plt.bar(x - width/2, res1, width, label='8×2', color='skyblue')
plt.bar(x + width/2, res2, width, label='8×8', color='salmon')

# 标签和标题
plt.ylabel('Value')
plt.title('Comparison of 8×2 and 8×8 on State3')
plt.xticks(x, labels, fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(0, max(max(res1), max(res2)) + 0.05)
plt.legend(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
