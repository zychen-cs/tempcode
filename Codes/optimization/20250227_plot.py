
import numpy as np
import matplotlib.pyplot as plt

# 数据列表
zhy = [0.22, 0.17, 0.12, 0.21, 0.23]
cph = [0.29, 0.20, 0.14, 0.26, 0.15]
sww = [0.29, 0.19, 0.14, 0.26, 0.17]
wjk = [0.14, 0.14, 0.14, 0.29, 0.24]
hjy = [0.29, 0.19, 0.16, 0.19, 0.22]
wsy = [0.25, 0.22, 0.26, 0.29, 0.15]
cxm = [0.24, 0.17, 0.10, 0.26, 0.12]
czy = [0.14, 0.15, 0.10, 0.31, 0.14]

# 计算均值
means = [np.mean(zhy), np.mean(cph), np.mean(sww), np.mean(wjk), np.mean(hjy), np.mean(wsy), np.mean(cxm), np.mean(czy)]
print(np.mean(means))
print(np.std(means))
# 设置柱状图标签
labels = ['1', '2', '3', '4', '5', '6', '7', '8']
plt.figure(figsize=(8, 6))
# plt.bar(letters, error_raw, color='royalblue')
# 绘制柱状图
plt.bar(labels, means,width=0.5,color='royalblue')

# 设置横纵轴字体大小
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# 设置图表标题和轴标签（可选）
# plt.title('Mean Values of Lists', fontsize=25)
plt.xlabel('User ID', fontsize=20)
plt.ylabel('Error (cm)', fontsize=20)
plt.grid(axis="y")
# 使图形布局紧凑
plt.tight_layout()
# plt.savefig("Figure17.jpg",dpi=300)
# 显示图表
plt.show()
