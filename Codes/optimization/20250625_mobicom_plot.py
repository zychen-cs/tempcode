import matplotlib.pyplot as plt
import numpy as np

# Data for the three unseen regions



#不同区域
# regions = ['Region 1', 'Region 2', 'Region 3']
# mse = [20.897, 5.558, 3.805]
# mae = [2.685, 1.654, 1.378]

#不同干扰源
regions = ['8*4', '5*5*5', '15*10*2']
mse = [11.28, 7.77, 13.70]
mae = [2.16, 1.85, 2.28]


#不同方向
regions = ['8*2', '5*5*5', '8*4']
mse = [11.81, 15.49, 40.21]
mae = [2.19, 2.41, 3.55]

#不同训练策略
regions = ['Training on\nsynthetic data',
           'Training on\nreal-world data',
           'Fine-tuning']
mse = [12.36, 2.94, 5.74]
mae = [1.94, 1.14, 1.61]


#微调模型在不同干扰源的精度

regions = ['8*8', '8*2', 'Left','Right']
mse = [11.46,4.62, 8.01,8.07]
mae = [2.30, 1.52, 1.90,1.92]

# Position and bar width
x = np.arange(len(regions))
width = 0.35

# Custom colors: purple and light green
purple = '#5D3A9B'     # Matplotlib's tab:purple
light_green = '#E69F00'  # From ColorBrewer's pastel set

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 5))

# Plot bars
rects1 = ax.bar(x - width/2, mse, width, label='MSE', color=purple,alpha=0.7)
rects2 = ax.bar(x + width/2, mae, width, label='MAE', color=light_green,alpha=0.7)

# Labels and title
ax.set_ylabel('Error', fontsize=20)
# ax.set_title('Reconstruction Error in Unseen Regions with Magnetic Interference', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(regions, fontsize=20)
ax.legend(fontsize=20, loc='upper right',framealpha=0.5)

# Tick labels (the values along axes)
ax.tick_params(axis='x', labelsize=20)   # X-axis tick font size
ax.tick_params(axis='y', labelsize=20)   # Y-axis tick font size
# Add value labels on bars
# def add_labels(rects):
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate(f'{height:.2f}',
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),
#                     textcoords="offset points",
#                     ha='center', va='bottom', fontsize=9)

# add_labels(rects1)
# add_labels(rects2)

# Aesthetics
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.6)
# plt.savefig("4_2_specific_results.pdf")

plt.show()
