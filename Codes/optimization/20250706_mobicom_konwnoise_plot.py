import matplotlib.pyplot as plt
import numpy as np

# Data for the three unseen regions



#state2
# regions = ['1', '2', '3','4','5','6']
# regions = ['Motion 1', 'Motion 2']
regions = ['Pos 1', 'Pos 2']
#8*2 LM
motion1_pos = [0.93,0.65,0.53,   3.58,2.49,2.14]
motion1_ori = [5.49,3.18,3.46,   12.35,9.35,7.47]

motion2_pos = [0.64,0.48,0.51,    3.13,2.29,2.02]
motion2_ori = [8.98,4.58,4.36,     17.72,18.46,11.26]

#8*8 LM

motion1_pos = [3.20,1.84,1.70,      6.82,5.06,4.46]
motion1_ori = [30.33,15.18,10.33,         31.25,14.53,5.36]

motion2_pos = [3.56,1.57,1.59,      7.16,5.31,4.44]
motion2_ori = [74.3,25.11,13.23,      25.51,19.04,9.49]



#8*2
motion1_pos = [0.07,0.07,0.14,  0.19,0.32,0.26]
motion2_pos = [0.12,0.05,0.19,   0.22,0.07,0.39]

motion1_ori = [4.63,4.78,5.72,  4.53,5.26,6.70]
motion2_ori = [4.89,4.92,6.29,   3.90,5.32,6.58]

#8*8
motion1_pos = [0.28,0.15,0.27,    0.15,0.19,0.14]
motion2_pos = [0.29,0.06,0.29,     0.39,0.33,0.17]

motion1_ori = [4.58,5.68,5.53,     5.66,6.88,7.43]
motion2_ori = [4.82,5.53,6.19,     6.06,7.03,6.58]

#8*2 state3
res1=[0.29,0.30]

#8*8 state3
res2=[0.23,0.27]


# #8*2 state4
# res1=[0.26,0.20]

# #8*8 state4
# res2=[0.22,0.21]

# Position and bar width
x = np.arange(len(regions))
width = 0.35

# Custom colors: purple and light green
purple = '#5D3A9B'     # Matplotlib's tab:purple
light_green = '#E69F00'  # From ColorBrewer's pastel set

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 5))

# Plot bars
# rects1 = ax.bar(x - width/2, motion1_ori, width, label='Motion 1', color=purple,alpha=0.7)
# rects2 = ax.bar(x + width/2, motion2_ori, width, label='Motion 2', color=light_green,alpha=0.7)

rects1 = ax.bar(x - width/2, res1, width, label='8*2', color=purple,alpha=0.7)
rects2 = ax.bar(x + width/2, res2, width, label='8*8', color=light_green,alpha=0.7)
# Labels and title

# ax.set_ylabel('Position Error(cm)', fontsize=20)
# ax.set_ylabel('Orientation Error (°)', fontsize=20)
ax.set_ylabel('Error(cm)', fontsize=20)
ax.set_xlabel('Position ID', fontsize=20)
# ax.set_xlabel('Interference Motion', fontsize=20)
# ax.set_xlabel('Position ID', fontsize=20)
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
# plt.savefig("4_3_state3.pdf")

plt.show()
