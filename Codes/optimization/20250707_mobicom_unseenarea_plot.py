import matplotlib.pyplot as plt
import numpy as np

# Data for the three unseen regions



#state2
# regions = ['1', '2', '3','4','5','6']
# regions = ['Motion 1', 'Motion 2']
regions = ['Pos 1', 'Pos 2']

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
plt.savefig("4_3_state3.pdf")

plt.show()
