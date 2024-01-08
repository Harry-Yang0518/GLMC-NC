import os
import sys
import pickle
import numpy as np
from matplotlib import pyplot as plt

root = f"/scratch/hy2611/GLMC-NC/output/cifar10_resnet32/{sys.argv[1]}"
epoch = sys.argv[2]
file = os.path.join(root, 'analysis{}.pkl'.format(epoch))

with open(file, 'rb') as f:
    nc_dt = pickle.load(f)

# Define keys to plot
k_lst = [k for k in nc_dt if k.endswith('_cos') or k.endswith('_norm')]

# Create a figure with a specified size for better visibility
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 8))

# Iterate over keys and plot
for k, key in enumerate(k_lst):
    row, col = divmod(k, 2)
    ax = axes[row, col]
    data = nc_dt[key]

    if key in ['w_cos', 'h_cos']:
        im = ax.imshow(data, cmap='RdBu', vmin=-0.9, vmax=0.9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)  # Adjust colorbar size and padding
        ax.set_title(key, fontsize=10)
    else:
        ax.bar(np.arange(len(data)), data, color='skyblue')  # Uniform color for bars
        ax.set_title(key, fontsize=10)

    # Set axis labels as needed
    ax.set_xlabel('X-axis Label', fontsize=8)
    ax.set_ylabel('Y-axis Label', fontsize=8)

# Adjust layout and save the figure
plt.tight_layout()
fig.savefig(os.path.join(root, f'cos{epoch}.png'))
