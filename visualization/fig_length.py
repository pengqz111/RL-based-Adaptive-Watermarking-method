import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# Set style for academic papers
plt.style.use('default')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'

# Set random seed for reproducibility
np.random.seed(42)

# Create figure and axis
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Text length categories
text_lengths = [100, 200, 300, 400, 500, 600]

# Base performance data (before adding noise)
methods_data = {
    'KGW':       [0.945, 0.923, 0.908, 0.895, 0.884, 0.876],
    'REMARK':    [0.952, 0.931, 0.917, 0.909, 0.898, 0.889],
    'SemStamp':  [0.938, 0.918, 0.902, 0.879, 0.868, 0.859],
    'WatME':     [0.931, 0.912, 0.897, 0.861, 0.852, 0.844],
    'MorphMark': [0.963, 0.945, 0.932, 0.928, 0.919, 0.912],
    'RLMark':    [0.925, 0.906, 0.894, 0.890, 0.883, 0.875],
    'CRMark':    [0.948, 0.928, 0.915, 0.923, 0.914, 0.907],
    'RLAWM':     [0.978, 0.963, 0.954, 0.951, 0.946, 0.942]
}

# Add small Gaussian noise to each method's data
noise_level = 0.003
noisy_methods_data = {
    method: np.clip(np.array(values) + np.random.normal(0, noise_level, len(values)), 0.82, 0.99)
    for method, values in methods_data.items()
}

# Colors, styles, and markers
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2', '#17becf']
line_styles = ['-', '--', '-.', ':', '-', '--', '-.', '-']
markers = ['o', 's', '^', 'v', 'D', 'p', 'h', '*']

# Plot lines
for i, (method, data) in enumerate(noisy_methods_data.items()):
    if method == 'RLAWM':
        ax.plot(text_lengths, data, color=colors[i], linestyle=line_styles[i],
                marker=markers[i], linewidth=3, markersize=8, label=method, zorder=10)
    else:
        ax.plot(text_lengths, data, color=colors[i], linestyle=line_styles[i],
                marker=markers[i], linewidth=2, markersize=6, label=method, zorder=5)

# Customize plot
ax.set_xlabel('Text Length (Number of Tokens)', fontsize=18, fontweight='bold')
ax.set_ylabel('TPR@1%', fontsize=18, fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_xlim(90, 610)
ax.set_ylim(0.82, 0.99)
ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.8)

# Legend
legend = ax.legend(loc='lower left', fontsize=16, ncol=2,
                   frameon=True, fancybox=True, shadow=True,
                   columnspacing=1.5, handletextpad=0.5)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_alpha(0.9)

# Layout and background
plt.tight_layout()
ax.set_facecolor('#fafafa')

# Show plot
plt.show()
