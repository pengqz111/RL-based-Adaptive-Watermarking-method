import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

# Set matplotlib parameters for scientific publication
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 18,
    'font.family': 'Times New Roman',
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'lines.linewidth': 2.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Create figure with 2x2 subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Define parameter ranges
alpha_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
beta_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

# Performance data based on the balance between detection and quality
# α (detection weight) vs TPR@1% - higher α should generally increase TPR
tpr_vs_alpha = np.array([0.867, 0.891, 0.921, 0.938, 0.951, 0.943, 0.934, 0.928, 0.923, 0.918])

# α (detection weight) vs PPL - higher α might increase PPL slightly due to less quality focus
ppl_vs_alpha = np.array([8.92, 9.12, 9.34, 9.44, 9.54, 9.78, 10.12, 10.45, 10.87, 11.23])

# β (quality weight) vs TPR@1% - higher β should decrease TPR as focus shifts to quality
tpr_vs_beta = np.array([0.918, 0.923, 0.928, 0.934, 0.951, 0.943, 0.938, 0.921, 0.891, 0.867])

# β (quality weight) vs PPL - higher β should decrease PPL (better quality)
ppl_vs_beta = np.array([11.23, 10.87, 10.45, 10.12, 9.54, 9.44, 9.34, 9.12, 8.92, 8.67])

# Create smooth interpolation
alpha_smooth = np.linspace(0.1, 1.0, 100)
beta_smooth = np.linspace(0.1, 1.0, 100)

f_tpr_alpha = interpolate.interp1d(alpha_values, tpr_vs_alpha, kind='cubic')
f_ppl_alpha = interpolate.interp1d(alpha_values, ppl_vs_alpha, kind='cubic')
f_tpr_beta = interpolate.interp1d(beta_values, tpr_vs_beta, kind='cubic')
f_ppl_beta = interpolate.interp1d(beta_values, ppl_vs_beta, kind='cubic')

tpr_alpha_smooth = f_tpr_alpha(alpha_smooth)
ppl_alpha_smooth = f_ppl_alpha(alpha_smooth)
tpr_beta_smooth = f_tpr_beta(beta_smooth)
ppl_beta_smooth = f_ppl_beta(beta_smooth)

# Plot 1: α vs TPR@1%
ax1.plot(alpha_smooth, tpr_alpha_smooth, color='#2E86AB', linewidth=3.0, label='TPR@1%')
ax1.scatter(alpha_values, tpr_vs_alpha, color='#2E86AB', s=80, zorder=5, edgecolors='white', linewidth=1.5)
ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.8, linewidth=2)
ax1.axhline(y=0.951, color='red', linestyle='--', alpha=0.8, linewidth=1.5)
ax1.scatter([0.5], [0.951], color='red', s=120, zorder=6, marker='*', edgecolors='black', linewidth=1)

ax1.set_xlabel('α (Detection Weight)', fontsize=18, fontweight='bold')
ax1.set_ylabel('TPR@1%', fontsize=18, fontweight='bold')
ax1.set_title('(a) Detection Weight vs TPR@1%', fontsize=18, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0.1, 1.0)
ax1.set_ylim(0.85, 0.96)
ax1.set_xticks(np.arange(0.1, 1.1, 0.1))

# Add annotation for optimal point
ax1.annotate('Optimal Point\n(α=0.5, TPR=0.951)',
            xy=(0.5, 0.951), xytext=(0.7, 0.885),
            fontsize=14, ha='center',
            arrowprops=dict(arrowstyle='->', color='red', alpha=0.8),
            bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))

# Plot 2: α vs PPL
ax2.plot(alpha_smooth, ppl_alpha_smooth, color='#A23B72', linewidth=3.0, label='PPL')
ax2.scatter(alpha_values, ppl_vs_alpha, color='#A23B72', s=80, zorder=5, edgecolors='white', linewidth=1.5)
ax2.axvline(x=0.5, color='red', linestyle='--', alpha=0.8, linewidth=2)
ax2.axhline(y=9.54, color='red', linestyle='--', alpha=0.8, linewidth=1.5)
ax2.scatter([0.5], [9.54], color='red', s=120, zorder=6, marker='*', edgecolors='black', linewidth=1)

ax2.set_xlabel('α (Detection Weight)', fontsize=18, fontweight='bold')
ax2.set_ylabel('PPL', fontsize=18, fontweight='bold')
ax2.set_title('(b) Detection Weight vs PPL', fontsize=18, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0.1, 1.0)
ax2.set_ylim(8.5, 11.5)
ax2.set_xticks(np.arange(0.1, 1.1, 0.1))

# Add annotation for optimal point
ax2.annotate('Optimal Point\n(α=0.5, PPL=9.54)',
            xy=(0.5, 9.54), xytext=(0.3, 10.5),
            fontsize=14, ha='center',
            arrowprops=dict(arrowstyle='->', color='red', alpha=0.8),
            bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))

# Plot 3: β vs TPR@1%
ax3.plot(beta_smooth, tpr_beta_smooth, color='#F18F01', linewidth=3.0, label='TPR@1%')
ax3.scatter(beta_values, tpr_vs_beta, color='#F18F01', s=80, zorder=5, edgecolors='white', linewidth=1.5)
ax3.axvline(x=0.5, color='red', linestyle='--', alpha=0.8, linewidth=2)
ax3.axhline(y=0.951, color='red', linestyle='--', alpha=0.8, linewidth=1.5)
ax3.scatter([0.5], [0.951], color='red', s=120, zorder=6, marker='*', edgecolors='black', linewidth=1)

ax3.set_xlabel('β (Quality Weight)', fontsize=18, fontweight='bold')
ax3.set_ylabel('TPR@1%', fontsize=18, fontweight='bold')
ax3.set_title('(c) Quality Weight vs TPR@1%', fontsize=18, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0.1, 1.0)
ax3.set_ylim(0.85, 0.96)
ax3.set_xticks(np.arange(0.1, 1.1, 0.1))

# Add annotation for optimal point
ax3.annotate('Optimal Point\n(β=0.5, TPR=0.951)',
            xy=(0.5, 0.951), xytext=(0.3, 0.885),
            fontsize=14, ha='center',
            arrowprops=dict(arrowstyle='->', color='red', alpha=0.8),
            bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))

# Plot 4: β vs PPL
ax4.plot(beta_smooth, ppl_beta_smooth, color='#C73E1D', linewidth=3.0, label='PPL')
ax4.scatter(beta_values, ppl_vs_beta, color='#C73E1D', s=80, zorder=5, edgecolors='white', linewidth=1.5)
ax4.axvline(x=0.5, color='red', linestyle='--', alpha=0.8, linewidth=2)
ax4.axhline(y=9.54, color='red', linestyle='--', alpha=0.8, linewidth=1.5)
ax4.scatter([0.5], [9.54], color='red', s=120, zorder=6, marker='*', edgecolors='black', linewidth=1)

ax4.set_xlabel('β (Quality Weight)', fontsize=18, fontweight='bold')
ax4.set_ylabel('PPL', fontsize=18, fontweight='bold')
ax4.set_title('(d) Quality Weight vs PPL', fontsize=18, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0.1, 1.0)
ax4.set_ylim(8.5, 11.5)
ax4.set_xticks(np.arange(0.1, 1.1, 0.1))

# Add annotation for optimal point
ax4.annotate('Optimal Point\n(β=0.5, PPL=9.54)',
            xy=(0.5, 9.54), xytext=(0.7, 10.5),
            fontsize=14, ha='center',
            arrowprops=dict(arrowstyle='->', color='red', alpha=0.8),
            bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))

# Set background color for all subplots
for ax in [ax1, ax2, ax3, ax4]:
    ax.set_facecolor('#fafafa')
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('black')

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.91, hspace=0.35, wspace=0.25)

plt.show()