import matplotlib.pyplot as plt
import numpy as np

# 设置matplotlib参数以符合科研论文规范
plt.rcParams.update({
    'font.size': 16,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.linewidth': 1.2,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.figsize': (10, 7),
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'grid.alpha': 0.3,
    'axes.grid': True,
    'grid.linewidth': 0.5
})

# 数据
gamma_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# 各方法的TPR@1%数据
rlawm_data = [0.723, 0.834, 0.891, 0.923, 0.951, 0.939, 0.912, 0.876, 0.834]
morphmark_data = [0.678, 0.756, 0.823, 0.867, 0.892, 0.876, 0.845, 0.812, 0.789]
crmark_data = [0.634, 0.723, 0.789, 0.834, 0.856, 0.843, 0.821, 0.798, 0.767]
kgw_data = [0.567, 0.654, 0.723, 0.789, 0.812, 0.798, 0.776, 0.743, 0.712]

# 创建图形
fig, ax = plt.subplots(figsize=(10, 7))

# 绘制曲线
ax.plot(gamma_values, rlawm_data, 'o-', color='#E74C3C', linewidth=3,
        markersize=10, label='RLAWM', markerfacecolor='#E74C3C',
        markeredgecolor='white', markeredgewidth=2)

ax.plot(gamma_values, morphmark_data, 's--', color='#3498DB', linewidth=2.5,
        markersize=9, label='MorphMark', markerfacecolor='#3498DB',
        markeredgecolor='white', markeredgewidth=1.5)

ax.plot(gamma_values, crmark_data, '^-.', color='#F39C12', linewidth=2.5,
        markersize=9, label='CRMark', markerfacecolor='#F39C12',
        markeredgecolor='white', markeredgewidth=1.5)

ax.plot(gamma_values, kgw_data, 'd:', color='#9B59B6', linewidth=2.5,
        markersize=9, label='KGW', markerfacecolor='#9B59B6',
        markeredgecolor='white', markeredgewidth=1.5)

# 设置坐标轴
ax.set_xlabel('Green Vocabulary Ratio (γ)', fontsize=18, fontweight='bold')
ax.set_ylabel('TPR@1%', fontsize=18, fontweight='bold')

# 设置坐标轴范围和刻度
ax.set_xlim(0.05, 0.95)
ax.set_ylim(0.50, 1.00)
ax.set_xticks(gamma_values)
ax.set_yticks(np.arange(0.5, 1.05, 0.1))

# 添加网格
ax.grid(True, linestyle='-', alpha=0.2, linewidth=0.5)
ax.set_axisbelow(True)

# 设置图例
legend = ax.legend(loc='lower right', frameon=True, fancybox=True,
                  shadow=True, ncol=1, fontsize=16)
legend.get_frame().set_facecolor('#FFFFFF')
legend.get_frame().set_alpha(0.9)
legend.get_frame().set_edgecolor('#CCCCCC')
legend.get_frame().set_linewidth(1)

# 突出显示RLAWM的最佳性能点
best_idx = rlawm_data.index(max(rlawm_data))
ax.annotate(f'Peak: {max(rlawm_data):.3f}',
            xy=(gamma_values[best_idx], max(rlawm_data)),
            xytext=(gamma_values[best_idx] + 0.05, max(rlawm_data) - 0.05),
            fontsize=14, fontweight='bold', color='#E74C3C',
            arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=1.5),
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                     edgecolor='#E74C3C', alpha=0.8))

# 设置边框
for spine in ax.spines.values():
    spine.set_linewidth(1.2)
    spine.set_color('#333333')

# 调整布局
plt.tight_layout()

# 显示图片
plt.show()