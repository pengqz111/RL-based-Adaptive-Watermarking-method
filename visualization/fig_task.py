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
    'xtick.labelsize': 14,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.figsize': (16, 6),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5
})

# 生成数据函数
def generate_violin_data(mean, std, size=100, distribution='beta'):
    if distribution == 'normal':
        return np.random.normal(mean, std, size)
    elif distribution == 'beta':
        a = mean * (mean * (1 - mean) / (std ** 2) - 1)
        b = (1 - mean) * (mean * (1 - mean) / (std ** 2) - 1)
        return np.random.beta(max(0.1, a), max(0.1, b), size)

# 方法和颜色
methods = ['KGW', 'REMARK', 'SemStamp', 'WatME', 'MorphMark', 'RLMark', 'CRMark', 'RLAWM']
colors = ['#E74C3C', '#3498DB', '#F39C12', '#9B59B6', '#2ECC71', '#E67E22', '#34495E', '#27AE60']

# BERTScore均值和标准差
mt_bert_means = [0.612, 0.643, 0.628, 0.635, 0.671, 0.639, 0.658, 0.724]
mt_bert_stds = [0.039, 0.038, 0.042, 0.041, 0.035, 0.043, 0.039, 0.028]
ts_bert_means = [0.617, 0.624, 0.608, 0.615, 0.642, 0.609, 0.628, 0.691]
ts_bert_stds = [0.042, 0.030, 0.031, 0.043, 0.035, 0.033, 0.047, 0.036]

# 模拟数据
mt_bert_data = [generate_violin_data(m, s, 120) for m, s in zip(mt_bert_means, mt_bert_stds)]
ts_bert_data = [generate_violin_data(m, s, 120) for m, s in zip(ts_bert_means, ts_bert_stds)]

# 创建图形和子图
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(wspace=0.25)
titles = []

for idx, (ax, data, means, task) in enumerate(zip(
        axes, [mt_bert_data, ts_bert_data], [mt_bert_means, ts_bert_means],
        ['Machine Translation', 'Text Summarization'])):

    parts = ax.violinplot(data, positions=range(len(methods)),
                          widths=0.6, showmeans=True, showmedians=True)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(1)

    parts['bodies'][-1].set_alpha(0.9)
    parts['bodies'][-1].set_edgecolor('#1B4F72')
    parts['bodies'][-1].set_linewidth(2)

    parts['cmeans'].set_color('red')
    parts['cmeans'].set_linewidth(2)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(1.5)

    ax.set_ylabel('BERTScore', fontsize=18, fontweight='bold')
    ax.set_ylim(0.5, 0.8)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=15, ha='right', fontsize=14)
    ax.grid(True, alpha=0.3)

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('#333333')

    rlawm_score = means[-1]
    second_best = sorted(means[:-1])[-1]
    improvement = ((rlawm_score - second_best) / second_best) * 100
    ax.annotate(f'+{improvement:.1f}%',
                xy=(7, rlawm_score),
                xytext=(6.2, rlawm_score + 0.03),
                fontsize=12, fontweight='bold', color='#1B4F72',
                arrowprops=dict(arrowstyle='->', color='#1B4F72', lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", facecolor='#E8F6F3',
                          edgecolor='#1B4F72', alpha=0.8))

    # 保存标题位置
    titles.append((ax.get_position().x0 + ax.get_position().width / 2,
                   ax.get_position().y0))

# 图例放在左上角（相对于整个 figure）
legend_elements = [
    plt.Line2D([0], [0], color='red', lw=2, label='Mean'),
    plt.Line2D([0], [0], color='black', lw=1.5, label='Median'),
    plt.Rectangle((0, 0), 1, 1, facecolor='#27AE60', alpha=0.9,
                  edgecolor='#1B4F72', linewidth=2, label='RLAWM')
]
fig.legend(handles=legend_elements, loc='upper left',
           bbox_to_anchor=(0.07, 0.95), ncol=3,
           frameon=True, fancybox=True, shadow=True, fontsize=14)

# 子图标题放在下方
for (x, y), label in zip(titles, ['(a) Machine Translation', '(b) Text Summarization']):
    fig.text(x, y - 0.07, label, ha='center', va='top', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.show()
