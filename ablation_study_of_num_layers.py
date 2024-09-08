import matplotlib.pyplot as plt
import seaborn as sns

# 设置Seaborn主题样式
sns.set(style="whitegrid")

# MMGAT层数
layers = [1, 2, 3, 4]

# F1分数
f1_scores_iemocap = [72.6, 72.99, 70.96, 66.63]  # IEMOCAP数据集的F1分数
f1_scores_meld = [67.38, 68.66, 67.57, 67.45]     # MELD数据集的F1分数

# 设置柱状图的宽度
bar_width = 0.4

# 计算每个柱的x位置
index_iemocap = [x - bar_width/2 for x in range(len(layers))]
index_meld = [x + bar_width/2 for x in range(len(layers))]

# 创建柱状图
plt.figure(figsize=(10, 6))
bars_iemocap = plt.bar(index_iemocap, f1_scores_iemocap, width=bar_width, color='#ADD8E6', label='IEMOCAP')
bars_meld = plt.bar(index_meld, f1_scores_meld, width=bar_width, color='#98FB98', label='MELD')

# 在每个柱状图上方标注具体数值
for bar in bars_iemocap:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval:.2f}', ha='center', va='bottom', fontsize=10)

for bar in bars_meld:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval:.2f}', ha='center', va='bottom', fontsize=10)

# 添加标题和标签
plt.title('Ablation Study on Number of MMGAT Layers')
plt.xlabel('Number of Layers')
plt.ylabel('F1 Score')
plt.xticks(range(len(layers)), layers)  # 设置x轴的刻度为层数
plt.ylim(65, 75)   # 根据实际数据设置y轴范围

# 显示图例
plt.legend()

# 添加网格线
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 显示图形
plt.tight_layout()
plt.show()