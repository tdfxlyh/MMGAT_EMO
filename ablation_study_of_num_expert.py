import matplotlib.pyplot as plt

# 专家数
experts = [1, 2, 3, 4]

# F1分数
f1_scores_iemocap = [ 72.32,71.54, 72.99, 71.48]  # IEMOCAP数据集的F1分数
f1_scores_meld = [67.14, 67.59, 68.66, 67.15]     # MELD数据集的F1分数

# 找到最高点的索引
max_idx_iemocap = f1_scores_iemocap.index(max(f1_scores_iemocap))
max_idx_meld = f1_scores_meld.index(max(f1_scores_meld))

# 创建折线图
plt.figure(figsize=(10, 6))
plt.plot(experts, f1_scores_iemocap, marker='o', color='blue', label='IEMOCAP')
plt.plot(experts, f1_scores_meld, marker='o', color='green', label='MELD')


# 标注IEMOCAP的最高点
plt.text(experts[max_idx_iemocap], f1_scores_iemocap[max_idx_iemocap]+ 0.05, 
         f'{f1_scores_iemocap[max_idx_iemocap]:.2f}', 
         ha='center', va='bottom', fontsize=10, color='blue')

# 标注MELD的最高点
plt.text(experts[max_idx_meld], f1_scores_meld[max_idx_meld]+ 0.05, 
         f'{f1_scores_meld[max_idx_meld]:.2f}', 
         ha='center', va='bottom', fontsize=10, color='green')


plt.title('Ablation Study on Number of Experts')
plt.xlabel('Number of Experts')
plt.ylabel('F1 Score')
plt.xticks(experts)  # 设置x轴的刻度为专家数
plt.ylim(63, 76)   # 根据实际数据设置y轴范围

# 显示图例
plt.legend()

# 显示图形
plt.grid(True)
plt.show()