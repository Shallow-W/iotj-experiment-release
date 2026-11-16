"""Historical ablation plotting script with the original labels and transforms."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import smooth_data
# 设置字体为 SimHei
plt.rcParams['font.sans-serif'] = ['SongNTR'] ##设置字体
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False
# color_list1 = ['#C82221', '#000000']   # 红蓝灰黑
# color_list1 = ['#C82221', '#000000']   # 红蓝灰黑
# color_list2 = ['#C82221', '#000000']   # 四色
color_list1 = ['#E1D5EB', '#FBE5D7']   # 红蓝灰黑
color_list2 = ['#E1D5EB', '#FBE5D7']   # 四色
# 读取Excel文件（修改文件路径）
ROOT = Path(__file__).resolve().parents[2]
file_path = ROOT / "results" / "training"
reward_data_list1 = ["rewards_data_d3qn_mask_pri","rewards_data_d3qn_mask_nopri"]
reward_data_list2 = ["rewards_data_d3qn_mask_pri","rewards_data_d3qn_nomask_pri"]
delay_data_list1 = ["delays_data_d3qn_mask_pri","delays_data_d3qn_nomask_pri"]
delay_data_list2 = ["delays_data_d3qn_mask_pri","delays_data_d3qn_mask_nopri"]
name_list = ['rewards', 'delays']
label_list1 = ['With action masks and status masks', 'Without action masks and status masks']
label_list2 = ['With priority experience replays', 'Without priority experience replays']
# 创建曲线图
fig1, ax1 = plt.subplots(figsize=(10, 6))
fig2, ax2 = plt.subplots(figsize=(10, 6))
fig3, ax3 = plt.subplots(figsize=(10, 6))
fig4, ax4 = plt.subplots(figsize=(10, 6))
init_weight, weight_decay = 0.8, 0.4

def delay_smooth(delay_value):
    y_list = []
    x_list = []
    y_value = np.inf
    count = 0
    for i in range(len(delay_value)):
        if delay_value[i] < y_value:
            print('delay_value[i]',delay_value[i])
            y_value = delay_value[i]
            y_list.append(delay_value[i])
            x_list.append(count)
        count += 1

    y_list.append(y_value)
    x_list.append(len(delay_value)-1)
    return y_list, x_list

# 比较有无掩码
for j in range(len(reward_data_list1)):
    file_name1 = f"{file_path}/{reward_data_list1[j]}.xlsx"
    df1 = pd.read_excel(file_name1)

    file_name2 = f"{file_path}/{delay_data_list1[j]}.xlsx"
    df2 = pd.read_excel(file_name2)
    # 自动选择数值型列并计算行和
    numeric_df1 = df1.select_dtypes(include=['number'])
    row_sums1 = numeric_df1.sum(axis=1)
    smoothed_data1 = smooth_data.auto(row_sums1.values, init_weight, weight_decay)
    ax1.plot(smoothed_data1,
             color=color_list1[j],
             linestyle='-',
             linewidth=1.5,
             markersize=6, label=label_list1[j], zorder=len(reward_data_list1) - j)

    numeric_df2 = df2.select_dtypes(include=['number'])
    row_sums2 = numeric_df2.sum(axis=1)
    y_list, x_list = delay_smooth(row_sums2.values)
    ax2.plot(x_list, y_list,
             color=color_list2[j],
             linestyle='--',
             linewidth=2,
             marker='s',
             markersize=5, label=label_list1[j], zorder=len(reward_data_list1) - j)
# 添加图表元素
ax1.set_title("")
ax1.set_xlabel('Number of iterations', fontsize=22)
ax1.set_ylabel("Reward", fontsize=22)
ax1.legend(fontsize=22)  # 显示图例
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.tick_params(axis='x', labelsize=20)
ax1.tick_params(axis='y', labelsize=20)
ax2.set_title("")
ax2.set_xlabel('Number of iterations', fontsize=22)
ax2.set_ylabel("Total delay (ms)", fontsize=22)
ax2.legend(fontsize=22)  # 显示图例
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.set_yticks([10000, 12500, 15000, 17500, 20000, 22500, 25000, 27500, 30000])  # 设置y轴的刻度位置
ax2.set_yticklabels([1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000])  # 替换y轴的刻度标签
ax2.tick_params(axis='x', labelsize=20)
ax2.tick_params(axis='y', labelsize=20)
# 比较有无优先级经验回放
for j in range(len(reward_data_list2)):
    file_name1 = f"{file_path}/{reward_data_list2[j]}.xlsx"
    df1 = pd.read_excel(file_name1)

    file_name2 = f"{file_path}/{delay_data_list2[j]}.xlsx"
    df2 = pd.read_excel(file_name2)
    # 自动选择数值型列并计算行和
    numeric_df1 = df1.select_dtypes(include=['number'])
    row_sums1 = numeric_df1.sum(axis=1)
    smoothed_data1 = smooth_data.auto(row_sums1.values, init_weight, weight_decay)
    ax3.plot(smoothed_data1,
             color=color_list1[j],
             linestyle='-',
             linewidth=1.5,
             markersize=6, label=label_list2[j], zorder=len(reward_data_list1) - j)

    numeric_df2 = df2.select_dtypes(include=['number'])
    row_sums2 = numeric_df2.sum(axis=1)
    # print('row_sums2', len(row_sums2))
    y_list, x_list = delay_smooth(row_sums2.values)
    ax4.plot(x_list, y_list,
             color=color_list2[j],
             linestyle='--',
             linewidth=2,
             marker='s',
             markersize=5, label=label_list2[j], zorder=len(reward_data_list1) - j)
# 添加图表元素
ax3.set_title("")
ax3.set_xlabel('Number of iterations', fontsize=22)
ax3.set_ylabel("Reward", fontsize=22)
ax3.tick_params(axis='x', labelsize=20)
ax3.tick_params(axis='y', labelsize=20)
ax3.legend(fontsize=22)  # 显示图例
ax3.grid(True, linestyle='--', alpha=0.7)
ax4.set_title("")
ax4.set_xlabel('Number of iterations', fontsize=22)
ax4.set_ylabel("Total delay (ms)", fontsize=22)
ax4.legend(fontsize=22)  # 显示图例
ax4.grid(True, linestyle='--', alpha=0.7)
ax4.set_yticks([10000, 12500, 15000, 17500, 20000, 22500, 25000, 27500, 30000])  # 设置y轴的刻度位置
ax4.set_yticklabels([1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000])  # 替换y轴的刻度标签
ax4.tick_params(axis='x', labelsize=20)
ax4.tick_params(axis='y', labelsize=20)
output = ROOT / "reproduced" / "archived_plots"
output.mkdir(parents=True, exist_ok=True)
for name, figure in zip(
    ("ablation_reward_1", "ablation_delay_1", "ablation_reward_2", "ablation_delay_2"),
    (fig1, fig2, fig3, fig4),
):
    figure.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
