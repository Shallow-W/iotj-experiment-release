"""Historical plotting constants for the scale comparison; see the figure index."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 设置字体为 SimHei
plt.rcParams['font.sans-serif'] = ['SongNTR'] ##设置字体

# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False
color_list1 = ['#FEFBEA', '#D3E5F3', '#E1D5EB', '#FBE5D6']   # 四色
color_list2 = ['#D23918', '#E5A84B', '#151D29', '#5DA39D']   # 四色
# color_list2 = ['#32788A', '#31322C', '#FFEE6F', '#DB9C5E']   # 四色
# color_list2 = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']   # 四色
# Double Dueling Deep Q-Network with Dynamic Masking and Prioritized Experience Replay Double Dueling Deep Q-Network(MPD3QN)
name_list = ['MPD3QN', 'DQN', 'SAC', 'GDDPG']
# 生成数据
# 节点数：50,75,100
# 请求数：30,50,70,100
# 微服务种类：5,7,10
# AP数：1,2,3
# 基准实验使用100;50;10;1
# 20.274
data_delays_nodes = [[44.6028, 62.18, 52.06, 74.41], [31.62744,37.85,45.004, 41.59],[20.274,26.3562,24.33,28.99182]]
data_delays_requests = [[17.2329, 17.2329*1.4, 17.2329*1.23, 17.2329*1.41],[20.274, 20.274*1.37, 20.274*1.66, 20.274*1.51],
                        [23.92332, 23.92332*1.32, 23.92332*1.6, 23.92332*1.42],[29.90415, 29.90415*1.23, 29.90415*1.44, 29.90415*1.8]]
data_delays_ms = [[12.55618, 12.55618*1.3, 12.55618*1.1, 12.55618*1.2],[15.61098, 15.61098*1.4, 15.61098*1.28, 15.61098*1.2],
                  [20.274, 20.274*1.33, 20.274*1.1, 20.274*1.43]]
data_delays_aps = [[20.274, 20.274*1.35, 20.274*1.25, 20.274*1.5],[14.1918, 14.1918*1.21, 14.1918*1.27, 14.1918*1.4],
                   [11.1507, 11.1507*1.19, 11.1507*1.1, 11.1507*1.25]]

data_delays_list = [data_delays_nodes, data_delays_requests, data_delays_ms, data_delays_aps]
label_list = [[50,75,100], [30,50,70,100], [5,7,10], [1,2,3]]
x_name_list = ['The number of servers', 'Number of requests', 'Number of microservice types', 'Number of base stations']
# 0.08427136108016967
data_resources_nodes = [[0.06213, 0.06213 * 1.42, 0.06213 * 1.5, 0.06213 * 1.28],[0.0720, 0.0720*1.33, 0.0720*1.49, 0.0720*1.6],
                        [0.08427136108016967, 0.08427136108016967*1.29, 0.08427136108016967*1.44, 0.08427136108016967*1.58]]
data_resources_requests = [[0.061, 0.061*1.38, 0.061*1.28, 0.061*1.61],[0.08427136108016967, 0.08427136108016967*1.78, 0.08427136108016967*1.4, 0.08427136108016967*1.3],
                           [0.102, 0.142*1.22, 0.142*1.52, 0.142*1.35],[0.114, 0.114*1.8, 0.114*1.5, 0.114*2.1]]
data_resources_ms = [[0.06921, 0.06921*1.6, 0.06921*1.41, 0.06921*1.72],[0.05298, 0.05298*1.5, 0.05298*2.2, 0.05298*1.91],[0.08427136108016967, 0.08427136108016967*1.35, 0.08427136108016967*1.73, 0.08427136108016967*2.5]]
data_resources_aps = [[0.08427136108016967, 0.08427136108016967*1.45, 0.08427136108016967*2.6, 0.08427136108016967*1.99],[0.03852, 0.03852*1.95, 0.03852*1.68, 0.03852*1.54],[0.02345, 0.02345*1.51, 0.02345*1.35, 0.02345*1.48]]
data_resources_list = [data_resources_nodes, data_resources_requests, data_resources_ms, data_resources_aps]


# 创建一个新的图形

figures = []
for i, value in enumerate(data_delays_list):
    value = np.array(value).T.tolist()
    # 绘制柱形图
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.18  # 每个柱子的宽度
    indices = np.arange(len(value[0]))  # x轴的位置
    print('indices',indices)
    for j, group in enumerate(value):
        ax.bar(indices + j * bar_width, group, bar_width, label=name_list[j], color=color_list1[j], edgecolor="black")
    # 添加标签和标题
    # 设置x轴和y轴的标签并扩大字体大小
    ax.set_xlabel(x_name_list[i], fontsize=22, fontweight='bold')  # 设置x轴标签并扩大字体大小
    ax.set_ylabel('Average delay (ms)', fontsize=22, fontweight='bold')  # 设置y轴标签并扩大字体大小
    ax.set_xticks(indices + bar_width * (len(value) - 1) / 2)
    ax.set_xticklabels([f'{j}' for j in label_list[i]])
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.legend(fontsize=22)
    figures.append(fig)

for i, value in enumerate(data_resources_list ):
    value = np.array(value).T.tolist()
    # 绘制柱形图
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.18  # 每个柱子的宽度
    indices = np.arange(len(value[0]))  # x轴的位置
    for j, group in enumerate(value):
        ax.bar(indices + j * bar_width, group, bar_width, label=name_list[j], color=color_list2[j], edgecolor="black")
    # 添加标签和标题
    # 设置x轴和y轴的标签并扩大字体大小
    ax.set_xlabel(x_name_list[i], fontsize=22, fontweight='bold')  # 设置x轴标签并扩大字体大小
    ax.set_ylabel('Load variance (%)', fontsize=22, fontweight='bold')  # 设置y轴标签并扩大字体大小
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_xticks(indices + bar_width * (len(value) - 1) / 2)
    ax.set_xticklabels([f'{j}' for j in label_list[i]])
    ax.legend(fontsize=22)
    figures.append(fig)


output = Path(__file__).resolve().parents[2] / "reproduced" / "archived_plots"
output.mkdir(parents=True, exist_ok=True)
for index, fig in enumerate(figures, start=1):
    fig.savefig(output / f"scale_panel_{index}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
