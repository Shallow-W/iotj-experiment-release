import random
import copy

def main(node_position, ms_resource_info, whole_traffic_weight, node_resource, seed):
    random.seed(seed)
    # 初始化全0的列表
    whole_state = [[0 for _ in range(len(node_position))] for _ in range(len(ms_resource_info))]

    # 获取行和列的索引列表
    rows = list(range(len(ms_resource_info)))
    cols = list(range(len(node_position)))

    # 随机选择10个不同的行和列
    selected_rows = random.sample(rows, len(ms_resource_info))
    selected_cols = random.sample(cols, len(ms_resource_info))

    # 将选中的位置置为1
    for row, col in zip(selected_rows, selected_cols):
        whole_state[row][col] = 1
    temp = copy.deepcopy(whole_traffic_weight)
    for row in temp:
        whole_state.append(row)  # 现在是20*50的矩阵，前10行是微服务的实例数量，后10行是整个网络的流量权重
    resource_state = [[node_resource[j][i] for j in range(len(node_resource))] for i in range(len(node_resource[0]))]
    whole_state.extend(resource_state)  # 现在是24*50的矩阵，前10行是微服务的实例数量，后4行是节点的现存资源数量
    return whole_state
