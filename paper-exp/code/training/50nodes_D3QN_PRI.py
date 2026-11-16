import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import openpyxl
import copy
from experiment_paths import episode_count, input_prefix, output_directory
import matplotlib.pyplot as plt
import os
from openpyxl import Workbook
from datetime import datetime
# 让用户输入文件名和保存路径
original_name = "rewards_data_d3qn_50nodes"
save_directory = output_directory(__file__)
# 自动生成带时间戳的文件名
timestamp = datetime.now().strftime("%Y%m%d_%H%M")  # 格式示例：20231015_1645
file_name = f"{original_name}_{timestamp}.xlsx"  # 强制使用.xlsx扩展名

# 创建Excel工作簿和工作表
wb = Workbook()
ws = wb.active

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

ap_position_traffic = [[250, 250, 14]]

# 文件路径前缀
file_prefix = input_prefix("50endpoints_nms")

node_info_workbook = openpyxl.load_workbook(file_prefix + 'node_resource.xlsx', data_only=True)
node_info_sheet = node_info_workbook['Sheet1']
node_position = []
start_row = 2
for i in range(2, 52):
    x_value = node_info_sheet.cell(row=i, column=6).value
    y_value = node_info_sheet.cell(row=i, column=7).value
    node_position.append([x_value, y_value])

node_resource = []
for i in range(2, 52):
    cpu_value = node_info_sheet.cell(row=i, column=2).value
    mem_value = node_info_sheet.cell(row=i, column=3).value
    disk_value = node_info_sheet.cell(row=i, column=4).value
    gpu_value = node_info_sheet.cell(row=i, column=5).value
    node_resource.append([cpu_value, mem_value, disk_value, gpu_value])

app_ms_workbook = openpyxl.load_workbook(file_prefix + 'app.xlsx', data_only=True)
app_ms_sheet = app_ms_workbook['Sheet1']

app_ms_info = []
for i in range(2, 8):
    temp_ms = []
    for j in range(2, 6):
        cell_value = app_ms_sheet.cell(row=i, column=j).value
        if cell_value is not None:
            mapped_value = int(cell_value[1:]) if cell_value.startswith('m') else cell_value
            temp_ms.append(mapped_value)
    if temp_ms:
        app_ms_info.append(temp_ms)
request_accept_info = []  # 每个服务从父节点接收流量的数量，一维的维度是10，就是十种微服务，二维的维度是11，也就是AP加十种微服务。第二维的意义就是父节点了。
for i in range(41, 51):
    temp_request_accept = []
    for j in range(2, 13):
        cell_value = app_ms_sheet.cell(row=i, column=j).value
        if cell_value is not None:
            temp_request_accept.append(cell_value)
    if temp_request_accept:
        request_accept_info.append(temp_request_accept)
traffic_deliver_info = []  # 每个服务下发流量的数量，一维的维度是11，也就是十种微服务加AP，二维的维度是11，也就是十种微服务加AP。第二维的意义就是子节点了。
for i in range(55, 66):
    temp_traffic_deliver = []
    for j in range(2, 13):
        cell_value = app_ms_sheet.cell(row=i, column=j).value
        if cell_value is not None:
            temp_traffic_deliver.append(cell_value)
    if temp_traffic_deliver:
        traffic_deliver_info.append(temp_traffic_deliver)
print(request_accept_info)
ms_accoount_info = []  # 每种微服务最多要部署多少个，也是请求所需要的微服务的最多的数目，侧面反映了微服务的部署压力
for i in range(28, 38):
    ms_accoount_info.append(app_ms_sheet.cell(row=i, column=2).value)  # 每种微服务最多要部署多少个，也是请求所需要的微服务的最多的数目，侧面反映了微服务的部署压力
ap_account_info = app_ms_sheet.cell(row=27, column=2).value  # AP的下发的请求数目

ms_resource_workbook = openpyxl.load_workbook(file_prefix + 'ms_resource.xlsx', data_only=True)
ms_resource_sheet = ms_resource_workbook['Sheet1']
ms_resource_info = []
for i in range(2, 12):
    temp_ms_resource = []
    for j in range(2, 6):
        cell_value = ms_resource_sheet.cell(row=i, column=j).value
        if cell_value is not None:
            temp_ms_resource.append(cell_value)
    if temp_ms_resource:
        ms_resource_info.append(temp_ms_resource)

request_workbook = openpyxl.load_workbook(file_prefix + 'request.xlsx', data_only=True)
request_sheet = request_workbook['Sheet1']
request_account_info = []
for i in range(2, 7):
    request_account_info.append(request_sheet.cell(row=302, column=i).value)

whole_traffic_weight = [[0 for _ in range(len(node_position))] for _ in
                        range(len(ms_resource_info))]  # 整个网络的流量权重，重点突出在哪个节点上的哪种微服务会接收多少流量，如果一个节点上面有多个同类微服务会合并计算
missing_traffic_weight = [ms_accoount_info[i] for i in range(
    len(ms_accoount_info))]  # 整个网络中没有成功分配的流量的情况，具体含义就是这个微服务没有任何的部署，那么原本计划要通过它的流量就全部处于待分配状态，此时计算时延是一个定值
# whole_state = [[0 for _ in range(len(node_position))] for _ in
#                range(len(ms_resource_info))]  # 现在是10*50的矩阵，每一行代表一个微服务，每一列代表一个节点，每个元素代表这个微服务在这个节点上的实例数量
# for row in whole_traffic_weight:
#     whole_state.append(row)  # 现在是20*50的矩阵，前10行是微服务的实例数量，后10行是整个网络的流量权重
resource_state = [[node_resource[j][i] for j in range(len(node_resource))] for i in range(len(node_resource[0]))]
# print('')
# whole_state.extend(resource_state)  # 现在是24*50的矩阵，前10行是微服务的实例数量，后4行是节点的现存资源数量
# whole_state.extend(resource_state) #本来这里要放节点的初始资源数量，做负载均衡的，但是这个版本还不考虑在奖励函数中加入负载均衡的考虑，所以先不加。
ms_queue_num = [[0 for _ in range(len(node_position))] for _ in
                range(len(ms_resource_info))]  # 每个服务器上每种微服务的请求队列的长度，注意如果这个服务器上有多个微服务，需要均分排队长度。
missing_traffic_delay = 1500  # 整个网络中没有成功分配的流量的情况，具体含义就是这个微服务没有任何的部署，那么原本计划要通过它的流量就全部处于待分配状态，此时计算时延是一个定值,现在是400ms

def generate_whole_state(node_position, ms_resource_info, whole_traffic_weight, node_resource):
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
    print('cur whole_state', whole_state)
    return whole_state

class TreeNode:
    def __init__(self, value):
        self.value = value
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)


class Env(object):
    def __init__(self):
        self.action_space = len(node_position)
        self.reward = 0

    def reset(self):
        pass

    # 建立N叉树
    def build_tree(self):
        # 创建节点
        ap = TreeNode('-1')
        m1 = TreeNode('0')
        m2 = TreeNode('1')
        m3 = TreeNode('2')
        m4 = TreeNode('3')
        m5 = TreeNode('4')
        m6 = TreeNode('5')
        m7 = TreeNode('6')
        m8 = TreeNode('7')
        m9 = TreeNode('8')
        m10 = TreeNode('9')

        # 建立父子关系
        ap.add_child(m1)
        ap.add_child(m8)
        m1.add_child(m2)
        m1.add_child(m3)
        m2.add_child(m4)
        m2.add_child(m5)
        m3.add_child(m6)
        m5.add_child(m7)
        m6.add_child(m7)
        m8.add_child(m9)
        m8.add_child(m10)

        return ap

    def traffic_num(self, state):  # 计算整个网络的流量分布,基于现有的状态，全流量下发
        delay_total = 0
        trans_delay_total = 0
        global traffic_deliver_info  # 使用全局变量 traffic_deliver_info
        global ap_position_traffic  # 使用全局变量 ap_position_traffic
        AP_traffic_count = copy.deepcopy(ap_account_info)  # 表示AP下发的请求总数
        queue = [(self.build_tree(), AP_traffic_count)]
        missing_traffic_weight_temp = copy.deepcopy(missing_traffic_weight)
        whole_traffic_weight_temp = copy.deepcopy(whole_traffic_weight)
        while queue:
            node, traffic_count = queue.pop(0)
            for child in node.children:
                child_traffic_num = traffic_deliver_info[int(node.value) + 1][int(child.value)]  # 表示父节点应该下发给子节点的流量数量
                if sum(state[int(child.value)]) == 0:  # 如果这个微服务没有在任何节点上部署，那么这个流量就是待分配的流量，这些流量没有任何地方可以去
                    # trans_delay_total += child_traffic_num * missing_traffic_delay #因为这部分流量无处可去，直接超时了，所以加一个大一点的时延
                    # missing_traffic_weight_temp[int(child.value)] += child_traffic_num
                    continue
                else:
                    if int(node.value) == -1:
                        for i in range(len(ap_position_traffic)):
                            distance = []
                            for j in range(len(state[int(child.value)])):
                                if state[int(child.value)][j] != 0:
                                    distance.append(max(1, ((node_position[j][0] - ap_position_traffic[i][0]) ** 2 + (
                                            node_position[j][1] - ap_position_traffic[i][1]) ** 2) ** 0.5 /
                                                        state[int(child.value)][j]))
                            distance_count = sum(distance)
                            weight = []
                            for j in range(len(distance)):
                                weight.append(1 / distance[j])
                            weight_total = sum(weight)
                            # non_zero_indices = np.where(state[int(child.value)] != 0)[0]
                            index_temp_weight = 0
                            for j in range(len(state[int(child.value)])):
                                if state[int(child.value)][j] == 0:
                                    continue
                                present_distance = max(1, ((node_position[j][0] - ap_position_traffic[i][0]) ** 2 + (
                                        node_position[j][1] - ap_position_traffic[i][1]) ** 2) ** 0.5)
                                whole_traffic_weight_temp[int(child.value)][j] += (weight[
                                                                                       index_temp_weight] / weight_total) * child_traffic_num
                                missing_traffic_weight_temp[int(child.value)] -= (weight[
                                                                                      index_temp_weight] / weight_total) * child_traffic_num
                                trans_delay_total += (weight[
                                                          index_temp_weight] / weight_total) * child_traffic_num * present_distance
                                index_temp_weight += 1
                    else:
                        for i in range(len(state[int(node.value)])):
                            distance = []
                            for j in range(len(state[int(child.value)])):
                                if state[int(child.value)][j] != 0:
                                    distance.append(max(1, ((node_position[j][0] - node_position[i][0]) ** 2 + (
                                            node_position[j][1] - node_position[i][1]) ** 2) ** 0.5 /
                                                        state[int(child.value)][j]))
                            distance_count = sum(distance)
                            weight = []
                            for j in range(len(distance)):
                                weight.append(1 / distance[j])
                            weight_total = sum(weight)
                            # non_zero_indices = np.where(state[int(child.value)] != 0)[0]
                            index_temp_weight = 0
                            for j in range(len(state[int(child.value)])):
                                if state[int(child.value)][j] == 0:
                                    continue
                                present_distance = max(1, ((node_position[j][0] - node_position[i][0]) ** 2 + (
                                        node_position[j][1] - node_position[i][1]) ** 2) ** 0.5)
                                whole_traffic_weight_temp[int(child.value)][j] += (weight[
                                                                                       index_temp_weight] / weight_total) * \
                                                                                  whole_traffic_weight_temp[
                                                                                      int(node.value)][i] * (
                                                                                              child_traffic_num /
                                                                                              ms_accoount_info[
                                                                                                  int(node.value)])
                                missing_traffic_weight_temp[int(child.value)] -= (weight[
                                                                                      index_temp_weight] / weight_total) * \
                                                                                 whole_traffic_weight_temp[
                                                                                     int(node.value)][i] * (
                                                                                             child_traffic_num /
                                                                                             ms_accoount_info[
                                                                                                 int(node.value)])
                                trans_delay_total += (weight[index_temp_weight] / weight_total) * \
                                                     whole_traffic_weight_temp[int(node.value)][i] * present_distance
                                index_temp_weight += 1
                queue.append((child, child_traffic_num))
        queueing_delay_total = 0
        for i in range(len(whole_traffic_weight_temp)):
            for j in range(len(whole_traffic_weight_temp[0])):
                state[len(ms_resource_info) + i][j] = whole_traffic_weight_temp[i][j]
                if whole_traffic_weight_temp[i][j] == 0:
                    continue
                for k in range(state[i][j]):
                    queueing_delay_total += self.queue_delay((whole_traffic_weight_temp[i][j] / state[i][j]))
        missing_traffic_delay_total = 0
        for i in range(len(missing_traffic_weight_temp)):
            if missing_traffic_weight_temp[i] > 0:
                missing_traffic_delay_total += missing_traffic_weight_temp[
                                                   i] * missing_traffic_delay  # 因为这部分流量无处可去，直接超时了，所以加一个大一点的时延
        delay_total = trans_delay_total + queueing_delay_total + missing_traffic_delay_total
        return delay_total, state, trans_delay_total, queueing_delay_total, missing_traffic_delay_total

    def delay_nms(self, state):
        delay_total, state, trans_delay_total, queueing_delay_total, missing_traffic_delay_total = self.traffic_num(
            state)
        return delay_total, state, trans_delay_total, queueing_delay_total, missing_traffic_delay_total

    def queue_delay(self, n):  # 排队时延，基数是1ms，也就是t_exe的时间是2ms
        return 50 * n * (n + 1) // 2

    def delay(self, state, ms):
        father_status = [[250, 250, 16]]
        distance_total = 0
        trans_delay_total = 0
        request_accept = []
        m1_sum = sum(state[:8])
        if max(state[ms]) == 0:
            return 16 * 400 + int(self.queue_delay(16))
        for i in range(len(father_status)):
            for j in range(len(state[ms])):
                for _ in range(state[ms][j]):
                    distance_total += ((node_position[j][0] - father_status[i][0]) ** 2 + (
                                node_position[j][1] - father_status[i][1]) ** 2) ** 0.5
        for i in range(len(father_status)):
            request_num = father_status[i][2]
            weight = []
            for j in range(len(state[ms])):
                if state[ms][j] == 0:
                    weight.append(0)
                else:
                    weight.append(state[ms][j] / ((node_position[j][0] - father_status[i][0]) ** 2 + (
                                node_position[j][1] - father_status[i][1]) ** 2) ** 0.5)
            weight_total = sum(weight)
            for j in range(len(state[ms])):
                if weight[j] == 0 or request_num == 0:
                    request_accept.append(0)
                else:
                    temp_request_num = int(father_status[i][2] * (weight[j] / weight_total))
                    if temp_request_num <= request_num:
                        request_accept.append(temp_request_num)
                        trans_delay_total += state[ms][j] * self.queue_delay(int(temp_request_num / state[ms][j]))
                        request_num -= temp_request_num
                    else:
                        request_accept.append(request_num)
                        trans_delay_total += state[ms][j] * self.queue_delay(int(request_num / state[ms][j]))
                        request_num = 0
            if request_num > 0:
                min_weight_index = weight.index(min(weight))
                request_accept[min_weight_index] += request_num
                # trans_delay_total += self.queue_delay(request_num)
                request_num = 0
            delay_temp = 0
            for j in range(len(request_accept)):
                delay_temp += request_accept[j] * ((node_position[j][0] - father_status[i][0]) ** 2 + (
                            node_position[j][1] - father_status[i][1]) ** 2) ** 0.5
            trans_delay_total += delay_temp
        return trans_delay_total

    def step(self, state, action, ms):
        reward = 0
        state_old = copy.deepcopy(state)
        for i in range(2 * len(ms_resource_info), 2 * len(ms_resource_info) + len(node_resource[0])):
            if state[i][action] >= ms_resource_info[ms][i - 2 * len(ms_resource_info)]:
                # reward += 50
                pass
            else:
                new_delay, state_next, trans_delay_total, queueing_delay_total, missing_traffic_delay_total = self.delay_nms(
                    state)
                reward -= new_delay
                return reward, state_next, trans_delay_total, queueing_delay_total, missing_traffic_delay_total, new_delay
        # reward += 2000
        for i in range(2 * len(ms_resource_info), 2 * len(ms_resource_info) + len(node_resource[0])):
            state[i][action] -= ms_resource_info[ms][i - 2 * len(ms_resource_info)]
        state[ms][action] += 1
        # old_delay = self.delay(state_old, ms)
        new_delay, state_next, trans_delay_total, queueing_delay_total, missing_traffic_delay_total = self.delay_nms(
            state)
        # reward += old_delay - new_delay
        reward -= new_delay
        # print(f'old_delay: {old_delay}, new_delay: {new_delay}, reward: {reward}')

        return reward, state_next, trans_delay_total, queueing_delay_total, missing_traffic_delay_total, new_delay


# 用于优先级经验回放的SumTree
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(left + 1, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[data_idx])


# 定义Q网络
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        features = self.feature(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + (advantages - advantages.mean())


# 定义DQN代理
class D3QNAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=200, min_epsilon=0.005,
                 max_epsilon=0.95, buffer_size=1024, batch_size=128, alpha=0.6, beta=0.4,
                 beta_increment=0.001, eps=1e-6):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.eps = eps
        self.max_priority = 1.0  # 初始最大优先级

        self.q_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # self.replay_buffer = deque(maxlen=buffer_size)
        # self.criterion = nn.MSELoss()
        self.replay_buffer = SumTree(buffer_size)  # 替换为SumTree
        self.buffer_size = buffer_size
        self.criterion = nn.MSELoss(reduction='none')  # 修改为逐个样本计算损失

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state = torch.tensor([state], dtype=torch.float32).to(device)
            q_values = self.q_network(state)
            return torch.argmax(q_values).item()

    def store_experience(self, state, action, reward, next_state):
        experience = (state, action, reward, next_state)
        # 使用当前最大优先级存储新经验
        self.replay_buffer.add(self.max_priority, experience)

    def update_network(self):
        if self.replay_buffer.n_entries < self.batch_size:
            return

        # 采样
        batch_size = self.batch_size
        indices = []
        experiences = []
        priorities = []
        segment = self.replay_buffer.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, priority, exp = self.replay_buffer.get(s)
            indices.append(idx)
            experiences.append(exp)
            priorities.append(priority)

        # 解析经验数据
        states, actions, rewards, next_states = zip(*experiences)

        # 转换为Tensor
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.int64).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)

        # 计算当前Q值和目标Q值
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            # Double DQN核心改进：用q_network选择动作，target_network评估
            next_actions = self.q_network(next_states).argmax(dim=1)  # 用当前网络选择动作
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()  # 用目标网络评估
            target_q_values = rewards + self.gamma * next_q_values

        # 计算TD误差并更新优先级
        td_errors = target_q_values - q_values
        new_priorities = (td_errors.abs().detach().cpu().numpy() + self.eps) ** self.alpha

        # 更新最大优先级
        current_max = np.max(new_priorities)
        if current_max > self.max_priority:
            self.max_priority = current_max

        # 更新SumTree中的优先级
        for idx, priority in zip(indices, new_priorities):
            self.replay_buffer.update(idx, priority)

        # 计算重要性采样权重
        priorities = np.array(priorities)
        probabilities = priorities / self.replay_buffer.total()
        weights = (self.buffer_size * probabilities) ** -self.beta
        weights = torch.tensor(weights / weights.max(), dtype=torch.float32).to(device)

        # 计算加权损失
        loss = (weights * self.criterion(q_values, target_q_values)).mean()

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新beta
        self.beta = min(1.0, self.beta + self.beta_increment)

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        global episode
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-episode / self.epsilon_decay)


def normalize_data(ap_account_info, ms_accoount_info, traffic_deliver_info):
    ms_accoount_info_copy = copy.deepcopy(ms_accoount_info)
    # 实例数量少的优先部署1个，避免部署结束后导致这种实例一个没有部署成功从而产生大量的相关请求无法处理的延迟
    # queue_b = sorted(range(len(ms_accoount_info_copy)), key=lambda i: ms_accoount_info_copy[i], reverse=False)
    queue_b = []
    while any(num != 1 for num in ms_accoount_info_copy):
        # 找到最大值的索引
        max_value = max(ms_accoount_info_copy)
        max_indices = [i for i, num in enumerate(ms_accoount_info_copy) if num == max_value]

        # 从左到右减1
        for i in max_indices:
            ms_accoount_info_copy[i] -= 1
            queue_b.append(i)


    return queue_b


def mask_state(state, ms):
    global request_accept_info
    state_mask = [0 for _ in range(len(node_position))]
    if ms != 0 and ms != 7:
        for i in range(1, len(request_accept_info[ms])):  # 这里的1是因为第一个是AP，不是微服务
            if request_accept_info[ms][i] != 0:
                for j in range(len(state[len(ms_resource_info) + i])):
                    state_mask[j] += state[len(ms_resource_info) + i - 1][j]
    for i in range(len(state[ms])):
        state_mask.append(state[ms][i])
    for i in range(2 * len(ms_resource_info), 2 * len(ms_resource_info) + len(node_resource[0])):
        for j in range(len(node_position)):
            state_mask.append(state[i][j])
    state_mask.append(ms)
    return state_mask


# 调用函数
queue = normalize_data(ap_account_info, ms_accoount_info, traffic_deliver_info)
print(queue)
print(len(queue))

# 环境
env = Env()
state_dim = 301
action_dim = len(node_position)

# 初始化代理
agent = D3QNAgent(state_dim, action_dim)

# 训练循环
episodes = episode_count(1000)
rewards = []
final_delays = []
cur_trans_delay = []
cur_queueing_delay = []
cur_missing_traffic_delay = []
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
line1, = ax1.plot(rewards, label='Reward')
line2, = ax2.plot(final_delays, label='final_delay')
line3, = ax3.plot(cur_trans_delay, label='trans_delay')
line4, = ax3.plot(cur_queueing_delay, label='queueing_delay')
line5, = ax3.plot(cur_missing_traffic_delay, label='missing_traffic_delay')
# 设置标题和标签
ax1.set_title("Reward Curve")
ax1.set_xlabel("Episodes")
ax1.set_ylabel("Reward")
ax1.legend()  # 显示图例
ax2.set_title("Final Delay Curve")
ax2.set_xlabel("Episodes")
ax2.set_ylabel("Final Delay")
ax2.legend()  # 显示图例
ax3.set_title("Different Dalay Curve")
ax3.set_xlabel("Episodes")
ax3.set_ylabel("Dalay")
ax3.legend()  # 显示图例

WHOLE_STATE = generate_whole_state(node_position, ms_resource_info, whole_traffic_weight, node_resource)
for episode in range(episodes):
    # state = np.array(whole_state).flatten()
    state = copy.deepcopy(WHOLE_STATE)
    total_reward = 0
    queue_temp = copy.deepcopy(queue)
    rewards_in_episode = []  # 用于存储当前episode的所有reward
    while queue_temp:
        ms_temp = int(queue_temp.pop(0))
        state_mask_action = mask_state(state, ms_temp)
        action = agent.choose_action(state_mask_action)

        reward, next_state, trans_delay_total, queueing_delay_total, missing_traffic_delay_total, new_delay = env.step(
            state, action, ms_temp)
        next_state_mask = mask_state(next_state, ms_temp)
        state = next_state
        agent.store_experience(state_mask_action, action, reward, next_state_mask)
        agent.update_network()
        total_reward += reward
        rewards_in_episode.append(reward)  # 记录当前步骤的reward
    # 添加total_reward到当前episode数据的末尾
    row_data = rewards_in_episode.copy()
    row_data.append(total_reward)

    # 将数据写入Excel新行
    ws.append(row_data)

    # 绘图
    agent.update_target_network()
    agent.decay_epsilon()
    rewards.append(total_reward)
    final_delays.append(new_delay)
    cur_trans_delay.append(trans_delay_total)
    cur_queueing_delay.append(queueing_delay_total)
    cur_missing_traffic_delay.append(missing_traffic_delay_total)

    print(f'Episode {episode + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}')
    # 图1
    line1.set_ydata(rewards)
    line1.set_xdata(range(len(rewards)))
    ax1.relim()
    ax1.autoscale_view()
    fig1.canvas.draw()
    # 图2
    line2.set_ydata(final_delays)
    line2.set_xdata(range(len(final_delays)))
    ax2.relim()
    ax2.autoscale_view()
    fig2.canvas.draw()
    # 图3
    line3.set_ydata(cur_trans_delay)
    line3.set_xdata(range(len(cur_trans_delay)))
    line4.set_ydata(cur_queueing_delay)
    line4.set_xdata(range(len(cur_queueing_delay)))
    line5.set_ydata(cur_missing_traffic_delay)
    line5.set_xdata(range(len(cur_missing_traffic_delay)))
    ax3.relim()
    ax3.autoscale_view()
    fig3.canvas.draw()
plt.ioff()
# 自动保存图片
image_name = "figure"
file_name = f"{image_name}_{timestamp}.png"
file_path = os.path.join(save_directory, file_name)
plt.savefig(file_path)
plt.close()

# 保存Excel文件（自动处理路径拼接）
if not os.path.exists(save_directory):
    os.makedirs(save_directory)
full_path = os.path.join(save_directory, file_name)
wb.save(full_path)
print(f"文件已保存至：{full_path}")
