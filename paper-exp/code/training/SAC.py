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
from torch.distributions import Categorical
import torch.nn.functional as F
from collections import namedtuple
import collections
import math
# 修改经验存储方式（确保维度一致）
Transition = namedtuple('Transition',
    ['state', 'action', 'reward', 'next_state', 'done', 'log_prob', 'value'])
from collections import namedtuple
import generate_whole_state as gws
# 让用户输入文件名和保存路径
original1_name = "rewards_data_sac"
original2_name = "delays_data_sac"
original3_name = "resources_data_sac"
save_directory = output_directory(__file__)

# 自动生成带时间戳的文件名
timestamp = datetime.now().strftime("%Y%m%d_%H%M")  # 格式示例：20231015_1645
file_name1 = f"{original1_name}_{timestamp}.xlsx"  # 强制使用.xlsx扩展名
file_name2 = f"{original2_name}_{timestamp}.xlsx"  # 强制使用.xlsx扩展名
file_name3 = f"{original3_name}_{timestamp}.xlsx"  # 强制使用.xlsx扩展名
# 创建Excel工作簿和工作表
wb1 = Workbook()
ws1 = wb1.active
wb2 = Workbook()
ws2 = wb2.active
wb3 = Workbook()
ws3 = wb3.active

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
ap_position_traffic = [[250, 250, 14]]

# 文件路径前缀
node_number = 50
file_prefix = input_prefix(f"{node_number}endpoints_nms")

node_info_workbook = openpyxl.load_workbook(file_prefix + 'node_resource.xlsx', data_only=True)
node_info_sheet = node_info_workbook['Sheet1']
node_position = []
start_row = 2
for i in range(2, node_number + 2):
    x_value = node_info_sheet.cell(row=i, column=6).value
    y_value = node_info_sheet.cell(row=i, column=7).value
    node_position.append([x_value, y_value])

node_resource = []
for i in range(2, node_number + 2):
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
whole_state = [[0 for _ in range(len(node_position))] for _ in
               range(len(ms_resource_info))]  # 现在是10*50的矩阵，每一行代表一个微服务，每一列代表一个节点，每个元素代表这个微服务在这个节点上的实例数量
for row in whole_traffic_weight:
    whole_state.append(row)  # 现在是20*50的矩阵，前10行是微服务的实例数量，后10行是整个网络的流量权重
resource_state = [[node_resource[j][i] for j in range(len(node_resource))] for i in range(len(node_resource[0]))]
# print('')
whole_state.extend(resource_state)  # 现在是24*50的矩阵，前10行是微服务的实例数量，后4行是节点的现存资源数量
# whole_state.extend(resource_state) #本来这里要放节点的初始资源数量，做负载均衡的，但是这个版本还不考虑在奖励函数中加入负载均衡的考虑，所以先不加。
ms_queue_num = [[0 for _ in range(len(node_position))] for _ in
                range(len(ms_resource_info))]  # 每个服务器上每种微服务的请求队列的长度，注意如果这个服务器上有多个微服务，需要均分排队长度。
missing_traffic_delay = 400  # 整个网络中没有成功分配的流量的情况，具体含义就是这个微服务没有任何的部署，那么原本计划要通过它的流量就全部处于待分配状态，此时计算时延是一个定值,现在是400ms


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
                reward -= 2 * new_delay
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

def onehot_from_logits(logits, eps=0.01):
    ''' 生成最优动作的独热（one-hot）形式 '''
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    # 生成随机动作,转换成独热形式
    rand_acs = torch.autograd.Variable(torch.eye(logits.shape[1])[[
        np.random.choice(range(logits.shape[1]), size=logits.shape[0])
    ]],
                                       requires_grad=False).to(logits.device)
    # 通过epsilon-贪婪算法来选择用哪个动作
    return torch.stack([
        argmax_acs[i] if r > eps else rand_acs[i]
        for i, r in enumerate(torch.rand(logits.shape[0]))
    ])


def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """从Gumbel(0,1)分布中采样"""
    U = torch.autograd.Variable(tens_type(*shape).uniform_(),
                                requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ 从Gumbel-Softmax分布中采样"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(
        logits.device)
    return F.softmax(y / temperature, dim=1)


def gumbel_softmax(logits, temperature=1.0):
    """从Gumbel-Softmax分布中采样,并进行离散化"""
    y = gumbel_softmax_sample(logits, temperature)
    y_hard = onehot_from_logits(y)
    y = (y_hard.to(logits.device) - y).detach() + y
    # 返回一个y_hard的独热量,但是它的梯度是y,我们既能够得到一个与环境交互的离散动作,又可以
    # 正确地反传梯度
    return y

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class QValueNet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class SAC:
    ''' 处理离散动作的SAC算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 alpha_lr, target_entropy, tau, gamma, device):
        # 策略网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        # 第一个Q网络
        self.critic_1 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        # 第二个Q网络
        self.critic_2 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_1 = QValueNet(state_dim, hidden_dim,
                                         action_dim).to(device)  # 第一个目标Q网络
        self.target_critic_2 = QValueNet(state_dim, hidden_dim,
                                         action_dim).to(device)  # 第二个目标Q网络
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(),
                                                   lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(),
                                                   lr=critic_lr)
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr)
        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.epsilon = 1
        self.min_epsilon = 0.005
        self.max_epsilon = 0.995
        self.epsilon_decay = 200
        self.action_dim = action_dim
        self.temperature = 10.0  # 初始温度较高以促进探索

    def take_action(self, state, explore=True):
        # if random.random() < self.epsilon:
        #     action = random.randint(0, self.action_dim - 1)
        #     return action
        # else:
        #     state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        #     action_logits = self.actor(state)
        #     if explore:
        #         # 使用当前温度生成动作概率
        #         action_probs = gumbel_softmax(action_logits, temperature=self.temperature)
        #     else:
        #         action_probs = onehot_from_logits(action_logits, eps=0.0)  # 测试时无探索
        #     action = torch.multinomial(action_probs, 1).item()


        if random.random() < self.epsilon:
            action = random.randint(0, self.action_dim - 1)
            return action
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            probs = self.actor(state)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()

        return action

    # 计算目标Q值,直接用策略网络的输出概率进行期望计算
    def calc_target(self, rewards, next_states, dones):
        next_probs = self.actor(next_states)
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
        q1_value = self.target_critic_1(next_states)
        q2_value = self.target_critic_2(next_states)
        min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value),
                               dim=1,
                               keepdim=True)
        next_value = min_qvalue + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)  # 动作不再是float类型
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_q_values = self.critic_1(states).gather(1, actions)
        critic_1_loss = torch.mean(
            F.mse_loss(critic_1_q_values, td_target.detach()))
        critic_2_q_values = self.critic_2(states).gather(1, actions)
        critic_2_loss = torch.mean(
            F.mse_loss(critic_2_q_values, td_target.detach()))

        # # 转换动作索引为独热编码
        # actions_onehot = F.one_hot(actions.squeeze(), num_classes=self.action_dim).float().to(device)
        # # 更新两个Q网络
        # td_target = self.calc_target(rewards, next_states, dones)
        # q_input = torch.cat([states, actions_onehot], dim=1)
        #
        # critic_1_q_values = self.critic_1(q_input)
        # critic_1_loss = torch.mean(F.mse_loss(critic_1_q_values, td_target.detach()))
        #
        # critic_2_q_values = self.critic_2(q_input)
        # critic_2_loss = torch.mean(F.mse_loss(critic_2_q_values, td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 更新策略网络
        probs = self.actor(states)
        log_probs = torch.log(probs + 1e-8)
        # 直接根据概率计算熵
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)  #
        q1_value = self.critic_1(states)
        q2_value = self.critic_2(states)
        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value),
                               dim=1,
                               keepdim=True)  # 直接根据概率计算期望
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean(
            (entropy - target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

    def decay_epsilon(self):
        global episode
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-episode / self.epsilon_decay)

def normalize_data(ms_accoount_info):
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

    print('cur whole_state:\n', whole_state)
    return whole_state

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)

def mask_state(state, ms):
    whole_state_temp = state.copy()
    # 添加归一化处理
    # max_val = max([max(row) for row in whole_state])
    # min_val = min([min(row) for row in whole_state])
    # whole_state = [[(x - min_val) / (max_val - min_val + 1e-8) for x in row] for row in whole_state]
    whole_state_temp2 = np.array(whole_state_temp).flatten()
    # whole_state = np.append(whole_state, ms)
    return whole_state_temp2

def norm_state(whole_state, state, ms):
    state_temp = state.copy()
    # 添加归一化处理
    max_val_list = [max(row) for row in state_temp]
    min_val_list = [min(row) for row in state_temp]
    normed_state = []
    for i in range(len(state_temp)):
        row_normed_state = []
        for j in range(len(state_temp[0])):
            if max_val_list[i] == 0:
                row_normed_state.append(0)
            else:
                row_normed_state.append((state_temp[i][j]-min_val_list[i]) / (max_val_list[i] - min_val_list[i]) + 1e-8)
        normed_state.append(row_normed_state)

    # 变为1维
    whole_state_temp2 = np.array(normed_state).flatten()
    # whole_state = np.append(whole_state, ms)
    return whole_state_temp2
# 调用函数
queue = normalize_data(ms_accoount_info)
print('len(queue)', len(queue))
# 环境
env = Env()
state_dim = len(whole_state) * len(whole_state[0])
action_dim = len(node_position)
actor_lr = 1e-3
critic_lr = 1e-2
alpha_lr = 1e-2
num_episodes = 200
hidden_dim = 128
gamma = 0.98
tau = 0.005  # 软更新参数
buffer_size = 10000
minimal_size = 500
batch_size = 64
target_entropy = -1
# 初始化代理
replay_buffer = ReplayBuffer(buffer_size)
agent = SAC(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, alpha_lr,
            target_entropy, tau, gamma, device)

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
ax3.legend()  # 显示图例\

episodes = episode_count(1000)
whole_state = gws.main(node_position, ms_resource_info, whole_traffic_weight, node_resource, seed=0)

return_list = []
for episode in range(episodes):
    total_reward = 0
    state = copy.deepcopy(whole_state)
    queue_temp = queue.copy()
    done = False
    rewards_in_episode = []
    while not done:
        ms_temp = int(queue_temp.pop(0))
        masked_state1 = norm_state(whole_state, state, ms_temp)
        action = agent.take_action(masked_state1)
        # print('one action', action)
        reward, next_state, trans_delay_total, queueing_delay_total, missing_traffic_delay_total, new_delay = env.step(state, action, ms_temp)
        if len(queue_temp) == 0:
            done = True
        masked_state2 = norm_state(next_state, state, ms_temp)
        replay_buffer.add(masked_state1, action, reward, masked_state2, done)
        state = next_state
        total_reward += reward
        if replay_buffer.size() > minimal_size:
            b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
            transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
            agent.update(transition_dict)
    rewards_in_episode.append(reward)  # 记录当前步骤的reward
    agent.decay_epsilon()
    # 每个episode结束后更新网络
    row_data = rewards_in_episode.copy()
    trans_copy = copy.deepcopy(trans_delay_total)
    queueing_copy = copy.deepcopy(queueing_delay_total)
    # 将数据写入Excel新行
    ws1.append(row_data)
    ws2.append([trans_copy + queueing_copy])
    state_reshape = np.array(state).reshape(len(whole_state), len(whole_state[0]))
    data = state_reshape[20:, ].tolist()
    for row in data:
        ws3.append(row)
    # 绘图
    rewards.append(total_reward)
    final_delays.append(new_delay)
    cur_trans_delay.append(trans_delay_total)
    cur_queueing_delay.append(queueing_delay_total)
    cur_missing_traffic_delay.append(missing_traffic_delay_total)

    print(f'Episode {episode + 1}/{episodes}, Total Reward: {total_reward}')
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

# 保存文件（自动处理路径拼接）
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# 自动保存图片
image_name = "figure_sac"
figure_name1 = f"{image_name}1_{timestamp}.png"
figure_name2 = f"{image_name}2_{timestamp}.png"
figure_name3 = f"{image_name}3_{timestamp}.png"
file_path1 = os.path.join(save_directory, figure_name1)
file_path2 = os.path.join(save_directory, figure_name2)
file_path3 = os.path.join(save_directory, figure_name3)
fig1.savefig(file_path1)
fig2.savefig(file_path2)
fig3.savefig(file_path3)
plt.close()

full_path1 = os.path.join(save_directory, file_name1)
wb1.save(full_path1)
full_path2 = os.path.join(save_directory, file_name2)
wb2.save(full_path2)
full_path3 = os.path.join(save_directory, file_name3)
wb3.save(full_path3)

print(f"文件已保存至：{save_directory}")


# for episode in tqdm(range(episodes), desc="Training Episodes"):
#     total_reward = 0
#     state = copy.deepcopy(whole_state)
#     queue_temp = queue.copy()
#     done = False
#     rewards_in_episode = []
#     while not done:
#         ms_temp = int(queue_temp.pop(0))
#         masked_state1 = mask_state(state, ms_temp)
#         action_probs, action = agent.take_action(masked_state1)
#         reward, next_state, trans_delay_total, queueing_delay_total, missing_traffic_delay_total = env.step(state, action, ms_temp)
#         if len(queue_temp) == 0:
#             done = True
#         masked_state2 = mask_state(next_state, ms_temp)
#         agent.save_experience(masked_state1, action_probs, reward, masked_state2, done)
#         agent.update()
#         state = next_state
#         total_reward += reward
#         rewards_in_episode.append(reward)  # 记录当前步骤的rewar
#     # 每个episode结束后更新网络
#     row_data = rewards_in_episode.copy()
#     trans_copy = copy.deepcopy(trans_delay_total)
#     queueing_copy = copy.deepcopy(queueing_delay_total)
#     # 将数据写入Excel新行
#     ws1.append(row_data)
#     ws2.append([trans_copy + queueing_copy])
#     state_reshape = np.array(state).reshape(len(whole_state), len(whole_state[0]))
#     data = state_reshape[20:, ].tolist()
#     for row in data:
#         ws3.append(row)
#     # 绘图
#     rewards.append(total_reward)
#     final_delays.append(-reward)
#     cur_trans_delay.append(trans_delay_total)
#     cur_queueing_delay.append(queueing_delay_total)
#     cur_missing_traffic_delay.append(missing_traffic_delay_total)
#
#     print(f'Episode {episode + 1}/{episodes}, Total Reward: {total_reward}')
#     # 图1
#     line1.set_ydata(rewards)
#     line1.set_xdata(range(len(rewards)))
#     ax1.relim()
#     ax1.autoscale_view()
#     fig1.canvas.draw()
#     # 图2
#     line2.set_ydata(final_delays)
#     line2.set_xdata(range(len(final_delays)))
#     ax2.relim()
#     ax2.autoscale_view()
#     fig2.canvas.draw()
#     # 图3
#     line3.set_ydata(cur_trans_delay)
#     line3.set_xdata(range(len(cur_trans_delay)))
#     line4.set_ydata(cur_queueing_delay)
#     line4.set_xdata(range(len(cur_queueing_delay)))
#     line5.set_ydata(cur_missing_traffic_delay)
#     line5.set_xdata(range(len(cur_missing_traffic_delay)))
#     ax3.relim()
#     ax3.autoscale_view()
#     fig3.canvas.draw()
#     plt.pause(0.01)
# plt.ioff()
#
# # 保存文件（自动处理路径拼接）
# if not os.path.exists(save_directory):
#     os.makedirs(save_directory)
#
# # 自动保存图片
# image_name = "figure_ddpg"
# figure_name1 = f"{image_name}1_{timestamp}.png"
# figure_name2 = f"{image_name}2_{timestamp}.png"
# figure_name3 = f"{image_name}3_{timestamp}.png"
# file_path1 = os.path.join(save_directory, figure_name1)
# file_path2 = os.path.join(save_directory, figure_name2)
# file_path3 = os.path.join(save_directory, figure_name3)
# fig1.savefig(file_path1)
# fig2.savefig(file_path2)
# fig3.savefig(file_path3)
# plt.close()
#
# full_path1 = os.path.join(save_directory, file_name1)
# wb1.save(full_path1)
# full_path2 = os.path.join(save_directory, file_name2)
# wb2.save(full_path2)
# full_path3 = os.path.join(save_directory, file_name3)
# wb3.save(full_path3)
#
# print(f"文件已保存至：{save_directory}")
