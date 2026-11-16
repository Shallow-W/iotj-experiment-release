import matplotlib.pyplot as plt
import numpy as np

def normal(scalars, weight):
    """
    使用指数移动平均平滑数据。
    参数:
        scalars (list或np.array): 原始数据。
        weight (float): 平滑权重，介于0到1之间。越大平滑程度越高。
    返回:
        list: 平滑后的数据。
    """
    if len(scalars) == 0:
        return []
    smoothed = [float(scalars[0])]
    for s in scalars[1:]:
        prev = smoothed[-1]
        smoothed_val = prev * weight + s * (1 - weight)
        smoothed.append(smoothed_val)
    return smoothed

def auto(scalars, init_weight, weight_decay):
    """
    使用指数移动平均平滑数据。
    参数:
        scalars (list或np.array): 原始数据。
        weight (float): 平滑权重，介于0到1之间。越大平滑程度越高。
    返回:
        list: 平滑后的数据。
    """
    if len(scalars) == 0:
        return []
    smoothed = [float(scalars[0])]
    for s in scalars[1:]:
        prev = smoothed[-1]
        smoothed_val = prev * init_weight + s * (1 - init_weight)
        smoothed.append(smoothed_val)
        init_weight *= weight_decay
        init_weight = 1.0 - init_weight
    return smoothed
# # 生成随机奖励数据（示例用）
# np.random.seed(42)
# raw_rewards = np.random.randn(200).cumsum()  # 随机游走模拟奖励曲线
#
# # 应用平滑函数
# smoothed_rewards = smooth_data(raw_rewards, weight=0.9)  # 调整 weight 值改变平滑程度
#
# # 绘制对比图
# plt.figure(figsize=(12, 6))
# plt.plot(raw_rewards, alpha=0.3, label='Raw Rewards')
# plt.plot(smoothed_rewards, color='red', linewidth=2, label='Smoothed (weight=0.9)')
# plt.title('Reward Curve Comparison')
# plt.xlabel('Episode')
# plt.ylabel('Reward')
# plt.grid(True, alpha=0.3)
# plt.legend()
# plt.show()