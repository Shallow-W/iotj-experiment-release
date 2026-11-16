import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import math

def constrained_kmeans(points, centers, max_iterations, max_capacity, tolerance):
    cluster_ids = np.zeros(len(points), dtype=int)
    prev_centers = centers.copy()
    for iteration in range(max_iterations):
        # 计算所有点到各中心的距离
        distances = np.linalg.norm(points[:, np.newaxis] - centers, axis=2)
        # 生成随机处理顺序
        processing_order = np.random.permutation(len(points))
        # 初始化簇容量计数器
        cluster_counts = np.zeros(len(centers), dtype=int)
        new_cluster_ids = np.zeros_like(cluster_ids)
        # 分配节点到最近的可用簇
        for idx in processing_order:
            sorted_clusters = np.argsort(distances[idx])
            for cluster in sorted_clusters:
                if cluster_counts[cluster] < max_capacity:
                    new_cluster_ids[idx] = cluster
                    cluster_counts[cluster] += 1
                    break
        # 更新簇中心
        new_centers = np.zeros_like(centers)
        for i in range(len(centers)):
            members = points[new_cluster_ids == i]
            if len(members) > 0:
                new_centers[i] = members.mean(axis=0)
            else:
                new_centers[i] = centers[i]  # 保持原中心避免空簇
        # 检查收敛条件
        if np.linalg.norm(new_centers - prev_centers) < tolerance:
            break
        prev_centers = new_centers.copy()
        centers[:] = new_centers
        cluster_ids[:] = new_cluster_ids
    return cluster_ids, centers

def main(points, num_clusters):
    max_capacity = math.ceil(len(points)/num_clusters)
    print('max_capacity', max_capacity)
    max_iterations = 100
    tolerance = 1e-4
    points = np.array(points)
    # 初始化聚类中心（使用K-means++）
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', n_init=10)
    kmeans.fit(points)
    centers = kmeans.cluster_centers_
    # 运行改进算法
    final_clusters, final_centers = constrained_kmeans(points, centers, max_iterations, max_capacity, tolerance)
    # 生成存放聚类标签的列表
    cluster_labels = final_clusters.tolist()
    # 输出聚类标签列表
    print("cluster_labels:\n", cluster_labels)
    # 结果分析
    cluster_sizes = np.bincount(final_clusters, minlength=num_clusters)
    print('cluster_sizes:\n',cluster_sizes)
    print("\nCluster size distribution:")
    print(f"Minimum size: {np.min(cluster_sizes)}")
    print(f"Maximum size: {np.max(cluster_sizes)}")
    print(f"Average size: {np.mean(cluster_sizes):.2f}")
    # 计算簇内平均距离
    total_distance = 0
    for i in range(num_clusters):
        members = points[final_clusters == i]
        if len(members) > 0:
            centroid = final_centers[i]
            total_distance += np.sum(np.linalg.norm(members - centroid, axis=1))
    print(f"\nAverage intra-cluster distance: {total_distance / len(points):.2f} meters")
    # 可视化结果
    plt.figure(figsize=(12, 8))
    plt.scatter(points[:, 0], points[:, 1], c=final_clusters, cmap='tab20', s=15, alpha=0.6)
    plt.scatter(final_centers[:, 0], final_centers[:, 1], c='red', s=50, marker='X', edgecolor='black')
    plt.title('Capacity Constrained Clustering Result')
    plt.xlabel('X Coordinate (meters)')
    plt.ylabel('Y Coordinate (meters)')
    plt.show()
    return cluster_labels, cluster_sizes

# 主函数运行
if __name__ == "__main__":
    # 参数设置
    num_points = 490
    area_size = 500
    num_clusters = 50
    # 生成模拟数据（500x500范围内的随机点）
    np.random.seed(42)
    points = np.random.rand(num_points, 2) * area_size
    print('points',points)
    cluster_labels, cluster_sizes = main(points, num_clusters)
