# Reviewer 3 图片与数据说明

本目录对应 R3 回复草稿中嵌入的 3 张主图，覆盖高负载退化、聚类基线和非 RL 基线。

| 子目录 | 图片 | 数据文件 | 绘图脚本 | 用途 |
|---|---|---|---|---|
| 01_r3_1_high_load_degradation/ | r3_1_high_load_degradation.png | r3_1_high_load_degradation_sweep.csv、r3_1_degradation_function_near_rho1.csv | run_r3_1_high_load_degradation.py | rho≈1 及更高负载下的时延、负载方差、队列时延与有界退化函数。 |
| 02_r3_2_clustering_baselines/ | r3_2_clustering_baseline_metrics.png | r3_2_clustering_baseline_metrics.csv | plot_r3_2_manuscript_aligned.py | global、static、K-means、spectral 和 tuned FPDWGC 的对比。 |
| 03_r3_3_paper_aligned_non_rl/ | r3_3_paper_aligned_unified_delay.png | r3_3_paper_aligned_unified_metrics.csv | plot_r3_3_paper_aligned_unified.py | 论文尺度对齐的 RL 与新增 deterministic non-RL 基线对比。 |

R3-3 是在相同 evaluator 和论文尺度设置下的统一比较。它支持平均响应时延结论，但不应描述为所有 RL 基线都进行了 checkpoint 级重新训练，或所有指标均最优。
