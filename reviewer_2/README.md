# Reviewer 2 图片与数据说明

本目录对应 R2 回复草稿中嵌入的 6 张图，覆盖分层贡献归因、实际运行时间和模块消融。

| 子目录 | 图片 | 数据文件 | 绘图脚本 | 用途 |
|---|---|---|---|---|
| 01_r2_1_clustering_contribution/ | r2_1_clustering_contribution.png | r2_1_strengthened_clustering_test.csv | plot_r2_1_contribution_figures.py | 固定下层部署策略，比较五种聚类方法。 |
| 02_r2_1_rl_optimizer_contribution/ | r2_1_rl_optimizer_contribution.png | r2_1_strengthened_rl_test.csv | plot_r2_1_contribution_figures.py | 固定 tuned FPDWGC，比较 RL 优化器。 |
| 03_r2_2_fpdwgc_maintenance_scaling/ | r2_2_fpdwgc_maintenance_scaling.png | r2_2_runtime_summary.csv | plot_r2_2_scalability_figures.py | FPDWGC 控制面维护时间扩展性。 |
| 04_r2_2_mpd3qn_inference_scaling/ | r2_2_mpd3qn_inference_scaling.png | r2_2_runtime_summary.csv | plot_r2_2_scalability_figures.py | MPD3QN 在线推理与全局候选评估时间。 |
| 05_r2_3_factorial_module_metrics/ | r2_3_factorial_module_metrics.png | r2_3_module_factorial_ablation.csv | plot_r2_3_module_contribution_figures.py | FPDWGC、PER、动作掩码的 2×2×2 因子消融。 |
| 06_r2_3_single_factor_contribution/ | r2_3_single_factor_contribution.png | r2_3_single_factor_effects.csv | plot_r2_3_module_contribution_figures.py | 三个模块的单因素主效应。 |

R2-2 的两张图共用运行时间汇总数据，因此各自目录均保存了一份。R2-3 的训练收敛统计来自对应历史训练轨迹；此目录中的图数据不支持“8 个组合均完整重新训练”的说法。
