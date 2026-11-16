# 实验代码与输入

发布包保留 resub-exp/experiments/ 下的原始运行脚本和共享评估代码。每个实验的 scripts/、configs/（如有）和 results/ 分别保存运行入口、参数配置和已产出的结果。下表入口相对于解压后的仓库根目录。

| 实验 | 主要入口 |
|---|---|
| R1-1 参数配置与选择实验 | resub-exp/experiments/reviewer_1/01_parameter_reproducibility/scripts/run_r1_1_parameter_selection_experiments.py |
| R1-2 退化与权重敏感性 | resub-exp/experiments/reviewer_1/02_parameter_sensitivity/scripts/ 下的 run_r1_2_degradation_univariate_trends.py、run_r1_2_sigma_fullflow_sensitivity.py |
| R2-1 分层贡献 | resub-exp/experiments/reviewer_2/01_hierarchical_contribution_ablation/scripts/run_r2_1_strengthened_ablation.py |
| R2-2 扩展性计时 | resub-exp/experiments/reviewer_2/02_scalability_runtime/scripts/run_r2_2_scalability_runtime.py |
| R2-3 模块因子消融 | resub-exp/experiments/reviewer_2/03_module_contribution_ablation/scripts/run_r2_3_module_contribution_ablation.py |
| R3-1 高负载退化 | resub-exp/experiments/reviewer_3/01_high_load_degradation/scripts/run_r3_1_high_load_degradation.py |
| R3-2 聚类基线 | resub-exp/experiments/reviewer_3/02_clustering_baselines/scripts/run_r3_2_clustering_metrics.py |
| R3-3 论文尺度基线 | resub-exp/experiments/reviewer_3/03_non_rl_baselines/paper_aligned_unified/scripts/run_r3_3_paper_aligned_unified.py |
| R4-1 受控热点适应 | resub-exp/experiments/reviewer_4/01_dynamic_workload_evolution/scripts/run_r4_1_controlled_hotspot_adaptation.py |

全量重跑需要先安装 requirements-experiments.txt。原始运行脚本按照以下相对路径读取项目输入，发布包提供已计算的 CSV/JSON。

| 本地输入目录 | 文件和用途 |
|---|---|
| hjw_disanzhangbishe/50endpoints_nms/ | node_resource.xlsx、ms_resource.xlsx、app.xlsx、request.xlsx，用于 50 节点评估。 |
| hjw_disanzhangbishe/100endpoints_nms/ | 同名四个工作簿，用于 R3-3 的 100 节点数据评估。 |
| hjw/chapter_3/choosed_data/ | 历史训练时延与回报工作簿，用于相应的训练曲线和收敛统计。各脚本记录所需文件名。 |
| hjw_disanzhangbishe/ | R1-1 的实现证据检查脚本在此查找 D3QN_STATEMASK_PRI_YANMA.py 和 D3QN_STATEMASK_PRI.py。 |

原始运行入口会写入各实验的 results/ 或 figures/，建议在解压副本中执行。上面列出的原始输入由研究者单独提供。已有结果的作图步骤见 [总 README](README.md)。
