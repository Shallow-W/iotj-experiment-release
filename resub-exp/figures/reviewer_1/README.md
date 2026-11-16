# Reviewer 1 图片与数据说明

本目录对应 R1 回复草稿中嵌入的 3 张 R1-2 参数敏感性图片。每个子目录含该图 PNG 及对应的原始 CSV。

| 子目录 | 图片 | 数据文件 | 绘图脚本 | 用途 |
|---|---|---|---|---|
| 01_r1_2_degradation_univariate/ | r1_2_degradation_univariate_trends.png | r1_2_degradation_univariate_trends.csv | run_r1_2_degradation_univariate_trends.py | rho_th、varphi、f_min 的单变量敏感性。 |
| 02_r1_2_sigma_fullflow/ | r1_2_sigma_fullflow_sensitivity_2x2.png | r1_2_sigma_fullflow_sensitivity.csv | run_r1_2_sigma_fullflow_sensitivity.py | 4×4 sigma_delay × sigma_res 全流程网格。 |
| 03_r1_2_sigma_default_slices/ | r1_2_sigma_default_slice_trends.png | r1_2_sigma_default_slices.csv | run_r1_2_sigma_fullflow_sensitivity.py | 固定一个 sigma 为默认值时的相邻趋势。 |

前两张是正文主图，第三张为辅助图。默认 sigma_delay=sigma_res=1.0 是时延、负载方差、跨域比例与资源满足率之间的平衡设置，不应写成所有指标的全局最优。
