# Reviewer 4 图片与数据说明

本目录对应 R4 回复草稿中嵌入的 2 张动态工作负载图片。

| 子目录 | 图片 | 数据文件 | 绘图脚本 | 用途 |
|---|---|---|---|---|
| 01_r4_1_workload_and_improvement/ | r4_1_workload_and_improvement.png | r4_1_controlled_hotspot_slot_metrics.csv、r4_1_controlled_hotspot_update_events.csv、r4_1_controlled_hotspot_phase_summary.csv、r4_1_controlled_hotspot_integrity_check.json | plot_r4_1_workload_and_improvement.py | 300 个 PeMS 时隙中的工作负载、三指标改善、更新事件和因果性检查。 |
| 02_r4_1_simple_dynamic_improvement/ | r4_1_simple_dynamic_improvement.png | r4_1_controlled_hotspot_slot_metrics.csv、r4_1_controlled_hotspot_update_events.csv | plot_r4_1_simple_dynamic_improvement.py | 热点阶段三项性能改善趋势及更新时刻。 |

两张图共用逐时隙数据。辅图使用更新事件数据绘制第 71、161、181 个时隙的竖线。主图目录也保留更新事件、阶段汇总和完整性检查。逐时隙数据中有少量短时负改善窗口。
