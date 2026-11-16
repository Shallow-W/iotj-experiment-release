# IoTJ 补充实验代码与结果

本目录按“审稿人 → 单张最终入文图片 → 图片与对应数据”的层级整理。范围严格以四份 response_draft/reviewer_*_response_with_data.md 中实际嵌入的 PNG 为准，共 14 张图：

| 审稿人 | 图片数 | 目录 |
|---|---:|---|
| Reviewer 1 | 3 | [参数敏感性](reviewer_1/README.md) |
| Reviewer 2 | 6 | [分层归因、扩展性与消融](reviewer_2/README.md) |
| Reviewer 3 | 3 | [高负载退化与基线比较](reviewer_3/README.md) |
| Reviewer 4 | 2 | [动态工作负载](reviewer_4/README.md) |

每个图片子目录均可独立使用，包含：

- 最终进入回复草稿的 PNG 图片；
- 该图直接使用的 CSV/JSON 数据；
- 生成该图的 Python 绘图脚本；
- 图名与数据文件均保持源文件名，未改写数据内容。

两张图若使用同一份源数据，会在两个图片目录中各复制一份，以避免取用时跨目录查找。例如，R2-2 的两张扩展性图共用 r2_2_runtime_summary.csv，R4 的两张动态图共用逐时隙数据。

## 用现有结果重画图片

使用 Python 3.9 安装绘图依赖。从本目录运行以下命令可重画 R4 主图，其余图片的脚本名称见上表链接。

~~~bash
python3 -m pip install -r requirements.txt
python3 reviewer_4/01_r4_1_workload_and_improvement/plot_r4_1_workload_and_improvement.py
~~~

各脚本读取同目录中的数据，输出 PNG 和 PDF 到该图子目录的 reproduced/。参考 PNG 保存在子目录顶层，保留了原始导出的版本。不同平台上的字体可能使重新导出的图片略有差异。

## 实验代码与结果

发布包中的 resub-exp/experiments/ 保留原始实验脚本、参数配置和选定的结果表，沿用实验间相互引用的目录结构。图目录中的 Python 脚本提供从结果表重画的入口。全量重跑所需输入见 [实验复现说明](EXPERIMENTS.md)。
