# C4 闭环 oracle 与域外速度诊断

已完成20请求/232真实DiT，0失败。四个已知训练源：NONE/V/KV均0/4数值合格；新鲜KV+域外velocity完整轨迹与PNG精确复现F。结果不构成可部署加速或训练无损结论。

- [实验协议和过程](01_实验协议与计划进度.md)
- [论文式结果、公式、流程和失败案例](02_闭环诊断结果与训练决策.md)
- [完成进度、近期预算与八个月计划](03_完成进度与后续计划.md)
- [条件训练方案](训练方案草案.md)：冻结底座、velocity/KV小头、闭环标签、边界损失与RL门槛，尚未训练。
- [20行完整质量表](data/quality_c4.csv)、[汇总JSON](data/summary_c4.json)、[独立审计](data/final_audit.json)。

远端目录为 `/workspace/diffusers-fix/research/closed_loop_oracle_20261023`。复现代码依赖相邻原实验目录 `causal_structure_pilot_20261023`、`fewshot_calibration_start_20261023` 和 `local_executor_boundary_20261022`，以及原模型权重；此归档不重新分发模型。模型SHA、种子、时间表与源码SHA见 `data/preregistration_c4.json`。Git基点4222cd5；本轮没有commit或push。

执行环境固定 `PYTHONPATH=/workspace/diffusers-fix/src`，Python为 `/workspace/venvs/qwen-edit-pilot/bin/python`。脚本职责如下：

| 脚本 | 职责 | 真实DiT |
|---|---|---:|
| `check_closed_loop.py --output <新路径>` | CPU微型FP32/BF16正确性 | 0，132微型forward |
| `run_c4.py` | 预登记和独立闭环；拒绝覆盖已有账本 | 232 |
| `audit_c4.py` | 标准库独立事件、状态、旧文件审计 | 0 |
| `analyze_c4.py --panels-only` | 生成面板；要求20请求全部完成 | 0 |
| `analyze_c4.py` | 同口径LPIPS/边界/远背景指标 | 0，使用已有评价网络 |
| `plot_c4.py` | 科学图表PNG/SVG | 0 |
| `validate_artifact.py` | 文档、数据、图及归档清单一致性 | 0 |

原记录不可删掉后重跑以替换本轮。任何新增模型对照另建阶段；C4b56DiT和条件训练/成本筛查都只是后续建议。诊断耗时包含dense donor和检查，不能用于部署加速比。测试媒体未读取，旧912个冻结文件保持一致。
