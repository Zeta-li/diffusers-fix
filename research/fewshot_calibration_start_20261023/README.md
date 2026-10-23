# 小样本校准启动

本阶段已完成工程验证、Train8的112次DiT采集和A/B统计拟合。独立Val/Test及RL尚未执行。

- [启动协议](01_启动协议与完成记录.md)：工程探针、Train8数据与调用合同。
- [启动实测结果](02_启动实测结果.md)：实际完成与待完成状态。
- [数据来源核验](data_source_evidence.md)：MagicBrush固定版本、数据隔离、许可与下载范围。
- `scripts/calibration_collector.py`：真实anchor norm1及same-state raw Value流式采集。
- `scripts/moments.py`：图级等权A/B拟合及Train内部LOO选正则。
- `scripts/value_fitted_correction.py`：独立B系数加载与只读校正。

zeta目录：`/workspace/diffusers-fix/research/fewshot_calibration_start_20261023`。工程探针与正式Train8分账，旧3demo不计入训练源图。正式数据/模型调用是否完成只认当前JSON账本，不以计划数预填。
