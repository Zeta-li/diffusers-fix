# 独立 Val16 校准验证

本目录对应 2026-10-23 在 zeta 单卡 H800 上实际执行的 F/R2/PRIOR/A/B 独立验证。底座为原版 Qwen-Image-Edit + 对应 8 步 Lightning LoRA；Train8 校准冻结，Val16 不拟合，Test 不生成。已完成320请求/2560次DiT，0推理失败，独立审计PASS。结论为Revise：四候选净节时约11%–12%，但均仅7/16例通过固定数值质量门槛，且有语义/边界失败。H800已空闲。

- [实测报告与结论](02_独立验证实测报告.md)
- [生成前冻结协议](01_独立验证协议.md)
- [资产 QC](01_Val16资产QC.md) 与 [独立源资产审计](source_asset_audit.md)
- [完整逐例质量和延迟摘要](03_Val16数据摘要.md)
- [前8例助手视觉复核](05_前8例助手视觉复核.md) / [后8例](06_后8例助手视觉复核.md) / [完整80条证据](data/assistant_visual_review_all16.json)
- [后续条件式因果实验（未执行）](07_结论边界与后续因果实验建议.md)
- [五组原始请求账本](data/runs_primary.jsonl)
- [质量 CSV](data/quality_val16.csv) 与 [计时 CSV](data/timings_val16.csv)
- [完整统计 JSON](data/summary_val16.json) 与 [独立计账核验](data/independent_validation.json)
- [预注册矩阵与源码 SHA](data/preregistration.json)
- [冻结评价区域](masks/evaluation_regions.json)

远端目录：`/workspace/diffusers-fix/research/val16_calibration_validation_20261023`。模型运行使用 `/workspace/venvs/qwen-edit-pilot/bin/python`，`PYTHONPATH=/workspace/diffusers-fix/src`。文件中的绝对远端路径是采集时的真实路径；本地阅读图表、源图和 CSV 不需要连接远端。

执行器与拟合参数依赖上一阶段不可变目录 `../fewshot_calibration_start_20261023`（本地对应 `../小样本校准启动_20261023`），时间表和通用 Recorder 来自 `../lightning8_preflight_20261022`。SHA 在预注册及环境记录中固定。模型权重不包含在本目录。上一阶段记录的“Val pending”是当时快照，本轮状态以本目录为准。

## 运行和复核入口

以下命令在 zeta 的仓库根目录执行。主运行已开始或完成时保留现有账本；只在确有中断需要续跑时给 `run_val16.py --resume`，不要删除原始记录重新计账。

```bash
PYTHONPATH=/workspace/diffusers-fix/src /workspace/venvs/qwen-edit-pilot/bin/python research/val16_calibration_validation_20261023/scripts/run_val16.py

/workspace/venvs/qwen-edit-pilot/bin/python research/val16_calibration_validation_20261023/scripts/analyze_val16.py --device cuda

/workspace/venvs/qwen-edit-pilot/bin/python research/val16_calibration_validation_20261023/scripts/validate_val16.py --train-root /workspace/diffusers-fix/research/fewshot_calibration_start_20261023
```

分析在主生成进程退出后执行，避免 LPIPS 与主推理争用 GPU。源资产独立审计不调用模型；细节见 `source_asset_audit.md`。

## 数值解释

320 个请求由 16×5×(1 预热+3 测量)组成；质量仅评价 repeat0 的 80 个输出，独立源图仍为 16。运行时无影子教师调用、无 Val 校准或梯度步。净延迟以相同源图、相同 repeat 的同期 F 为分母，包含校正和缓存管理成本。LPIPS≤0.02 与 extraBG≤1/255 为固定逐例数值门槛；不代表完成指令、无损或通过正式 Test。

F 为 8 步计算参考，语义是否正确仍须单独看图。视觉复核由助手非盲查看，不是真人人工评测或独立盲评。不得把计时重复、数值门槛或助手标签写成正式人工准确率。
