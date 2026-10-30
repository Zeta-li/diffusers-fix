# C1 因果诊断、结构扩边与闭环补采

2026-10-23 已按 C1 → C3 → C2 完成 H800 实验：78 请求、864 次真实 DiT、0 推理失败；另计 220 次 CPU tiny forward。C1 发现 Value 校正空间，但结构扩边及单轮 B1 没有稳定终图增量，策略学习、RL 与 SGLang 暂不启动。

- **[阶段总结与后续计划](04_阶段总结与后续计划.md)**：主要结论、公式、流程、逐例数据、失败分析和下一阶段门槛。
- [实验协议与完成进度](01_实验协议与计划进度.md)。
- [C1 因果诊断详报](02_C1因果诊断结果.md)。
- [C3 全八例目视记录](03_C3目视检查记录.md)。
- [C2 全八例目视记录](05_C2目视检查记录.md)。

`data/` 保存预登记、环境、调用账、质量 CSV、哈希、B1 系数与审计；`moments_onpolicy/` 保存每源图补采矩；`inputs/`、`masks/`、`latents/`、`outputs/` 保存复现实物；`figures/` 包含所有对照图及科学图表；`scripts/` 保存实现和核验程序。

脚本在 zeta 的 `/workspace/diffusers-fix/research/causal_structure_pilot_20261023` 执行，依赖同仓库中冻结的 `fewshot_calibration_start_20261023` 与 `local_executor_boundary_20261022` 实现。不是独立安装包。生成脚本拒绝覆盖已有账本/拟合文件，复现实验应复制成新的阶段目录，明确计入新调用。

真实模型检查与指标均已完成；完整性清单见 `data/artifact_manifest.json`，远端一致性回执见 `data/bundle_verification.json`。本轮没有新 commit/push。
