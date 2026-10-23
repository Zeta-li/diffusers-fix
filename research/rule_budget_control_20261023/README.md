# H800规则预算控制实验

本目录为2026-10-23已完成实验，使用原版Qwen-Image-Edit及对应8步Lightning。仅改变R2核心外背景的刷新预算，不含Value校正、小样本训练或RL。

- [执行前协议](01_规则预算实验协议.md)：模型合同、比例定义、72请求计划、质量门槛和计时范围。
- [实测结果与结论](02_实测结果与结论.md)：已完成72请求/576次DiT；含逐例数值与非盲图像复核。
- `03_预算数据摘要.md`：分析脚本生成的完整质量/预算/延迟表。
- `masks/budget_plan.json`：整块最老优先策略、逐步整数预算与真实年龄的冻结参照。
- `data/preregistration.json`：协议与执行源码哈希；`data/budget_policy_checks.json`为48组无模型CPU检查。
- `data/runs_primary.jsonl`：只追加的真实请求账本；`outputs/`为全部输出。
- `data/summary_primary.json`、CSV及`figures/primary/`：派生指标与原分辨率对照。
- `data/final_validation_summary.json`、`artifact_manifest.json`：独立完整性检查与文件哈希；检查通过不表示质量通过。

zeta对应目录：`/workspace/diffusers-fix/research/rule_budget_control_20261023`。Python环境为`/workspace/venvs/qwen-edit-pilot/bin/python`，`PYTHONPATH=/workspace/diffusers-fix/src`。

源码中的`run_budget.py`只允许空账本启动，已有账本必须显式`--resume`；完整结果上的resume会跳过已完成请求，不产生新的独立测量。复现新矩阵应使用新目录并登记新attempt，不覆盖本次数据。

预算候选每请求都在计时内运行8步策略选择；原生F与静态R2不添加无用planner开销。CPU块预编码、模型加载、PNG与指标单列。背景预算b、实际target比例、八步平均比例、注意力查询比例和端到端耗时分别记录。

当前判断：执行完整性PASS，所有预算均未达成三例共同质量与时间目标。去毛发B250是局部正结果（约10.73%净节时），衬衫B500约5.08%节时，文字四档仍失败；小样本训练和RL尚未执行。
