# Qwen-Image-Edit 局部执行与小样本校准实验

截至 2026-10-23，本目录保存基于 Diffusers 的研究原型与实测记录。执行器通过包装 Qwen transformer forward 接入；本次归档没有修改 `src/diffusers/` 的公共 API。实验基于上游提交 `73faca368c0da77645ea58836aa8780d2d8c5353`，实际运行时的源码 SHA 保留在各阶段预注册和环境记录中。

## 当前结论

独立 Val16 已完成 320 次请求、2560 次真实 DiT，0 推理失败。模型为原版 Qwen-Image-Edit 加 8 步 Lightning V1.0 LoRA，BF16、CFG=1、1024²、batch=1；硬件为用户确认的单卡 H800。

| 配置 | 中位耗时（秒） | 同期 F 配对净节时 | 均值 LPIPS/F | 数值质量门槛通过 |
|---|---:|---:|---:|---:|
| F：全量 8 步 | 6.6755 | — | 0 | 自身参照 |
| R2：固定扩边局部执行 | 5.8850 | 12.18% | 0.082066 | 7/16 |
| PRIOR：无样本解析校正 | 5.9378 | 11.17% | 0.071204 | 7/16 |
| A：Train8 统计校正 | 5.9428 | 11.21% | 0.068800 | 7/16 |
| B：Train8 Value 仿射拟合 | 5.9492 | 11.01% | 0.078287 | 7/16 |

耗时列是源图中位数的中位数；节时使用同源图、同重复的 F 配对后按源图等权聚合，不能直接用两列中位数相除。数值门槛为 LPIPS/F≤0.02 且额外远背景 MAE≤1/255，不能替代编辑语义成功率。猫闭眼、池塘接缝、手腕截断和斑马结构断裂均保留在报告中。

判定为 **Revise**：执行可行且有净节时，当前逐通道 Value 校准没有建立可靠的小样本增量。尚未完成主模型梯度训练、强化学习、自动选区或 SGLang 移植。

## 代码与报告入口

| 目录 | 内容 |
|---|---|
| `lightning8_preflight_20261022/` | 原版 8 步模型装载、输入、计时与基线 |
| `local_executor_boundary_20261022/` | 局部目标执行、固定 halo、微型正确性检查 |
| `cache_error_attribution_20261022/` | 缓存误差受控诊断 |
| `rotation_context_optimization_20261022/` | 背景轮转和时间校正 |
| `channel_context_optimization_20261022/` | 解析 Value 通道校正 |
| `rule_budget_control_20261023/` | 规则预算策略与实测 |
| `fewshot_calibration_start_20261023/` | 冻结划分、Train8 矩采集、A/B 拟合与检查 |
| `val16_calibration_validation_20261023/` | 五配置独立验证、配对延迟、质量与失败记录 |
| `stage_report_20261022/` | 之前各阶段的论文格式总结与历史快照 |

当前完整结论见[Val16 实测报告](val16_calibration_validation_20261023/02_独立验证实测报告.md)、[数据摘要](val16_calibration_validation_20261023/03_Val16数据摘要.md)和[条件式后续实验](val16_calibration_validation_20261023/07_结论边界与后续因果实验建议.md)。历史文档中的“未提交”和旧进度描述保留其编写时状态；请以本入口和 Val16 最终报告判断当前研究进展。

## Git 归档范围

Git 保存源码、Markdown、配置、CSV/JSON/JSONL 结果和上述实测报告使用的六张关键图。模型权重、原始图像、其余生成图、校准 NPZ、latent/PT/NPY、运行日志和 Chrome trace 保留在 zeta 的完整实验目录，不写入 Git。旧报告中引用的完整图集与二进制文件须在完整实验目录中查看。

`data/artifact_manifest.json` 等文件描述的是当时的完整实验包，并非 Git 文件清单。缺少被排除资产的全新 clone 不能单独运行完整包校验器或直接生成。各阶段目录布局和脚本 SHA 保持原样，避免归档时格式化改变预注册证据。

完整 Val16 原始包为 459 文件、388832746 字节；manifest SHA256 为 `43b63dab01268ca6d778fccef414a22676bf20a143d9d9b09282ae5e718ddf2d`。冻结 Train8 校准 NPZ SHA256 为 `6df60e49ca8d618c5458063a80ce5bf3a5b02b0c28c7e618b04b76733964518f`。二者已在本地与 zeta 核验。

## 环境与检查

实测环境为 Python 3.12.3、PyTorch 2.9.1+cu129、Transformers 5.3.0、PEFT 0.21.2。运行使用 `PYTHONPATH` 指向本仓库源码，Diffusers 以冻结提交 SHA 为准；环境 JSON 中 `importlib.metadata` 记录的安装包版本为 0.37.0，不能用它代替实际导入源码的版本。精确环境以每阶段 `data/environment*.json` 和源码 SHA 为准。模型路径由基线脚本固定为 `/workspace/model-weights/` 下的本地快照；迁移环境应另建版本并记录路径变更。

从仓库根目录设置 `PYTHONPATH="$PWD/src"`，并使用具有上述依赖的 Python。以下检查不装载 20B 权重、不使用 H800：

```bash
PYTHONPATH="$PWD/src" python research/fewshot_calibration_start_20261023/scripts/check_fitted_correction.py --output /tmp/fitted-correction-check.json
```

在 zeta 完整资产目录中，可重做只读账本核验，并把新检查结果写入独立位置，避免改写冻结报告：

```bash
python research/val16_calibration_validation_20261023/scripts/validate_val16.py --output /tmp/val16-accounting-check.json
```

历史生成脚本写入各自阶段目录；不要在冻结证据目录启动新的实验或覆盖旧输出。复现实验需要先准备源资产、模型与校准文件，并在独立的新版本目录执行。

## 下一步

先执行 C1 同状态 NONE/B0/oracle V/oracle KV 诊断，名义 256 次 DiT，判断 Value 校正是否存在可利用空间。若支持，再做一次 on-policy 补采与 B1 拟合（C2）；同时在任何新探针输出生成前冻结 C3 的结构闭合与等 token 普通扩边区域。C2/C3 是条件式计划，尚未执行。

当前 Val16 已被查看，后续方法需要新的预登记留出验证；Test64 继续封存。只有结构/校正策略有稳定质量收益后，再考虑监督动作选择、上下文 bandit 或强化学习，以及 SGLang 端到端实现。
