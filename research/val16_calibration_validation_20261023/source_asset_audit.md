# Val16 源图、掩码与隔离审计

实际执行日期：2026-10-23。范围仅为冻结 Val16 的资产准备与 CPU 检查，不包含模型生成、校准、拟合或 GPU forward。

## 资产与冻结选择

16/16 组源图和原始掩码已取得，另生成 16 张目标画布二值掩码。48 个文件已在本地和 zeta 分别按 SHA256 核对通过；本地与远端 cases、资产清单、采集契约、网络日志、Parquet 布局日志、采集器源码 SHA256 一致。原 Train8 资产和原冻结划分未修改。

原父冻结选择文件 SHA256：`577ff3a6f7bcbfc94ff150c055e335db63adc9f76247f00135138482185ba01a`。

Val16 保留官方 dev、首轮、原始哈希排序，不替补：

`308192, 565269, 392687, 175584, 190360, 178572, 211860, 415314, 50151, 215135, 354126, 25139, 282572, 466845, 55458, 155790`

所有源图为 RGB 500×500，原始掩码为 RGBA 1024×1024，输出画布为 1024×1024、target token 网格为 64×64。沿用 alpha_v2：alpha=0 为明确编辑区域，alpha<255 为本研究纳入抗锯齿边缘的保守 ROI；取 bbox 并扩展 2 个 token。16 组全部通过几何检查，没有空/full ROI 或宽高比不匹配拒绝。

core/target 比例最小为 0.0349121，中位数为 0.2670898，均值为 0.2536011，最大为 0.5458984。该比例是选中 target token 的比例，不能直接解释为实际加速比或质量保证。

## 去重与数据隔离

逐样本与旧 3 个 demo、冻结 Train8 共 11 个源图参照比较，再检查 Val16 内部 120 对，共 **296 对**。核对 raw bytes SHA256、带宽高的 RGB SHA256，以及 64 位 average hash/difference hash。两种感知 hash 距离均≤6且宽高比 log 距离≤0.02 判为近重复；未命中前述规则但任一感知 hash 距离≤10记为灰区。

结果为 0 个 exact/near duplicate 拒绝、0 个灰区。这里只能证明未命中上述检测规则，不能证明任意裁剪、翻转等复杂变换下绝无近重复。未依据生成结果或质量分数重选样本。联系图只展示源图和标注 ROI，未查看目标图或生成输出。

## 实际网络与失败记录

数据只从固定 revision `6c254560879df7851233ea0fd2ae2a62764d0ef2` 的四个官方 dev Parquet shard 读取，访问了 9 个行组。全部 81 次 Parquet GET Range 均为精确 HTTP206；**失败 0 次，HTTP 重试 0 次**。逐次范围、Content-Range、响应体长度与 SHA256 均有记录。

成功 Range 响应体共 **510,427,841 字节，约 510.43 MB / 486.78 MiB**。HTTP 请求耗时之和为 **132.60 秒**，不包含 SSH 建连、SCP 回传、全部 CPU/QC 或模型运行时间。请求开始时间记录范围为北京时间 16:12:07–16:14:19。

最终写出的 16 组原始源图/掩码共 **25,593,251 字节**，16 张派生二值掩码共 **48,846 字节**。网络量大于最终保存量，是因为 Parquet 读取粒度为行组选定列，同组其他行的 source/mask bytes 可能一并传输；只解码、保存并使用冻结的首轮 Val16。不能把最终保存的 25.59 MB 写成实际下载量。

独立审计把每一个实际 GET Range 与 Parquet 元数据给出的 target_img column chunk 区间比较，交叠为 0；所有请求文件均为 `data/dev-*`，没有 Test 图像请求。必要 footer/schema 元数据仍会读取，不等于读取目标图像数据列或目标图像 payload。

最初一次受当前本地沙箱网络限制的 SSH 建连在数据提取前关闭；按已授权网络权限重试后成功。该连接失败未发起数据 GET，不计为上述 HTTP 下载失败或重试，也不计模型请求。

## 独立验收与复用入口

本地与远端均实际运行 `scripts/audit_val16_assets.py`，结果 PASS：48 次文件校验、16 次 bbox/网格/indices 独立重建、296 对重复阈值重算、81 次 HTTP 范围核验全部通过。该审计脚本不导入模型，不运行 GPU。

```bash
python scripts/prepare_val16.py --source-stage ../fewshot_calibration_start_20261023 --old-cases ../lightning8_preflight_20261022/cases.json
python scripts/audit_val16_assets.py --source-stage ../fewshot_calibration_start_20261023
```

远端根目录为 `/workspace/diffusers-fix/research/val16_calibration_validation_20261023`。主要证据为 `cases_val16.json`、`data/asset_preregistration.json`、`data/val16_assets.json`、`data/asset_independent_validation.json`、`data/network_ranges.jsonl`、`data/parquet_layouts.jsonl`、`logs/prepare_val16_stdout.log`、`logs/remote_asset_audit_stdout.log`。

cases SHA256：`d66d413ffa385e96def3392faed5156eba3fa55827e204080ef3923bea4eae25`。

资产清单 SHA256：`1a5d6156210d57f797873daa5f5c185692ff2aa2a06d6787dadd564b689fe4ce`。

本阶段数据准备计账：模型请求 0、DiT forward 0、训练步 0；不与之后正式 Val16 GPU 实验的计账混合。
