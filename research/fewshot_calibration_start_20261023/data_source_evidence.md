# MagicBrush 小样本校准数据来源核验

核验日期：2026-10-23。本文和采集器均为准备工作；本文件写入时尚未执行数据下载、冻结划分、Train8 校准或正式 Test 生成。不能把计划样本数写成已取得的样本数。

## 1. 一手来源、版本与许可

首选官方仓库 `osunlp/MagicBrush`，固定数据 revision：

`6c254560879df7851233ea0fd2ae2a62764d0ef2`

该版本的提交说明为 “Order test split as in the original release”。采集器显式向 Hub 传递这个完整 revision，按该 revision 列出 Parquet 文件并构造下载 URL，不使用可变的 `main` 作为数据版本。[官方数据卡](https://huggingface.co/datasets/osunlp/MagicBrush/blob/main/README.md)、[固定版本提交](https://huggingface.co/datasets/osunlp/MagicBrush/commit/6c254560879df7851233ea0fd2ae2a62764d0ef2)。

当前官方 schema 包含 `img_id`、`turn_index`、`source_img`、`mask_img`、`instruction`、`target_img`。`img_id` 为字符串形式的 COCO ID；第 1 轮的源图是原始真实图像，后续轮的源图可以是上一轮编辑图；白色掩码区域表示编辑区域。旧版本缓存可能使用 `input_img/output_img` 等字段，不应与本版本混用。[官方字段说明](https://huggingface.co/datasets/osunlp/MagicBrush#dataset-structure)。

官方 train/dev/test 分别含 8,807/528/1,053 个编辑轮次，对应 4,512/266/535 个编辑会话。完整下载规模的数据卡值为 **30,197,726,673 字节，约 30.2 GB**；它不代表本采集方案必须下载的字节数，也不等于各 split 的 `num_bytes` 字段。实际压缩 Parquet 总量以采集时固定 revision 的 `RepoFile.size` 求和为准。[官方数据卡和文件列表](https://huggingface.co/datasets/osunlp/MagicBrush/tree/main/data)。

官方 LICENSE 明确数据集和评测代码采用 **CC BY 4.0**，应保留作者、论文、来源和许可署名。仓库中模型的 CreativeML Open RAIL-M 许可属于该仓库模型，不能替代 Qwen 模型自己的许可。正式 Test 不进入参数拟合、校准、阈值调节或策略选择。[官方 LICENSE](https://raw.githubusercontent.com/OSU-NLP-Group/MagicBrush/main/LICENSE)、[官方项目仓库](https://github.com/OSU-NLP-Group/MagicBrush)。

## 2. 文件位置与最小读取方法

本版本包括 51 个 train、4 个 dev、6 个 test Parquet shard。具体路径由 `HfApi.list_repo_tree(..., revision=REVISION, repo_type="dataset", path_in_repo="data")` 返回；排序后保存每个文件的路径、大小、Git blob ID、可用的 LFS SHA256/Xet hash 和固定版本 URL，避免猜测 train/dev 文件名的哈希后缀。[Hub API 官方接口](https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api#huggingface_hub.HfApi.list_repo_tree)。

已核对的路径示例为：

```text
data/train-00000-of-00051-9fd9f23e2b1cb397.parquet
data/dev-00000-of-00004-f147d414270a90e1.parquet
data/test-00000-of-00006.parquet
```

URL 固定为 `https://huggingface.co/datasets/osunlp/MagicBrush/resolve/<revision>/<path>`。当前网页仅给单文件的四舍五入大小，不能把这些显示值相加声称得到了精确 split 字节数。

采集器分两遍读取：

1. 扫描全部 61 个 shard 的 `img_id/turn_index/instruction` 三个文本列，保存逐 shard 的行位置、行组位置和元数据 SHA256。只有全部行数和 shard 数校验通过才冻结划分；中断后复用已完整保存的 shard 元数据。
2. `--extract-train8` 只对冻结 Train8 所在的行组投影读取 `source_img/mask_img` 和定位所需的文本列，保存 8 组原始 bytes。不会投影、解码、保存或用于评分 `target_img`，不会提取 Val16/Test64 图像。官方导出脚本证明 Image 字段可以取原始 PNG bytes；采集器未直接运行该导出脚本，因为它会读取完整 Test 和目标图。[官方导出实现](https://raw.githubusercontent.com/OSU-NLP-Group/MagicBrush/main/evaluation/export_test_set.py)。

Arrow 支持按行组和列读取。读取粒度仍是整行组的选定列，因此网络可能传输同一行组内未入选行的源图/掩码 bytes；只解码和写出冻结 Train8，并不等于网络恰好只传输了 8 张图。[Arrow ParquetFile 官方接口](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetFile.html)。

采集器使用 fsspec 的缓冲文件接口和显式 HTTP Range 取数，要求每次响应为精确 `206`、正确 `Content-Range`、固定对象总长、正确响应体长度；可用 ETag 在同一文件会话内必须一致。先精确读取末尾 8 字节和指定长度 footer，再把已解析的 metadata 交给 Arrow，避免默认 footer 预读碰到邻接图像数据。通过 Parquet 的 page offset 与 compressed size 建立 `target_img` 列 chunk 禁读区间，后续任何请求区间与其重叠都会在发请求前停止。保留实际区间、响应体 SHA256、耗时、成功字节数和失败记录。footer 元数据本身需要读取，不声称完全未读取任何与 target 字段有关的 schema/统计元数据。[fsspec 文件接口](https://filesystem-spec.readthedocs.io/en/latest/api.html)。

Datasets 本身也支持流式 Parquet 列投影和 filters；`Image(decode=False)` 主要避免解码，不能单独证明少下载了数据。本脚本使用显式 row-group locator 便于审计。[Datasets 官方流式读取说明](https://huggingface.co/docs/datasets/stream)。

**不把 datasets-server 当成任意 revision 的冻结下载接口。** 官方 `/rows` 文档只声明 dataset/config/split/offset/length，`/parquet` 文档给出 dataset 参数和转换分支；未查到任意 `revision=` 的公开参数契约。附加未知参数不能证明服务端按该 revision 返回内容；本方案直接读取固定 Hub revision 的原始 Parquet。[官方 rows API](https://huggingface.co/docs/dataset-viewer/rows)、[官方 Parquet API](https://huggingface.co/docs/dataset-viewer/parquet)。

## 3. 冻结划分与泄漏边界

固定 seed 为 `2026102303`。只保留 `turn_index=1`；按规范化 COCO `img_id` 整组处理；跨官方 split 出现的同一 ID 全部排除，不移动到其他 split；某 ID 在同一 split 存在多个首轮记录时排除该组。每个 split 按 `SHA256(seed|revision|official_split|img_id)` 升序排列，取官方 train 的前 32、dev 的前 16、test 的前 64。Train8/16 为同一 Train32 的前缀。排序不读取目标图、生成结果或质量分数。

`data/frozen_selection.json` 保存划分契约、所有选中 ID 和 locator、元数据校验和及有序候选清单。已有冻结文件不得覆盖；脚本不会遇到拒绝项就顺延替换。如果冻结 Train8 存在重复、无效掩码或对齐歧义，状态为阻塞，必须保留原 manifest；改样需显式建立新版本。

Test64 在此阶段仅锁定 ID、指令和 locator；不读取源图/掩码，不运行模型，不接触目标图。由于尚未取得 Test 图像，**不能声称 Train/Val/Test 的感知近重复检查已完成**。当前只完成 ID 组隔离，内容去重范围仅为实际提取的 Train8 与旧 3 个 demo、Train8 内部。

旧 demo 以原 `cases.json` 及文件 SHA256 核对。源图比较使用原始 bytes SHA256、带宽高的 RGB 像素 SHA256、64 位 average hash 和 difference hash。两种感知 hash 距离均不超过 6 且宽高比接近的项拒绝；单 hash 可疑或距离接近的项写为灰区，要求仅根据源图复核，不依据生成质量筛选。该规则只能证明未命中已声明检测器，不能证明不存在任意裁剪、翻转或其他复杂近重复。

## 4. 固定 ROI、几何检查与 Qwen 适用性

Qwen-Image-Edit 使用源图和编辑指令；本数据的源图/指令可以作为其输入，掩码由局部执行器用于固定 ROI，不假设原版 pipeline 自带 masked editing 接口。

数据卡没有充分规定源图与掩码之间所有裁剪/补边关系，且 viewer 的宽度统计并不完全一致。脚本检查原始尺寸、颜色模式、EXIF 和 alpha；非不透明 alpha、非灰度掩码、空掩码、全正掩码、无法验证的宽高比差异会停止，不把未知坐标关系盲目拉伸解释成对齐。

通过几何检查的样本，按现有 Qwen `calculate_dimensions(1024*1024, width/height)` 的相同数学公式把输出宽高取最近 32 的倍数。原始掩码按相同画布最近邻映射；白色正像素取半开 bbox；映射到 VAE×packing 对应的每格 16 输出像素网格后，四周扩展 2 个 target token，并裁剪到画布。ROI 若覆盖全图同样停止，因为本前置校准需要至少一个 inactive token。

**这是人工提供掩码的 bbox 加固定 halo 基线，不是学习到的语义边界，不保证掩码外必然无需更新，也不构成无损编辑承诺。** 后续边界学习和预算策略应与此基线分别验证。

## 5. 运行入口与产物

以下为交给执行端的命令，本地尚未执行：

```bash
python scripts/prepare_magicbrush.py --metadata-only
python scripts/prepare_magicbrush.py --extract-train8 --old-cases ../Lightning8前置实验_20261022/cases.json
```

`--root` 可以指定本阶段输出根目录。HTTP 使用环境中已配置的代理；不在代码中硬编码地址或凭证。

主要产物为 `data/repo_inventory.json`、`data/metadata_shards/`、`data/frozen_selection.json`、`data/network_ranges.jsonl`、`data/parquet_layouts.jsonl`、`data/train8_assets.json`、`data/extraction_status.json`、`cases_train8.json` 和 `inputs/train8/`。只有 8 个冻结源图/掩码全部通过且无待复核灰区时，才设置 `ready_for_calibration=true`。所有状态记录模型调用、DiT forward 和训练步数为 0。

PIE-Bench 可以作为后续外部评测补充。官方 PnPInversion 仓库提供 700 张源图、指令和编辑 mask，但当前下载入口经过表单，尚未从官方材料核实独立数据许可及可立即自动化下载路径，因此本轮不把 PIE-Bench 记为已取得数据。[PIE-Bench 官方发布仓库](https://github.com/cure-lab/PnPInversion)。

## 6. alpha_v2 解码修订：保留 v1 历史，覆盖上述 RGBA 拒绝规则

原始采集执行后，冻结 Train8 的首个样本 `257102` 被 v1 的“非不透明 alpha 一律拒绝”检查阻止。执行端报告源图为 RGB 500×500；原始 mask 为 RGBA 1024×1024，其 RGB 是源图放大画布，左侧柜子对应区域被 alpha=0 的透明洞覆盖，alpha=0 像素为 147,106，另外有少量抗锯齿中间 alpha 值。执行端只检查源图与 mask，没有查看目标图。原始失败日志和原始 bytes 保留；没有更换冻结 ID，也没有重新按生成质量选择样本。

补充一手证据如下：MagicBrush 论文 §3.2 明确标注人员用 DALL·E 2 平台、提示和手绘区域 mask 完成标注；附录 E.1 说明上传源图并在平台继续编辑。[MagicBrush 原论文](https://arxiv.org/html/2306.10012v2#S3.SS2)。OpenAI 官方 DALL·E 2 教程把选区写入 PNG alpha 通道，并明确编辑透明区域；官方 API 类型定义进一步明确完全透明的 alpha=0 区域表示应编辑区域。[官方 DALL·E 2 教程](https://developers.openai.com/cookbook/examples/dalle/how_to_create_dynamic_masks_with_dall-e_and_segment_anything)、[官方 API mask 参数定义](https://github.com/openai/openai-python/blob/main/src/openai/types/image_edit_params.py)。MagicBrush 官方导出脚本直接写出 `mask_img["bytes"]`，不把发布 PNG 转成灰度或反转 alpha；现有公开 `image_eval.py` 排除 mask 文件，并不能提供另一种 alpha 解码规则。[官方原始 bytes 导出代码](https://raw.githubusercontent.com/OSU-NLP-Group/MagicBrush/main/evaluation/export_test_set.py)、[官方评测实现](https://raw.githubusercontent.com/OSU-NLP-Group/MagicBrush/main/evaluation/image_eval.py)。

因此，结合官方工作流、DALL·E 2 的明确极性，以及本次原始 mask 的源画布与透明洞观察，v2 对这种 RGBA mask **忽略 RGB 画布的亮度，用 alpha 解码**：`alpha==0` 是官方明确的待编辑区域；`alpha<255` 作为本研究的保守 ROI 支持区，纳入半透明抗锯齿边缘。后一条是本研究固定的边缘处理规则，不能写成官方规定的阈值。每张图记录 alpha 直方图、完全透明/中间值/保守支持区像素数和两种 bbox。完全不透明的灰度白色正区域 mask 继续兼容；不透明彩色源画布、没有 alpha=0 的不明半透明 mask、空/全图 mask、几何不匹配仍拒绝。

本节覆盖第 4 节和 v1 合约中对 RGBA 非不透明 alpha 的拒绝与白色亮度解释；其他几何、bbox、2-token halo、分组、排序和泄漏规则不变。原 `prepare_magicbrush.py`、原 `data/frozen_selection.json`、冻结协议和原失败记录不修改。新 `prepare_magicbrush_v2.py` 在 `preprocess_alpha_v2/` 保存 decoder amendment 和 effective selection；它们绑定原 selection SHA256，保留相同 ID、全部元数据、rank、locator 和 split，只修订几何解码契约。导出根目录的 `cases_train8.json` 与 `data/train8_assets.json` 使用此前缀下的新资产，并记录 parent/effective selection SHA256 与解码侧车来源。

CPU 几何验收入口为 `python scripts/check_magicbrush_geometry_v2.py`；v2 提取入口为 `python scripts/prepare_magicbrush_v2.py --extract-train8 --old-cases ../Lightning8前置实验_20261022/cases.json`。本节写入时尚未执行这两个入口，也尚未完成 Train8 校准。

## 执行状态补记（2026-10-23）

以上准备方案现已执行：61个元数据分片通过，32/16/64 ID池冻结，原Train8全部通过alpha_v2资产QC，未替换样本；112次真实DiT采集和A/B统计拟合已完成。正式执行结果以[启动实测结果](02_启动实测结果.md)及JSON账本为准。开头的“尚未执行”描述的是本文首次写入时的历史状态，不能代替本次完成记录。
