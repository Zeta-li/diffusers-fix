# LOCAL128 + R25 可恢复模型包

本包保存本研究实际训练得到的权重，不包含可重新下载的原版 Qwen 主干。模型文件已与 H800 实验冻结 SHA256 一致核验，不依赖 H800 继续运行。

## 包含什么

|文件|用途|
|---|---|
|LOCAL128_student.pt|最终缓存感知学生，rank8、4,423,680 参数、累计140个拟合来源、153次更新|
|SUP_refresh_head.pt|18参数监督刷新头，作为可选研究策略|
|RL_bandit_head.pt|真实128episode/16更新的bandit头；保留消融，不作为默认策略|
|explicit_noise.pt|历史评估噪声，用于精确复现|
|manifest.json|文件大小、SHA256、原路径及基座版本|
|scripts/|实际研究代码快照，含局部执行、训练、策略与CLI|
|environment.json|H800已验证环境版本|
|restore.py|校验与恢复到代码仓库研究目录|

默认使用 LOCAL128 + R25：完整步骤为第1、2、5、8步，其他步骤按局部预算执行。新增学生只在局部步开启；Lightning始终开启。不要把学生永久合并到BF16主干，也不要直接用普通PEFT加载器猜测本研究checkpoint格式。

## 校验与恢复

```bash
python restore.py --verify-only
python restore.py --target /path/to/diffusers-fix/research/manual_mask_local_20261028
```

恢复脚本仅复制包内冻结文件；同名不同内容会报错，避免覆盖其他实验。安装了PyTorch的环境可另外使用 `--torch-check` 检查最终学生包含144张量及4,423,680参数。

代码基座为 `Zeta-li/diffusers-fix`，起点 commit `4222cd5955b8033842748a7d676a5d180ab4c22d`。最终新增代码的提交与推送结果以交付目录的 Git 收据为准。不要只保留模型而丢失本包代码：局部缓存、完整步开关和时间步合同共同构成算法。

原版主干来自 [Qwen/Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit)，固定 revision `ac7f9318f633fc4b5778c59367c8128225f1e3de`；8步适配来自 [lightx2v/Qwen-Image-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Lightning)，固定 revision `d468c534b360e0754c6bf9e38ed49e1e09cd2e2a`，文件名及哈希见manifest。它们不属于本研究重新训练的权重，不重复放入本包；完整原始文件清单在 `base_model_receipt.json`，可在新GPU下载后校验。

运行代码默认寻找 `/workspace/model-weights/Qwen-Image-Edit` 和 `/workspace/model-weights/Qwen-Image-Lightning`。迁移时可以按这些路径放置模型，也可通过单独配置修改 `scripts/run_s1.py` 的BASE/LORA常量；修改后不能声称代码哈希与原实验完全相同。研究代码基于Python3.12/PyTorch2.9.1+cu129，精确包版本见environment；新硬件/内核可能造成浮点差异，应先做完整F与局部R25的小样本复核。

支持正方形原图、1024输出、8步、CFG1、BF16、单请求。完整推理约69GiB峰值，不保证在24GB单卡上直接运行。原图+prompt+ROI是公开输入；没有额外mask条件图或输出像素粘贴。使用方法和已验证范围详见论文及报告22。

## 保存范围

这是本地恢复包，尚未发布至ModelScope或其他模型社区。语义成功率、独立测试和SGLang状态以论文实验表为准；不能把模型包的完整性校验当作所有任务质量达标。最新远端原始轨迹的同步状态另见交付总表。
