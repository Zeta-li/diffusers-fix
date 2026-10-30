# Thesis working manuscript and evidence

Read [the full manuscript](论文_缓存感知低秩校准与区域预算刷新.md), [experiment catalog](实验全目录与复现方法.md), [eight-month plan](八个月研究计划与当前完成情况.md), and [recovery/data-gap report](交付清单_恢复步骤与数据缺口.md). The implementation is in `../manual_mask_local_20261028`.

The separate recovery ZIP contains the actual trained weights; the 13.10GB local snapshot contains available historical raw evidence. Neither large archive is committed here. Their exact hashes and receipts are included. `model_recovery` holds manifests and the restoration program; run it from the complete unzipped model package, where weight files exist.

The latest remote raw trajectories and per-request timing files were not downloaded before the GPU stopped. The manuscript explicitly identifies returned aggregates, locally preserved blind review images and missing evidence. No final Test64, SGLang native result, lossless claim, or positive RL claim is made.

Historical evidence paths in the experiment catalog refer to the original workspace snapshot. Reused source IDs across protocols are documented and must not be counted as independent samples.
