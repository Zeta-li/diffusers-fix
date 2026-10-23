# Primary-source evidence for thesis report

Verified 2026-10-22 (Asia/Shanghai). These are literature statements, not claims about our experiment completion. Do not describe planned few-shot fitting as already evaluated. No experiment directories were touched.

## Citation records

1. Pengtao Chen, Xianfang Zeng, Maosen Zhao, Mingzhu Shen, Peng Ye, Bangyin Xiang, Zhibo Wang, Wei Cheng, Gang Yu, Tao Chen. **RegionE: Adaptive Region-Aware Generation for Efficient Image Editing**. arXiv:2510.25590, submitted 2025-10-29; ICLR 2026.
   - Abstract/metadata: https://arxiv.org/abs/2510.25590
   - Full text: https://arxiv.org/html/2510.25590v1
   - Official repository: https://github.com/Peyton-Chen/RegionE
   - ICLR proceedings version found in primary search: https://openreview.net/pdf/83afcdd5b5158229284969c800eea53e092141e6.pdf

2. Zhibin Qin, Zhenxiong Tan, Zeqing Wang, Songhua Liu, Xinchao Wang. **SpotEdit: Selective Region Editing in Diffusion Transformers**. arXiv:2512.22323, submitted 2025-12-26. Use arXiv 2025 citation if conference metadata is not separately retained/verified.
   - Abstract/metadata: https://arxiv.org/abs/2512.22323
   - Full text: https://arxiv.org/html/2512.22323v1
   - Official repository: https://github.com/Biangbiang0321/SpotEdit
   - Official project: https://biangbiang0321.github.io/SpotEdit.github.io/

3. Jinshan Liu, Haoran Qin, Xiaobing Tu, Jiacheng Liu, Jiahui Hu, Zhengan Yan, Yukun Xie, Kerui Shen, Jinkui Ren, Yuqi Lin, Xiantao Zhang, Linfeng Zhang. **LinCa: Accelerating Diffusion Models via Learnable Decomposed Feature Caching**. arXiv:2608.17973, submitted 2026-08-18. The arXiv record explicitly says accepted to ECCV 2026.
   - Abstract/metadata: https://arxiv.org/abs/2608.17973
   - Full text: https://arxiv.org/html/2608.17973v1
   - Official repository: https://github.com/QHR69/LinCa

4. Chenfei Wu, Jiahao Li, Jingren Zhou, et al. **Qwen-Image Technical Report**. arXiv:2508.02324, submitted 2025-08-04.
   - Metadata/report: https://arxiv.org/abs/2508.02324
   - Official original editing model card: https://huggingface.co/Qwen/Qwen-Image-Edit
   - Official model repository: https://github.com/QwenLM/Qwen-Image

5. LightX2V / ModelTC. **Qwen-Image-Edit-Lightning-8steps-V1.0**. Released 2025-08-23 per developer repository. This is a developer-released distilled LoRA adapter, not a separate Qwen Team base model or a paper with a verified arXiv ID.
   - Developer model card (shared T2I/edit repository): https://huggingface.co/lightx2v/Qwen-Image-Lightning
   - Exact BF16 editing adapter record: https://huggingface.co/lightx2v/Qwen-Image-Lightning/blob/main/Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors
   - Developer repository and original editing example: https://github.com/ModelTC/LightX2V-Qwen-Image-Lightning

## RegionE: evidence and differentiation

Full text Section 4, equations 5-9: early clean-latent extrapolation and cosine comparison to the condition image determine edited/unmodified regions; morphological opening/closing improves mask continuity. The local DiT executes prompt and edited target tokens while global context comes from cached background/reference K and V. Background is advanced by a one-step velocity estimate; periodic full-image passes refresh caches, and final full denoising steps smooth seams.

Although the authors call it training-free, equation 7/8 text explicitly says the timestep correction gamma_t is fitted on a randomly sampled dataset. Therefore distinguish no backbone training from no data calibration.

Section 5/Table 1: 28 steps, single H800, single-image runs; 6 stabilization steps, forced refresh at step 16, 2 smoothing steps. End-to-end speedups: Step1X-Edit 2.572x; FLUX.1-Kontext 2.409x; original Qwen-Image-Edit 2.059x (32.125s to 15.604s). Accelerated-versus-vanilla PSNR is finite (30.520/32.133/31.115 dB); do not call the whole output mathematically lossless.

Overlap with our work is strong: local target-query execution and global background KV reuse already exist. A boundary expansion alone is not sufficient differentiation because RegionE already uses morphology and full-image seam smoothing. A possible implementation distinction is our explicit influence halo, leaving background cache storage unchanged and applying a read-time Value-only correction, rather than RegionE's one-step background velocity plus cached K/V and global refresh. This is a design comparison, not proof of novelty.

## SpotEdit: evidence and differentiation

Sections 4.1/4.2, equations 4-10: a reconstructed clean latent is compared to condition image Y through VAE-decoder perceptual features. Only regenerated target tokens plus prompt tokens form active queries. The keys/values include recomputed active tokens plus cached background and condition-image tokens.

SpotFusion interpolates BOTH background K and V toward reference-image K/V using alpha(t)=cos^2(pi*t/2); equations 8/9 are explicit. It is not simply static background caching, and it does not train a correction network. It also periodically resets caches. Current official repository adds a default velocity reuse mode, v=(x_t-x0_src)/sigma, and an alternative final-overwrite mode. Distinguish paper experiments from current repository behavior in a reproduction report.

Tables 1/4: Kontext speedups are 1.67x (ImgEdit partial-edit subsets) and 1.95x (PIE-Bench++); Qwen-Image-Edit 1.59x/1.72x. Experimental environment: one H200, 50 steps, 1024^2. Removing reset raises speedup 1.95x to 2.25x but decreases PSNR 18.73 to 17.10 dB (Table 3); this supports a quality/error-control motivation.

Our value-read correction differs in correction target (Value only versus both K/V), state semantics (unchanged stored background cache versus iterative interpolation), and potentially data fitting (planned few-shot fit versus fixed analytic schedule). The report must not state that context repair or partial attention is new. Need controlled K-only/V-only/KV correction and with/without writeback ablations to substantiate any benefit.

## LinCa: evidence and differentiation

Sections 3.1/3.2, equations 1-4, and Appendix 0.B: LinCa caches the cumulative residual at the diffusion network's final layer. A trained invertible channel mapping decomposes it into subspaces with different continuity; each subspace uses differentiated polynomial/Hermite prediction orders, then inverse mapping reconstructs predicted features. Models and timestep segments have separate trained mappings. After warmup it executes one full backbone computation every N steps and replaces the intervening backbone passes with predictions from history.

Section 3.2 and Appendix 0.A.5 explicitly say 100-200 pre-generated feature samples, approximately one hour on one 12GB GPU, without loading diffusion model weights during predictor training. Teacher feature extraction still requires the backbone. This directly overlaps a generic claim of few-shot learned caching.

Qwen-Image-Edit is evaluated, not only text-to-image: Table 4 on GEdit-CN/EN, H20 (Appendix 0.A.4), base 50-step latency 284.51s; N=7 is 60.02s / 4.74x actual latency speedup, 5.52x FLOPs reduction; N=10 is 49.05s / 5.80x latency, 7.08x FLOPs. Do not report 7.08x as measured wall-clock acceleration. N=7 overall scores 7.45/7.56 versus baseline 7.41/7.54; N=10 7.27/7.40.

The invertibility guarantees no information loss from mapping/inverse mapping itself; prediction error remains (Appendix 0.C.2). It is not proof of identical edited images. Inference already uses historical cache, so causal inference is not by itself a differentiator.

Our proposed distinction should be local background attention-value correction conditioned on boundary/space/time, with exact active-token execution; LinCa learns whole residual feature-space temporal decomposition and can skip the entire backbone on prediction steps. Small calibration data alone is not novel; a future few-shot module must be contrasted with LinCa and a simple fitted analytic schedule on held-out images/instructions.

## Qwen and Lightning baseline record

Qwen-Image-Edit original model card: built on 20B Qwen-Image; dual input through Qwen2.5-VL and VAE encoder; official example uses 50 inference steps, true_cfg_scale=4.0, negative_prompt=' '. This is the original model, not -2509/-2511.

Developer repo original editing example: Qwen/Qwen-Image-Edit + Qwen-Image-Edit-Lightning-8steps-V1.0.safetensors, 8 steps, cfg=1.0. Separate base edit command uses 50 steps/cfg=4.0. Distillation changes model behavior/schedule, so comparisons must separate (a) spatial acceleration relative to the same 8-step Lightning baseline and (b) total system acceleration relative to original 50-step model. A 50/8 step ratio is not measured latency speedup.

The developer model card currently emphasizes T2I and newer 2512 online deployment; cite the exact editing filename and developer editing example, not its default text-to-image inference block. Exact BF16 artifact is about 850 MB; do not confuse it with original FP32 or -2509/-2511 adapters.

## Suggested defensible report wording

Existing work has established region-selective token execution, global KV context reuse, timestep-dependent context fusion, and few-sample learned feature prediction. Our present work studies a narrower intervention: explicit expansion of the computation boundary together with read-time correction of cached background Values while preserving the stored cache. The future training component will estimate this correction from a small calibration set under a causal inference constraint. Whether this combination improves the quality-latency frontier remains an experimental question, requiring held-out generalization and ablations rather than an assertion of first-of-its-kind novelty.
