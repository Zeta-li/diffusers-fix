# Cache-aware regional image editing research

The final development configuration is LOCAL128 + R25: an unmerged FP32 rank-8 student enabled only on local steps, with full refreshes at steps 1, 2, 5, and 8. Inputs are an unmasked source image, prompt and source-side ROI. There is no hard output mask or pixel copying. The implementation targets original Qwen-Image-Edit + its eight-step Lightning adapter.

See reports 15–22 for training, ablations, policy learning, development quality and measured request latency. The latest result is approximately 1.50x model-resident end-to-end speedup; joint semantic success is only 7/16 on the latest development batch, including shared dense-base failures. SUP is optional and RL was not promoted. No final Test64 or native SGLang generation result is claimed.

Run `python unpack_protocols.py` to restore compressed frozen JSON/CSV files. Each restored file is checked against its original SHA256. These files contain repeated token indices and are compressed only for repository size. Complete locally available raw artifacts are in the separately checksummed research snapshot. Models are in the separate LOCAL128_R25 recovery ZIP; restore it into this directory with its `restore.py`. Base model and Lightning versions are pinned in the package.

Entry point: `scripts/run_local_edit.py`. It supports square source images, 1024x1024 output, eight steps, CFG=1 and one request at a time. Peak measured memory is about 69 GiB. Use the frozen request key and explicit noise for exact historical probes. `--export-plan-only` validates the input and exports the backend plan.

Some archived scripts implement abandoned controls and must not be mistaken for the final algorithm. The final executor is selected through `local_edit_runtime.py`; `controlled_mask.py` remains for historical experiments. Native SGLang support is preparatory, not a completed backend.

The GPU was stopped before the newest raw files were downloaded. Report 21 records tool-returned aggregates and the exact evidence gap. Historical manifests can also reference raw files absent from the local archive; consult the thesis delivery audit. Absence of raw files is not repaired by inventing records.
