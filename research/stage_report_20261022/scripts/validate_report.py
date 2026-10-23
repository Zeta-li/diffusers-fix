import argparse
import hashlib
import json
import re
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
REPORT = "局部图像编辑加速研究_阶段性总结报告.md"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    manifest_path = ROOT / "data/artifact_manifest.json"
    if args.manifest_only:
        manifest = read_json(manifest_path)
        for name, item in manifest.items():
            path = ROOT / name
            assert path.is_file() and sha(path) == item["sha256"], name
        print(json.dumps(dict(status="pass", manifest_files=len(manifest)), ensure_ascii=False))
        return
    report = ROOT / REPORT
    text = report.read_text()
    local_links = []
    for value in re.findall(r"\]\(([^\)]+)\)", text):
        if not value.startswith(("http://", "https://", "#")):
            assert (ROOT / value).is_file(), value
            local_links.append(value)
    assert text.count("$$") % 2 == 0
    assert text.count("```") % 2 == 0
    assert text.count(r"\(") == text.count(r"\)")
    tags = re.findall(r"\\tag\{(\d+)\}", text)
    assert tags == [str(i) for i in range(1, 25)]
    assert text.count("```mermaid") == 2 and text.count("```text") == 3
    snapshot = read_json(ROOT / "data/evidence_snapshot.json")
    for relative, expected in snapshot["source_files"].items():
        assert sha(ROOT.parent / relative) == expected, relative
    for dest, source in snapshot["copied_figures"].items():
        assert sha(ROOT / "assets" / dest) == sha(ROOT.parent / source), dest
    rows = snapshot["quality_timing_rows"]
    expected_rows = []
    for phase, configs, label in [("rotation", ["R2", "ROT"], "P1"), ("value", ["SHIFT", "AFFINE"], "P2"), ("channel", ["CHANNEL"], "P3")]:
        for config in configs:
            for row in rows:
                if row["experiment"] == phase and row["config"] == config:
                    expected_rows.append(f"| {label} | {row['case_id']} | {config} | {row['e2e_median_seconds']:.4f} | {(1-row['paired_geometric_latency_ratio_to_F'])*100:.2f}% | {row['lpips_mapmean_to_F']:.6f} | {row['band2_mae_to_F']:.6f} | {row['extra_background_mae_over_F']:.6f} | {'通过' if row['predefined_quality_gate_pass'] else '未过'} |")
    assert len(expected_rows) == 15 and all(line in text for line in expected_rows)
    audit = read_json(ROOT / "data/stage_audit.json")
    assert sum(s["successful_complete_requests"] for s in audit["stages"]) == 347
    assert sum(s["real_dit_forwards"] for s in audit["stages"]) == 3432
    assert sum(s["extra_dense_oracle_forwards"] for s in audit["stages"]) == 72
    assert sum(s.get("tiny_model_forward_count") or 0 for s in audit["stages"]) == 556
    assert (date(2027, 6, 22) - date(2026, 10, 22)).days == 243
    result = dict(status="pass", scope="Report/source consistency and artifact integrity, not algorithm quality acceptance",
                  report_sha256=sha(report), numbered_equations=24, mermaid_flows=2, algorithm_pseudocode_blocks=3,
                  embedded_figures=text.count("!["), generated_result_rows_verified=15,
                  source_file_hashes_verified=len(snapshot["source_files"]), copied_figures_verified=len(snapshot["copied_figures"]),
                  local_links_verified=len(local_links), successful_requests=347, real_dit_forwards=3432,
                  oracle_included_in_dit_total=72, new_model_experiments_in_initial_report_task=0)
    budget_evidence_path = ROOT / "data/budget_evidence_20261023.json"
    if budget_evidence_path.exists():
        budget = read_json(budget_evidence_path)
        for relative, expected in budget["source_files"].items():
            assert sha(ROOT.parent / relative) == expected, relative
        validation = budget["validation"]
        assert validation["status"].lower() == "pass"
        assert budget["successful_requests"] == 72 and budget["real_dit_forwards"] == 576
        assert len(budget["summary"]["quality_rows"]) == 18
        labels = {"blocks_text": "文字", "shirt_color": "衬衫", "remove_hair": "去毛发"}
        budget_rows = []
        for row in budget["summary"]["quality_rows"]:
            if row["config"] != "F":
                budget_rows.append(f"| {labels[row['case_id']]} | {row['config']} | {row['e2e_median_seconds']:.5f} | {(1-row['paired_geometric_latency_ratio_to_F'])*100:.2f}% | {row['lpips_mapmean_to_F']:.8f} | {row['extra_background_mae_over_F']:+.8f} | {'通过' if row['predefined_quality_gate_pass'] else '未通过'} |")
        assert len(budget_rows) == 15 and all(line in text for line in budget_rows)
        assert sha(ROOT / "assets/budget_quality_latency.png") == sha(ROOT.parent / "规则预算控制实验_20261023/figures/primary/budget_quality_latency.png")
        assert text.count("![") == 6
        result.update(budget_source_hashes_verified=len(budget["source_files"]),
            budget_generated_result_rows_verified=15, budget_copied_figures_verified=1,
            new_rule_budget_successful_requests=72, new_rule_budget_dit_forwards=576,
            cumulative_successful_requests=419, cumulative_real_dit_forwards=4008,
            oracle_included_in_cumulative_total=72, new_training_images_used=0)
    (ROOT / "data/report_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    manifest = {str(p.relative_to(ROOT)): dict(bytes=p.stat().st_size, sha256=sha(p))
                for p in sorted(ROOT.rglob("*")) if p.is_file() and p != manifest_path and "__pycache__" not in p.parts}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
