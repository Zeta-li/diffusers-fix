import collections
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/latency_temporal_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(tensor):
    value = tensor.detach().contiguous().cpu()
    return hashlib.sha256(f"{value.dtype}:{tuple(value.shape)}:".encode() + value.view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    torch.set_num_threads(8)
    spec = json.loads((RUN / "preregistration.json").read_text())
    assert sha(RUN / "preregistration.json") == sha(ROOT / "data/latency_temporal_v1_preregistration.json")
    for name, expected in spec["code_sha256"].items():
        assert sha(RUN / "code_snapshot" / name) == expected == sha(ROOT / "scripts" / name)
    status = json.loads((RUN / "status.json").read_text())
    assert status["phase"] == "generation_complete_analysis_pending"
    rows = [json.loads(line) for line in (RUN / "runs.jsonl").read_text().splitlines()]
    assert len(rows) == 82
    noise = torch.load(ROOT / "runs/s1_v2/explicit_noise.pt", weights_only=True, map_location="cpu")
    cases = {c["case_id"]:c for c in spec["cases"]}
    euler_steps, prefix_pairs, same_budget_sources, archived_pairs = 0, 0, 0, 0
    repeated = {}
    temporal_prefix = {}
    for i,(job,row) in enumerate(zip(spec["jobs"], rows)):
        for key,value in job.items():
            assert row[key] == value
        assert row["job_index"] == i and row["completed_dit"] == row["invoked_dit"] == 8*(i+1)
        assert row["successful_requests"] == i+1
        directory = RUN / job["phase"] / job["case_id"]
        stem = job["arm"]+"_r"+str(job["repeat"])
        metadata = json.loads((directory/(stem+".json")).read_text())
        raw = torch.load(directory/(stem+".pt"), weights_only=True, map_location="cpu")
        picture = Image.open(directory/(stem+".png")).convert("RGB")
        assert digest(raw["final"]) == row["final_tensor_sha256"]
        assert hashlib.sha256(picture.tobytes()).hexdigest() == row["raw_RGB_sha256"]
        assert row["timing_eligible"] == (job["phase"] == "latency")
        assert len(row["component_cuda_ms"]["dit"]) == 8
        assert row["quality_approved"] is False
        if job["phase"] == "latency":
            assert not raw["states"] and not raw["velocities"] and not job.get("profile",False)
        else:
            assert torch.equal(raw["input0"], noise)
            previous = noise
            assert len(raw["states"]) == len(raw["velocities"]) == 8
            for step,(state,velocity) in enumerate(zip(raw["states"],raw["velocities"])):
                delta = raw["sigmas"][step+1]-raw["sigmas"][step]
                expected = (previous.float()+delta*velocity).to(state.dtype)
                assert torch.equal(state,expected)
                previous = state
                euler_steps += 1
        normalized_arm = "FIX32" if job["arm"] == "EARLY" else job["arm"]
        key = (job["case_id"], normalized_arm)
        if key in repeated:
            assert repeated[key] == (row["final_tensor_sha256"], row["raw_RGB_sha256"])
        repeated[key] = (row["final_tensor_sha256"], row["raw_RGB_sha256"])
        case = cases[job["case_id"]]
        old = case.get("old_reference") if normalized_arm == "F" else case.get("old_fix32") if normalized_arm == "FIX32" else None
        if old:
            historical = torch.load(ROOT/old,weights_only=True,map_location="cpu")
            assert torch.equal(raw["final"],historical["final"])
            archived_pairs += 1
        if job["phase"] == "temporal":
            if job["arm"] == "F":
                temporal_prefix[job["case_id"]] = raw["states"][:2]
            else:
                assert all(torch.equal(a,b) for a,b in zip(temporal_prefix[job["case_id"]],raw["states"][:2]))
                prefix_pairs += 1
        if metadata["plan"] is not None:
            plan = metadata["plan"]
            trace = metadata["bridge_record"]["trace"]
            assert len(trace) == 24 and metadata["bridge_record"]["condition_count"] == 4096
            for event in trace:
                count = plan["target_tokens_per_step"][event["step"]-1]
                assert event["selected_target"] == count
                assert event["condition"] == 4096
                assert event["image_qkv_tokens"] == event["image_mlp_tokens"] == count+4096
                assert event["joint_kv_tokens"] == 8192+event["text_tokens"]
                assert event["joint_query_tokens"] == count+4096+event["text_tokens"]
    for case in cases.values():
        assert sha(ROOT/case["image"]) == case["source_sha256"]
        counters = []
        for arm in ("EARLY","MIDDLE","LATE"):
            p=case["plans"][arm]
            assert p["total_target_token_updates"] == sum(p["target_tokens_per_step"])
            for k in (0,1,7): assert p["selected_indices_per_step"][k] == list(range(4096))
            counters.append(collections.Counter(j for ids in p["selected_indices_per_step"] for j in ids))
        assert counters[0] == counters[1] == counters[2]
        same_budget_sources += 1
    assert euler_steps == 272 and prefix_pairs == 24 and same_budget_sources == 8
    result = dict(status="PASS", requests=82, actual_dit=656, warmup_dit=16, eligible_timing_dit=384,
                  temporal_dit=256, CPU_Euler_recomputed_steps=euler_steps, temporal_anchor_pairs=prefix_pairs,
                  equal_budget_and_token_multiset_sources=same_budget_sources, archived_final_pairs=archived_pairs,
                  raw_outputs=82, frozen_code_files=len(spec["code_sha256"]),
                  protocol_sha256=sha(RUN/"preregistration.json"), GPU_or_model_calls_in_audit=0,
                  test64_opened=False)
    (ROOT/"data/latency_temporal_audit_v1.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
