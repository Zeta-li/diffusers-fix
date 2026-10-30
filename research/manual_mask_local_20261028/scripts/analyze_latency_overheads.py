import json
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/"runs/latency_temporal_v1"


def main():
    spec=json.loads((RUN/"preregistration.json").read_text())
    latency=json.loads((RUN/"analysis/latency.json").read_text())
    first=spec["cases"][0]["case_id"]
    profiles={}
    for arm in ("F","EARLY"):
        path=RUN/"temporal"/first/(arm+"_r0_operator_profile.json")
        rows=json.loads(path.read_text())
        aten={r["name"]:r for r in rows if r["name"].startswith("aten::")}
        profiles[arm]=dict(aten_self_device_ms=sum(r["self_device_us"] for r in aten.values())/1000,
            selected_operators={name:dict(count=aten.get(name,{}).get("count",0),
                    self_device_ms=aten.get(name,{}).get("self_device_us",0)/1000,
                    self_cpu_ms=aten.get(name,{}).get("self_cpu_us",0)/1000)
                for name in ("aten::index_select","aten::index_copy_","aten::cat","aten::copy_","aten::addmm","aten::_cudnn_attention_forward")},
            aggregation_seconds_after_raw_save=(path.stat().st_mtime-(path.parent/(arm+"_r0.pt")).stat().st_mtime))
    caches=[];qratios=[];plans=[]
    for c in spec["cases"]:
        meta=json.loads((RUN/"latency"/c["case_id"]/'FIX32_r0.json').read_text())
        caches.append(meta["bridge_record"]["raw_cache_bytes_before_clear"])
        p=meta["plan"];plans.append(p["total_target_token_updates"]/(4096*8))
        trace=meta["bridge_record"]["trace"]
        text=trace[0]["text_tokens"]
        qratios.append((text+4096+4096*plans[-1])/(text+8192))
    result=dict(profile_scope="single paired goat F/EARLY full-pipeline profiler requests including trace capture, outside net latency; report self_device_us only, not nested device_total_us",
        limitation="operator self device sums do not measure removable end-to-end overhead; condition-image hidden rows are in joint matrix operations and are not independently timed",
        profiles=profiles,
        cache_bytes=caches,cache_GiB=[v/2**30 for v in caches],
        mean_target_update_fraction=statistics.mean(plans),mean_image_row_update_fraction=(1+statistics.mean(plans))/2,
        mean_joint_query_row_fraction=statistics.mean(qratios),joint_KV_length_fraction=1.0,
        condition_share_of_remaining_image_row_updates=1/(1+statistics.mean(plans)),
        plan_construction_mean_seconds=statistics.mean(sum(r["FIX32"]["plan_seconds"])/3 for r in latency["paired"]),
        mean_peak_F_GiB=statistics.mean(max(r["F"]["peak_allocated_GiB"]) for r in latency["paired"]),
        mean_peak_FIX32_GiB=statistics.mean(max(r["FIX32"]["peak_allocated_GiB"]) for r in latency["paired"]))
    (RUN/"analysis/overheads.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
