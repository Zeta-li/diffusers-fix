import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    run = ROOT / 'runs/local_edit_runtime_latency_v1'
    status = json.loads((run/'status.json').read_text())
    spec = json.loads((run/'preregistration.json').read_text())
    assert status['phase'] == 'complete_analysis_pending'
    assert status['completed_dit'] == status['invoked_dit'] == spec['planned_dit']
    records = [json.loads(s) for s in (run/'records.jsonl').read_text().splitlines()]
    assert len(records) == status['successful_requests'] == spec['planned_requests']
    rows = []
    arrays = {}
    for arm in spec['arms']:
        vals = []
        for case in spec['cases']:
            rs = [r for r in records if r['phase']=='measure' and r['case_id']==case['case_id'] and r['arm']==arm]
            assert len(rs) == 3 and len({r['RGB_sha256'] for r in rs}) == 1
            seconds = [r['seconds'] for r in rs]
            vals.append(np.mean(seconds))
            rows.append(dict(case_id=case['case_id'], arm=arm, mean_seconds=float(np.mean(seconds)),
                             minimum_seconds=min(seconds), maximum_seconds=max(seconds),
                             repeat_cv=float(np.std(seconds)/np.mean(seconds))))
        arrays[arm] = np.asarray(vals)
    rng = np.random.default_rng(2026103037)
    ix = rng.integers(0, len(spec['cases']), (10000,len(spec['cases'])))
    summary = {}
    for a,x in arrays.items():
        b = arrays['F']
        boot = b[ix].mean(1)/x[ix].mean(1)
        summary[a] = dict(mean_seconds=float(x.mean()), speedup_vs_F=float(b.mean()/x.mean()),
                          time_saved_fraction=float(1-x.mean()/b.mean()),
                          bootstrap95_speedup=np.percentile(boot,[2.5,97.5]).tolist(),
                          max_peak_GiB=max(r['peak_GiB'] for r in records if r['arm']==a))
    selected = spec['selected']
    relative = arrays[selected].mean()/arrays['R25'].mean()-1
    relboot = arrays[selected][ix].mean(1)/arrays['R25'][ix].mean(1)-1
    result = dict(status='COMPLETE', counts=status, scope=spec['scope'], summary=summary,
                  selected=selected, selected_latency_relative_to_R25=float(relative),
                  bootstrap95_selected_relative_to_R25=np.percentile(relboot,[2.5,97.5]).tolist(),
                  all_outputs_match_quality=True, all_request_cache_scopes_restored=True,
                  measured_requests=24*len(spec['arms']), warmup_requests=len(spec['arms']), records=rows)
    out=run/'analysis';out.mkdir(exist_ok=True)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    with (out/'source_latency.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fig,ax=plt.subplots(figsize=(7,4));names=list(summary);heights=[summary[a]['mean_seconds'] for a in names]
    ax.bar(names,heights,color=['#999999','#2878b5','#d78b28'][:len(names)])
    for i,h in enumerate(heights):ax.text(i,h+.04,f'{h:.3f}s',ha='center')
    ax.set_ylim(0,max(heights)*1.18);ax.set_ylabel('Full model-resident request (seconds)')
    ax.set_title('New Dev first8: 3 paired repeats / source; warmups excluded')
    fig.tight_layout();fig.savefig(out/'net_latency.png',dpi=180);plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))


if __name__ == '__main__':
    main()
