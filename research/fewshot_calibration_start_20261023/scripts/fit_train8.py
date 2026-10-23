import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np

from moments import fit_calibration,load_npz,save_npz

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    destination=ROOT/'data/calibration_train8.npz'
    if destination.exists():raise RuntimeError('Preserve existing fitted coefficients; use a new version for refitting')
    ledger=ROOT/'data/runs_train8.jsonl'
    rows=[json.loads(s) for s in ledger.read_text().splitlines()]
    complete=[r for r in rows if r['status']=='complete']
    ids=[r['case_id'] for r in complete]
    assert len(complete)==len(set(ids))==8
    assert all(r['formal_training_image'] and r['local_calls']==8 and r['oracle_calls']==6 for r in complete)
    cases=json.loads((ROOT/'cases_train8.json').read_text())['cases']
    assert set(ids)=={c['case_id'] for c in cases}
    images=[];sources={}
    for row in complete:
        path=ROOT/row['moments'];assert sha(path)==row['moments_sha256']
        images.append(load_npz(path));sources[row['moments']]=sha(path)
    layers=1+max(int(key.split(':')[0]) for key in images[0]['a'])
    channels=len(next(iter(images[0]['a'].values()))['sum_h'])
    assert all(len(image['a'])==layers and len(image['b'])==layers*6 for image in images)
    started=time.perf_counter()
    fitted=fit_calibration(images,layers,8,channels)
    assert fitted['source_image_count']==8
    assert all(np.isfinite(fitted['route_b'][key]).all() for key in ['alpha','beta'])
    fitted['provenance']={'source_moments_sha256':sources,'training_manifest_sha256':sha(ROOT/'cases_train8.json'),
        'ledger_sha256':sha(ledger),'fitter_sha256':sha(Path(__file__)),'new_model_queries':0}
    save_npz(destination,fitted)
    summary={'status':'complete_offline_train8_fit','source_image_count':8,'training_image_ids':fitted['source_image_ids'],
        'geometry':fitted['geometry'],'selected_eta':fitted['loo']['selected_eta'],
        'candidate_scores':[{'eta':v['eta'],'score':v['score']} for v in fitted['loo']['candidate_scores']],
        'route_a_observed_layer_anchors':int(fitted['route_a']['coverage'].sum()),
        'route_b_observed_layer_pairs':int(fitted['route_b']['coverage'].sum()),
        'fit_and_save_seconds':time.perf_counter()-started,'max_rss_kib_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'fitted_file':str(destination.relative_to(ROOT)),'fitted_sha256':sha(destination),'fitted_bytes':destination.stat().st_size,
        'new_model_queries':0,'gradient_steps':0,'closed_loop_val_performed':False,'quality_or_speed_claim':False,
        'provenance':fitted['provenance']}
    (ROOT/'data/fit_train8_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
