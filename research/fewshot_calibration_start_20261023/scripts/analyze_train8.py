import json
from pathlib import Path
import numpy as np
from moments import load_npz

ROOT=Path(__file__).resolve().parents[1]


def main():
    summary=json.loads((ROOT/'data/fit_train8_summary.json').read_text())
    rows=[json.loads(s) for s in (ROOT/'data/runs_train8.jsonl').read_text().splitlines()]
    scores=[]
    for row in rows:
        if row['status']!='complete':continue
        image=load_npz(ROOT/row['moments']);sse=energy=0.
        for record in image['b'].values():
            error=record['sum_x2']+record['sum_y2']-2*record['sum_xy']
            assert np.all(error>=-1e-8)
            sse+=float(np.maximum(error,0).sum());energy+=float(record['sum_y2'].sum())
        scores.append({'image_id':image['image_id'],'identity_normalized_sse':sse/(energy+1e-12)})
    identity=float(np.mean([row['identity_normalized_sse'] for row in scores]))
    best=next(row['score'] for row in summary['candidate_scores'] if row['eta']==summary['selected_eta'])
    result={'status':'complete','scope':'Exploratory within-Train8 LOO raw-Value prediction proxy; hyperparameter-selection LOO score is not independent Val',
        'source_image_count':8,'identity_equal_source_normalized_sse':identity,'selected_B_loo_normalized_sse':best,
        'relative_normalized_sse_reduction':1-best/identity,'selected_eta':summary['selected_eta'],
        'identity_per_image':scores,'additional_model_queries':0,'closed_loop_image_quality_established':False,
        'net_latency_gain_established':False,'rl_started':False}
    (ROOT/'data/train8_proxy_analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
