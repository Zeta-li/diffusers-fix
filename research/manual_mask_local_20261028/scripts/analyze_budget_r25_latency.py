import collections
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/budget_r25_latency_v1'


def main():
    status=json.loads((RUN/'status.json').read_text())
    if status['phase']!='complete_analysis_pending':raise RuntimeError('Timing incomplete')
    rows=[json.loads(s) for s in (RUN/'records.jsonl').read_text().splitlines()]
    spec=json.loads((RUN/'preregistration.json').read_text())
    assert len(rows)==100 and status['completed_dit']==status['invoked_dit']==800
    paired=[r for r in rows if r['phase']=='measured'];assert len(paired)==96
    cases=[c['case_id'] for c in spec['cases']];arms=['F','B','R25','C']
    matrix=np.empty((8,4));per_source=[]
    for i,cid in enumerate(cases):
        source={}
        for j,a in enumerate(arms):
            group=[r for r in paired if r['case_id']==cid and r['arm']==a]
            assert len(group)==3 and len({r['RGB_sha256'] for r in group})==1
            values=np.array([r['seconds'] for r in group]);matrix[i,j]=values.mean()
            source[a]={'mean_seconds':float(values.mean()),'seconds':values.tolist(),'CV':float(values.std(ddof=1)/values.mean())}
        per_source.append(dict(case_id=cid,arms=source))
    summary={}
    rng=np.random.default_rng(2026103003);indices=rng.integers(0,8,size=(10000,8))
    for j,a in enumerate(arms):
        values=np.array([r['seconds'] for r in paired if r['arm']==a])
        ratios=matrix[indices,0].mean(1)/matrix[indices,j].mean(1)
        summary[a]=dict(mean_seconds=float(values.mean()),median_seconds=float(np.median(values)),P90_seconds=float(np.percentile(values,90)),
            speedup_vs_F=float(matrix[:,0].mean()/matrix[:,j].mean()),time_saved_fraction=float(1-matrix[:,j].mean()/matrix[:,0].mean()),
            speedup_source_bootstrap95=np.percentile(ratios,[2.5,97.5]).tolist(),peak_GiB=max(r['peak_GiB'] for r in paired if r['arm']==a),
            mean_within_source_CV=float(np.mean([r['arms'][a]['CV'] for r in per_source])))
    summary['R25_time_change_vs_B']=summary['R25']['mean_seconds']/summary['B']['mean_seconds']-1
    summary['R25_time_change_vs_C']=summary['R25']['mean_seconds']/summary['C']['mean_seconds']-1
    summary['C_time_change_vs_B']=summary['C']['mean_seconds']/summary['B']['mean_seconds']-1
    numerical=json.loads((ROOT/'runs/budget_r25_v1/analysis/results.json').read_text())
    q={a:dict(RGB_MAE=float(np.mean([r['RGB_MAE'] for r in numerical['records'] if r['arm']==a and r['case_id'] in cases]))) for a in arms}
    result=dict(status='COMPLETE',candidate=spec['candidate'],actual_requests=100,actual_DiT=800,paired_requests=96,warmups=4,
        summary=summary,per_source=per_source,same8_quality=q,all_outputs_bitwise_match_quality_evaluation=True,
        all_repeat_outputs_identical=True,source_bootstrap_note='resampling8fixedCal sources,10000 draws;not a population generalization interval',
        common_engine='untiled VAE;residentGPU components;unmergedFP32LoRA;allplan,promptencode,sourceVAE,DiT,decode,PILandcachecleanupincluded')
    folder=RUN/'analysis';folder.mkdir(exist_ok=True);(folder/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,ax=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    labels=['F','B','R25','C']
    vals=[summary[a]['mean_seconds'] for a in arms]
    ax[0].bar(labels,vals,color=['#607d8b','#4c78a8','#2e8b57','#b279a2']);ax[0].set(ylabel='Full request seconds',title='H800:8 sources x3paired repeats')
    for i,(a,v) in enumerate(zip(arms,vals)):ax[0].text(i,v+.04,f'{v:.3f}s\n{summary[a]["speedup_vs_F"]:.2f}x',ha='center')
    ax[0].set_ylim(0,max(vals)*1.22)
    errors=[q[a]['RGB_MAE'] for a in arms]
    ax[1].scatter(vals,errors,s=60)
    for x,y,label in zip(vals,errors,labels):ax[1].annotate(label,(x,y),xytext=(5,6),textcoords='offset points')
    ax[1].set(xlabel='Full request seconds',ylabel='RGB MAE versus F on same8sources',title='Measured quality-reference / latency tradeoff')
    ax[1].margins(x=.2,y=.2)
    fig.savefig(folder/'latency_and_tradeoff.png',dpi=160);plt.close(fig)
    print(json.dumps(result['summary']))


if __name__=='__main__':main()
