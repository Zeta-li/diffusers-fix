import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/budget_sweep_v1'


def main():
    folder = RUN / 'blind_review'
    scores = json.loads((folder / 'scores_before_unblinding.json').read_text())
    receipt = json.loads((folder / 'freeze_receipt.json').read_text())
    assert scores['mapping_read'] is False and scores['status'] == 'FROZEN_BEFORE_UNBLINDING'
    digest = hashlib.sha256((folder / 'scores_before_unblinding.json').read_bytes()).hexdigest()
    assert digest == receipt['scores_sha256']
    key = json.loads((folder / 'unblinding_key.json').read_text())
    cases = {c['case_id']:c for c in json.loads((RUN / 'preregistration.json').read_text())['cases']}
    assert len(scores['rows']) == len(cases) == 48
    rows=[]
    for source in scores['rows']:
        cid=source['case_id']; assert set(source['scores']) == set('ABCD')
        for label, value in source['scores'].items():
            assert all(isinstance(value[k],bool) for k in ('edit','preservation','uncertain'))
            rows.append(dict(case_id=cid,role=cases[cid]['role'],label=label,arm=key[cid][label],**value,
                joint=value['edit'] and value['preservation'] and not value['uncertain']))
    summary={};discordance={}
    for role in ('train','calibration','all'):
        subset=rows if role=='all' else [r for r in rows if r['role']==role]
        summary[role]={};discordance[role]={}
        table={a:{r['case_id']:r for r in subset if r['arm']==a} for a in ('F','B','B50','C')}
        for arm in table:
            vals=list(table[arm].values())
            summary[role][arm]=dict(n=len(vals),**{k:sum(r[k] for r in vals) for k in ('edit','preservation','joint','uncertain')})
        for base,arm in (('F','B'),('F','B50'),('F','C'),('B','B50'),('B','C'),('B50','C')):
            discordance[role][base+'_to_'+arm]=dict(
                repairs=[cid for cid in table[arm] if table[arm][cid]['joint'] and not table[base][cid]['joint']],
                regressions=[cid for cid in table[arm] if not table[arm][cid]['joint'] and table[base][cid]['joint']])
    result=dict(reviewer='single assistant anonymous before unblinding; not human consensus',
        scores_sha256=digest,summary=summary,discordance=discordance,records=rows)
    (folder/'unblinded_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(summary=summary,discordance=discordance),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
