import datetime
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]
RUNS=['refresh_timing_pilot8_v1','refresh_timing_remaining40_v1','refresh_policy_v1','refresh_bandit_v1']


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    counts=[]
    for name in (RUNS[0],RUNS[1],RUNS[3]):
        s=json.loads((ROOT/'runs'/name/'status.json').read_text())
        assert s['phase']=='complete_analysis_pending'
        counts.append(dict(name=name,new_requests=s['successful_requests'],DiT=s['invoked_dit'],status_sha256=digest(ROOT/'runs'/name/'status.json')))
    assert sum(x['DiT'] for x in counts)==2048
    files=[];external=[]
    for name in RUNS:
        for p in sorted((ROOT/'runs'/name).rglob('*')):
            if not p.is_file():continue
            entry=dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=digest(p))
            if name.startswith('refresh_timing') and p.parent.parent.name=='evaluation' and p.stem in ('F','B','R25'):
                external.append(entry)
            else:files.append(entry)
    for p in sorted((ROOT/'scripts').glob('*refresh*py')):
        if p.name in ['freeze_refresh_timing.py','evaluate_refresh_timing.py','analyze_refresh_timing.py','refresh_policy.py','train_refresh_policy.py','run_refresh_bandit.py','analyze_refresh_policy.py','archive_refresh_policy.py','plot_refresh_policy.py']:
            files.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=digest(p)))
    for p in sorted((ROOT/'data').glob('refresh*json')):
        if 'manifest' not in p.name and 'receipt' not in p.name:
            files.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=digest(p)))
    for p in sorted((ROOT/'notes').glob('refresh*log')):
        files.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=digest(p)))
    report = ROOT/'20_刷新时机监督与真实Bandit实验_20261030.md'
    files.append(dict(path=str(report.relative_to(ROOT)),bytes=report.stat().st_size,sha256=digest(report)))
    manifest=dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,external_reused_outputs=external,
        external_weights='Frozen LOCAL128 and original noise/condition caches referenced by parent protocols; not duplicated.',counts=counts)
    mp=ROOT/'data/refresh_policy_raw_manifest_v1.json';assert not mp.exists()
    mp.write_text(json.dumps(manifest,indent=2)+'\n')
    tar=Path('/tmp/refresh_policy_raw_v1.tar.gz');assert not tar.exists()
    with tarfile.open(tar,'w:gz',compresslevel=1) as a:
        for row in files:a.add(ROOT/row['path'],arcname=row['path'],recursive=False)
        a.add(mp,arcname=str(mp.relative_to(ROOT)),recursive=False)
    verified=0
    expected={row['path']:row for row in files}
    with tarfile.open(tar,'r:gz') as a:
        for member in a:
            if member.name not in expected:continue
            h=hashlib.sha256();n=0
            with a.extractfile(member) as f:
                for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b);n+=len(b)
            assert h.hexdigest()==expected[member.name]['sha256'] and n==expected[member.name]['bytes']
            verified+=1
    assert verified==len(files)
    receipt=dict(archive=str(tar),sha256=digest(tar),bytes=tar.stat().st_size,verified_files=verified,
        manifest_sha256=digest(mp),external_reused_files=len(external),status='PASS')
    (ROOT/'data/refresh_policy_raw_archive_receipt_v1.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
