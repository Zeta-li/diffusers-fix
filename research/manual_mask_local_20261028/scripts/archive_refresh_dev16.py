import datetime
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    paths=set();counts=[]
    for name in ('refresh_dev16_v1','local_edit_runtime_latency_v1'):
        run=ROOT/'runs'/name;s=json.loads((run/'status.json').read_text())
        assert s['phase']=='complete_analysis_pending' and s['invoked_dit']==s['completed_dit']
        counts.append(dict(name=name,requests=s['successful_requests'],DiT=s['completed_dit']))
        paths.update(p for p in run.rglob('*') if p.is_file())
    assert sum(c['requests'] for c in counts)==139 and sum(c['DiT'] for c in counts)==1112
    spec=json.loads((ROOT/'runs/refresh_dev16_v1/preregistration.json').read_text())
    for c in spec['cases']:
        p=ROOT/c['image'];assert sha(p)==c['source_sha256'];paths.add(p)
    names=set(spec['code_sha256'])
    names.update(json.loads((ROOT/'runs/local_edit_runtime_latency_v1/preregistration.json').read_text())['code_sha256'])
    names.update(['run_local_edit.py','render_refresh_dev16.py','unblind_refresh_dev16.py','sglang_student_adapter.py','prepare_sglang_local_edit.py','archive_refresh_dev16.py'])
    paths.update(ROOT/'scripts'/n for n in names)
    for name in ('refresh_dev16_v1_preregistration.json','local_edit_runtime_latency_v1_preregistration.json',
                 'local_edit_runtime_cpu_contract_v1.json','local_edit_cli_smoke_v1.json','refresh_dev16_ledger_v1.json'):
        paths.add(ROOT/'data'/name)
    paths.update(p for p in (ROOT/'exports/sglang_local_edit_v1').rglob('*') if p.is_file())
    paths.update(ROOT/'notes'/n for n in ('refresh_dev16_v1.log','local_edit_runtime_latency_v1.log','sglang_prepare_v1.log'))
    paths.update(ROOT/n for n in ('21_新Dev16策略验证与统一入口净延迟_20261030.md','22_统一执行入口与SGLang适配说明_20261030.md'))
    files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(paths)]
    external=[]
    for k in ('checkpoint','head','noise'):
        p=ROOT/spec[k];assert sha(p)==spec[k+'_sha256'];external.append(dict(role=k,path=spec[k],sha256=spec[k+'_sha256']))
    manifest=dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,counts=counts,external_weights=external,
                  base_model='External frozen Qwen-Image-Edit + Lightning; same verified paths as run_free_pilot.py',
                  source_pool_QC='Existing source-only manifest/QC hashes retained in quality preregistration; Test64 never read')
    mp=ROOT/'data/refresh_dev16_raw_manifest_v1.json';assert not mp.exists();mp.write_text(json.dumps(manifest,indent=2)+'\n')
    archive=Path('/tmp/refresh_dev16_raw_v1.tar.gz');assert not archive.exists()
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
        for row in files:tar.add(ROOT/row['path'],arcname=row['path'],recursive=False)
        tar.add(mp,arcname=str(mp.relative_to(ROOT)),recursive=False)
    expected={r['path']:r for r in files};verified=set()
    with tarfile.open(archive,'r:gz') as tar:
        for m in tar:
            if m.name not in expected:continue
            assert m.name not in verified and m.isfile()
            h=hashlib.sha256();n=0
            with tar.extractfile(m) as f:
                for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b);n+=len(b)
            assert h.hexdigest()==expected[m.name]['sha256'] and n==expected[m.name]['bytes']
            verified.add(m.name)
    assert len(verified)==len(files)
    receipt=dict(status='PASS',archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,verified_files=len(verified),manifest_sha256=sha(mp))
    (ROOT/'data/refresh_dev16_raw_archive_receipt_v1.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
