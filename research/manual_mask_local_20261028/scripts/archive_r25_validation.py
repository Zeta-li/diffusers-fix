import datetime
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    runs = ['r25_seed2_v1', 'r25_fresh_dev32_v1', 'r25_fresh_latency_v1']
    files = set()
    for name in runs:
        run = ROOT / 'runs' / name
        status = json.loads((run / 'status.json').read_text())
        spec = json.loads((run / 'preregistration.json').read_text())
        assert status['successful_requests'] == spec['planned_requests']
        assert status['completed_dit'] == spec.get('planned_dit', spec.get('planned_DiT'))
        assert (run / 'analysis/final_decision.json').exists()
        files.update(p for p in run.rglob('*') if p.is_file())
        files.add(ROOT / spec.get('noise', 'runs/s1_v2/explicit_noise.pt'))
        for case in spec['cases']:
            files.update(p for p in (ROOT / case['image']).parent.iterdir() if p.is_file())
    files.update(p for p in (ROOT / 'inputs/r25_dev_candidates_v1').rglob('*') if p.is_file())
    files.update(p for p in (ROOT / 'data').glob('r25_*') if p.is_file() and 'archive' not in p.name and 'delivery' not in p.name and 'raw_manifest' not in p.name)
    files.update(ROOT / 'scripts' / n for n in ('archive_r25_validation.py', 'analyze_r25_validation.py',
        'render_r25_validation.py', 'unblind_r25_validation.py', 'prepare_r25_dev_sources.py', 'review_r25_dev_sources.py',
        'plot_r25_validation.py', 'analyze_r25_fresh_latency.py'))
    files.update(p for p in (ROOT / 'notes').glob('r25_*') if p.is_file())
    rows = [dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(files)]
    manifest = ROOT / 'data/r25_validation_raw_manifest_v1.json'
    manifest.write_text(json.dumps(dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        files=rows, count=len(rows), raw_bytes=sum(r['bytes'] for r in rows),
        external_fixed_checkpoint='runs/budget_expansion_train_v1/checkpoints/student_inference.pt',
        external_checkpoint_sha256='216794f27aacf696af7835612234dcdf04cb7dcc6dbfbc9ecffcfa3ef55332f7',
        inherited_seed2_conditions='runs/budget_expansion_train_v1/conditioning; hashes in frozen protocol and prior archive'), indent=2) + '\n')
    archive = Path('/tmp/r25_validation_raw_v1.tar.gz')
    assert not archive.exists()
    with tarfile.open(archive, 'w:gz', compresslevel=2) as output:
        for row in rows:
            output.add(ROOT / row['path'], arcname=row['path'])
    expected = {r['path']: r for r in rows}
    verified = 0
    with tarfile.open(archive, 'r:gz') as source:
        for member in source:
            assert member.isfile() and member.name in expected
            record = expected.pop(member.name)
            stream = source.extractfile(member)
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
            assert digest.hexdigest() == record['sha256'] and member.size == record['bytes'], member.name
            verified += 1
    assert not expected
    receipt = dict(status='PASS_ALL_MEMBERS_VERIFIED', archive=str(archive), archive_bytes=archive.stat().st_size,
        archive_sha256=sha(archive), manifest_sha256=sha(manifest), files=len(rows), verified=verified,
        raw_bytes=sum(r['bytes'] for r in rows))
    (ROOT / 'data/r25_validation_raw_archive_receipt_v1.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
