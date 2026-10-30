import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    archive = Path('/tmp/budget_r25_raw_v1.tar.gz')
    assert not archive.exists(), 'Preserve existing archive'
    paths = set()
    for name, count in [('budget_r25_v1', 384), ('budget_r25_latency_v1', 800)]:
        folder = ROOT / 'runs' / name
        status = json.loads((folder / 'status.json').read_text())
        assert status['phase'] == 'complete_analysis_pending'
        assert status['invoked_dit'] == status['completed_dit'] == count
        paths.update(p for p in folder.rglob('*') if p.is_file())
    paths.update(ROOT.glob('scripts/*budget_r25*.py'))
    paths.update(ROOT.glob('data/budget_r25_*preregistration.json'))
    paths.update(ROOT.glob('data/budget_r25_*protocol*.json'))
    paths.update(ROOT / 'data' / x for x in [
        'budget_r25_action_labels_v1.json', 'budget_r25_execution_notes_v1.json',
    ])
    rows = [dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=digest(p))
            for p in sorted(paths)]
    manifest = ROOT / 'data/budget_r25_raw_manifest_v1.json'
    manifest.write_text(json.dumps(dict(files=rows, count=len(rows),
        bytes=sum(x['bytes'] for x in rows), new_requests=148, new_DiT=1184,
        reused_quality_requests=192, input_checkpoint_references='See preregistration hashes'), indent=2) + '\n')
    print('Manifest complete', len(rows), flush=True)
    with tarfile.open(archive, 'w:gz', compresslevel=2) as tar:
        for row in rows:
            tar.add(ROOT / row['path'], arcname=row['path'])
        tar.add(manifest, arcname=str(manifest.relative_to(ROOT)))
    expected = {row['path']: row for row in rows}
    checked = 0
    with tarfile.open(archive) as tar:
        for member in tar:
            if member.name not in expected:
                assert member.name == str(manifest.relative_to(ROOT))
                continue
            assert member.isfile()
            f = tar.extractfile(member)
            h = hashlib.sha256()
            for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
                h.update(block)
            assert h.hexdigest() == expected[member.name]['sha256'], member.name
            checked += 1
    assert checked == len(rows)
    receipt = dict(status='ARCHIVED_REMOTE_AND_FULL_CONTENT_VERIFIED', files=len(rows),
        raw_bytes=sum(x['bytes'] for x in rows), archive=str(archive), archive_bytes=archive.stat().st_size,
        archive_sha256=digest(archive), manifest_sha256=digest(manifest),
        verified_members=checked, local_full_raw_copy=False)
    (ROOT / 'data/budget_r25_raw_archive_receipt_v1.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
