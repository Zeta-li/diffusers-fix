import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--target',type=Path);parser.add_argument('--torch-check',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parent;manifest=json.loads((root/'manifest.json').read_text())
    for row in manifest['files']:
        p=root/row['file'];assert p.is_file() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],row['file']
    code=json.loads((root/'code_manifest.json').read_text())
    for row in code['files']:
        p=root/row['path'];assert p.is_file() and sha(p)==row['sha256'],row['path']
    if args.torch_check:
        import torch
        state=torch.load(root/'LOCAL128_student.pt',weights_only=True,map_location='cpu')['adapters']
        assert len(state)==144 and sum(x.numel() for x in state.values())==4423680
    if args.target is not None:
        target=args.target.resolve();copies=[(root/r['file'],target/r['source']) for r in manifest['files']]
        copies += [(root/r['path'],target/r['path']) for r in code['files']]
        for src,dst in copies:
            assert dst.resolve().is_relative_to(target)
            if dst.exists() and sha(src)!=sha(dst):raise FileExistsError(str(dst))
        for src,dst in copies:
            dst.parent.mkdir(parents=True,exist_ok=True)
            if not dst.exists():shutil.copy2(src,dst)
        for src,dst in copies:assert sha(src)==sha(dst)
    elif not args.verify_only:
        parser.error('Choose --verify-only or --target')
    print(json.dumps(dict(status='PASS',weights=len(manifest['files']),code_files=len(code['files']),
                          torch_checked=args.torch_check,restored_to=str(args.target) if args.target else None)))


if __name__=='__main__':main()
