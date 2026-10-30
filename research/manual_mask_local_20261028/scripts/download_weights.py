import hashlib
import json
import os
from pathlib import Path
import time
import traceback

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")

from huggingface_hub import HfApi, snapshot_download

ROOT = Path("/workspace/model-weights")
STATUS = ROOT / "deployment_20261028.json"
SPECS = [
    ("Qwen/Qwen-Image-Edit", "ac7f9318f633fc4b5778c59367c8128225f1e3de", "Qwen-Image-Edit", None),
    ("lightx2v/Qwen-Image-Lightning", "d468c534b360e0754c6bf9e38ed49e1e09cd2e2a", "Qwen-Image-Lightning", ["Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors"]),
]


def write(value):
    value["updated_at_unix"] = time.time()
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(STATUS)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    state = {"phase": "metadata", "models": []}
    write(state)
    try:
        for repo, revision, folder, patterns in SPECS:
            info = HfApi().model_info(repo, revision=revision, files_metadata=True)
            if info.sha != revision:
                raise RuntimeError("revision mismatch")
            files = [s for s in info.siblings if patterns is None or s.rfilename in patterns]
            item = {"repo": repo, "revision": revision, "path": str(ROOT / folder), "phase": "download", "files": []}
            for s in files:
                lfs = s.lfs
                item["files"].append({"name": s.rfilename, "size": s.size, "sha256": getattr(lfs, "sha256", None) if lfs else None, "git_blob_id": s.blob_id})
            state["models"].append(item)
            write(state)
            snapshot_download(repo, revision=revision, local_dir=ROOT / folder, allow_patterns=patterns, max_workers=6)
            item["phase"] = "verify"
            write(state)
            for f in item["files"]:
                path = ROOT / folder / f["name"]
                if path.stat().st_size != f["size"]:
                    raise RuntimeError("size mismatch: " + f["name"])
                h = hashlib.sha256() if f["sha256"] else hashlib.sha1()
                if not f["sha256"]:
                    h.update(("blob " + str(f["size"]) + "\0").encode())
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        h.update(chunk)
                expected = f["sha256"] or f["git_blob_id"]
                if h.hexdigest() != expected:
                    raise RuntimeError("hash mismatch: " + f["name"])
                f["verified"] = True
                write(state)
            item["phase"] = "complete"
            item["verified_bytes"] = sum(f["size"] for f in item["files"])
            write(state)
        state["phase"] = "complete"
        write(state)
    except BaseException as exc:
        state["phase"] = "failed"
        state["error"] = type(exc).__name__ + ": " + str(exc)
        write(state)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
