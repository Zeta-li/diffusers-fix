import lzma, hashlib, json
from pathlib import Path
root = Path(__file__).resolve().parent
for item in json.loads((root / "compressed_protocols.json").read_text()):
    payload = lzma.decompress((root / item["path"]).read_bytes())
    if hashlib.sha256(payload).hexdigest() != item["sha256"]:
        raise ValueError(item["path"])
    target = root / item["restored_path"]
    if target.exists() and target.read_bytes() != payload:
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
print("Verified and restored frozen protocols")
