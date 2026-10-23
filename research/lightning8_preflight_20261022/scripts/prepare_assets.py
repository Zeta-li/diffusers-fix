import hashlib
import json
import urllib.request
from pathlib import Path

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parent.parent
ROOT.joinpath("inputs").mkdir(parents=True, exist_ok=True)
request = urllib.request.Request(
    "https://api.github.com/repos/ModelTC/LightX2V-Qwen-Image-Lightning/commits/main",
    headers={"User-Agent": "qwen-edit-research-pilot"},
)
with urllib.request.urlopen(request, timeout=30) as response:
    revision = json.load(response)["sha"]

specs = [
    (
        "blocks_text",
        "qwen_edit2.webp",
        "Replace the words 'HEALTH INSURANCE' on the letter blocks with 'Tomorrow will be better'.",
        "文字替换与局部结构",
    ),
    (
        "shirt_color",
        "qwen_edit3.png",
        "Replace the polka-dot shirt with a light blue shirt.",
        "衣服编辑与人物遮挡",
    ),
    (
        "remove_hair",
        "qwen_edit6.webp",
        "Remove the hair from the plate.",
        "细结构删除与邻域保持",
    ),
]
cases = []
for case_id, filename, prompt, category in specs:
    url = (
        "https://raw.githubusercontent.com/ModelTC/LightX2V-Qwen-Image-Lightning/"
        f"{revision}/examples/qwen_image_edit/{filename}"
    )
    source = ROOT / "inputs" / filename
    with urllib.request.urlopen(url, timeout=60) as response:
        source.write_bytes(response.read())
    image = ImageOps.exif_transpose(Image.open(source)).convert("RGB")
    path = ROOT / "inputs" / f"{case_id}.png"
    image.save(path)
    cases.append(
        {
            "case_id": case_id,
            "image": str(path.relative_to(ROOT)),
            "prompt": prompt,
            "category": category,
            "source_url": url,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "input_wh": list(image.size),
            "split": "public_demo_pilot_not_formal_test",
        }
    )

record = {"source_repo_revision": revision, "cases": cases}
ROOT.joinpath("cases.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
thumbs = []
for case in cases:
    image = Image.open(ROOT / case["image"])
    image.thumbnail((480, 480))
    canvas = Image.new("RGB", (500, 540), "white")
    canvas.paste(image, ((500 - image.width) // 2, 35))
    from PIL import ImageDraw

    ImageDraw.Draw(canvas).text((12, 8), case["case_id"], fill="black")
    thumbs.append(canvas)
sheet = Image.new("RGB", (500 * len(thumbs), 540), "white")
for i, thumb in enumerate(thumbs):
    sheet.paste(thumb, (500 * i, 0))
sheet.save(ROOT / "inputs" / "contact_sheet.png")
print(json.dumps(record, ensure_ascii=False, indent=2))
