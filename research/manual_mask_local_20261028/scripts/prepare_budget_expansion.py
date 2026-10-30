import collections
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import traceback

import numpy as np
from PIL import Image

import dev16_range_reader as reader

ROOT = Path(__file__).resolve().parents[1]


def decode(source, raw):
    rgba = np.asarray(Image.open(io.BytesIO(raw)).convert('RGBA'))
    alpha = rgba[:, :, 3]
    if np.any(alpha != 255):
        if not np.any(alpha == 0):
            raise ValueError('Ambiguous alpha mask')
        positive = alpha < 255
    else:
        if not np.array_equal(rgba[:, :, 0], rgba[:, :, 1]) or not np.array_equal(rgba[:, :, 1], rgba[:, :, 2]):
            raise ValueError('Ambiguous opaque colored mask')
        positive = rgba[:, :, 0] > 0
    if not positive.any():
        raise ValueError('Empty region')
    h, w = positive.shape
    if abs(source.width / source.height - w / h) > .01:
        raise ValueError('Mask/source aspect mismatch')
    ys, xs = np.nonzero(positive)
    bbox = [float(xs.min() * source.width / w), float(ys.min() * source.height / h),
            float((xs.max() + 1) * source.width / w), float((ys.max() + 1) * source.height / h)]
    return bbox, float(positive.mean())


def main():
    spec = json.loads((ROOT / 'data/budget_expansion_sources_v1.json').read_text())
    out = ROOT / 'inputs/budget_expansion_v1'
    out.mkdir(exist_ok=True)
    grouped = collections.defaultdict(list)
    for row in spec['rows']:
        grouped[row['shard_path']].append(row)
    columns = ['img_id', 'turn_index', 'instruction', 'source_img', 'mask_img']

    def fetch(shard):
        if shard['split'] not in ('train', 'dev'):
            raise RuntimeError('Test media forbidden')
        wanted = [r for r in grouped[shard['path']] if not (out / r['img_id'] / 'provenance.json').exists()]
        if not wanted:
            return []
        result = []
        with reader.AuditRangeFile(shard['source_url'], shard['size_bytes'], out / (Path(shard['path']).stem + '_network.jsonl'),
                                  'expanded_training_source_and_annotation_only', shard['path']) as handle:
            parquet = reader.open_parquet(handle)
            for group in sorted({r['row_group'] for r in wanted}):
                table = parquet.read_row_group(group, columns=columns, use_threads=False)
                if table.column_names != columns:
                    raise RuntimeError('Column projection differs')
                for row in [r for r in wanted if r['row_group'] == group]:
                    actual = table.slice(row['row_in_group'], 1).to_pylist()[0]
                    if (str(actual['img_id']) != row['original_img_id'] or actual['turn_index'] != 1
                            or actual['instruction'] != row['instruction']):
                        raise RuntimeError('Locator differs')
                    raw = actual['source_img']['bytes']; mask = actual['mask_img']['bytes']
                    im = Image.open(io.BytesIO(raw)).convert('RGB')
                    folder = out / row['img_id']; folder.mkdir(exist_ok=True)
                    (folder / 'source.original').write_bytes(raw)
                    (folder / 'mask.original').write_bytes(mask)
                    geometry = {}
                    try:
                        bbox, area = decode(im, mask)
                        fraction = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / (im.width * im.height)
                        geometry = dict(bbox=bbox, mask_fraction=area, bbox_fraction=fraction,
                            geometry_eligible=im.width == im.height and .002 <= fraction <= .50,
                            geometry_reason='square_source_and_bbox_0.2_to_50_percent')
                    except Exception as exc:
                        geometry = dict(geometry_eligible=False, geometry_reason=str(exc))
                    record = dict(row, **geometry, image=str((folder / 'source.original').relative_to(ROOT)),
                        source_wh=list(im.size), signature=reader.image_signature(im, raw),
                        source_sha256=hashlib.sha256(raw).hexdigest(), mask_sha256=hashlib.sha256(mask).hexdigest(),
                        columns_read=columns, source_url=shard['source_url'], target_column_read=False, test_media_read=False)
                    (folder / 'provenance.json').write_text(json.dumps(record, indent=2) + '\n')
                    result.append(record)
                    print('source_saved', row['img_id'], geometry['geometry_eligible'], flush=True)
        return result

    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(fetch, shard): shard['path'] for shard in spec['shards']}
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception:
                failures.append(dict(shard=futures[future], error=traceback.format_exc()))
    records = [json.loads((out / r['img_id'] / 'provenance.json').read_text()) for r in spec['rows']
               if (out / r['img_id'] / 'provenance.json').exists()]
    (out / 'manifest.json').write_text(json.dumps(dict(records=records, failures=failures,
        expected=len(spec['rows']), complete=len(records) == len(spec['rows']),
        source_only=True, model_requests=0, target_column_read=False, test_media_read=False), indent=2) + '\n')
    print('SOURCE_COMPLETE', len(records), 'of', len(spec['rows']), 'failures', len(failures), flush=True)
    if failures:
        raise RuntimeError('Incomplete acquisition; rerun only missing shards')


if __name__ == '__main__':
    main()
