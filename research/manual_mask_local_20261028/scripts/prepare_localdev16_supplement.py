import collections
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path

from PIL import Image

import dev16_range_reader as reader

ROOT = Path(__file__).resolve().parents[1]


def main():
    spec = json.loads((ROOT / 'data/localdev16_source_supplement_v1.json').read_text())
    if hashlib.sha256((ROOT / 'scripts/dev16_range_reader.py').read_bytes()).hexdigest() != spec['reader_sha256']:
        raise RuntimeError('Reader changed')
    output = ROOT / 'inputs/localdev16_supplement_v1'
    output.mkdir()
    grouped = collections.defaultdict(list)
    for row in spec['rows']:
        grouped[row['shard_path']].append(row)
    columns = spec['columns']

    def fetch(shard):
        path = shard['path']
        if shard['split'] != 'dev':
            raise RuntimeError('Only official dev sources permitted')
        audit = output / (Path(path).stem + '_network.jsonl')
        result = []
        with reader.AuditRangeFile(shard['source_url'], shard['size_bytes'], audit,
                                  'LocalDev16_source_only', path) as handle:
            parquet = reader.open_parquet(handle)
            for group in sorted({r['row_group'] for r in grouped[path]}):
                table = parquet.read_row_group(group, columns=columns, use_threads=False)
                if table.column_names != columns:
                    raise RuntimeError('Projection differs')
                for row in [r for r in grouped[path] if r['row_group'] == group]:
                    actual = table.slice(row['row_in_group'], 1).to_pylist()[0]
                    if (str(actual['img_id']) != row['original_img_id'] or actual['turn_index'] != 1
                            or actual['instruction'] != row['instruction']):
                        raise RuntimeError('Source locator differs')
                    raw = actual['source_img']['bytes']
                    dst = output / str(int(row['img_id']))
                    dst.mkdir()
                    (dst / 'source.original').write_bytes(raw)
                    im = Image.open(io.BytesIO(raw)).convert('RGB')
                    record = dict(row, image=str((dst / 'source.original').relative_to(ROOT)),
                                  source_wh=list(im.size), signature=reader.image_signature(im, raw),
                                  source_sha256=hashlib.sha256(raw).hexdigest(),
                                  source_url=shard['source_url'], columns_read=columns,
                                  target_column_read=False, test_media_read=False)
                    (dst / 'provenance.json').write_text(json.dumps(record, indent=2) + '\n')
                    result.append(record)
                    print('source_saved', row['img_id'], flush=True)
        return result

    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch, shard) for shard in spec['shards']]
        for future in concurrent.futures.as_completed(futures):
            records.extend(future.result())
    if len(records) != len(spec['rows']):
        raise RuntimeError('Incomplete source acquisition')
    (output / 'manifest.json').write_text(json.dumps(dict(records=records,
        source_only=True, model_requests=0, target_column_read=False, test_media_read=False), indent=2) + '\n')
    print('SOURCE_ACQUISITION_COMPLETE', len(records), flush=True)


if __name__ == '__main__':
    main()
