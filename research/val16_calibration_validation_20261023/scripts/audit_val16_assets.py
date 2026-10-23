import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--source-stage', type=Path, required=True)
    args = parser.parse_args()
    root, source = args.root.resolve(), args.source_stage.resolve()
    parent = read(source / 'data' / 'frozen_selection.json')
    prereg = read(root / 'data' / 'asset_preregistration.json')
    inventory = read(source / 'data' / 'repo_inventory.json')
    cases = read(root / 'cases_val16.json')
    report = read(root / 'data' / 'val16_assets.json')
    expected = parent['selections']['val16']
    rows = cases['cases']
    assert prereg['parent_frozen_selection_sha256'] == digest(source / 'data' / 'frozen_selection.json')
    assert prereg['parent_train8_assets_sha256'] == digest(source / 'data' / 'train8_assets.json')
    assert prereg['parent_inventory_sha256'] == digest(source / 'data' / 'repo_inventory.json')
    assert prereg['extraction_script_sha256'] == digest(root / 'scripts' / 'prepare_val16.py')
    assert len(rows) == 16 and report['val16_records'] == rows
    assert [row['img_id'] for row in rows] == [row['img_id'] for row in expected]
    assert report['ready_for_validation'] and cases['ready_for_validation']
    assert not report['hard_failures'] and not report['gray_zones']
    references = dict(prereg['reference_signatures'])
    assert len(references) == 11
    asset_checks, pair_checks, roi_checks = 0, 0, 0
    for row, original in zip(rows, expected):
        assert all(row[key] == value for key, value in original.items())
        assert row['official_split'] == 'dev' and row['turn_index'] == 1 and row['status'] == 'PASS'
        for path_key, hash_key in [('image', 'source_sha256'), ('original_mask', 'mask_sha256'), ('binary_mask', 'binary_mask_sha256')]:
            assert digest(root / row[path_key]) == row[hash_key]
            asset_checks += 1
        grid_h, grid_w = row['target_grid_hw']
        height, width = row['target_hw']
        assert width % 32 == 0 and height % 32 == 0 and grid_w * 16 == width and grid_h * 16 == height
        ratio = row['source_wh'][0] / row['source_wh'][1]
        raw_w = math.sqrt(1024 * 1024 * ratio)
        assert [width, height] == [round(raw_w / 32) * 32, round(raw_w / ratio / 32) * 32]
        x0, y0, x1, y1 = row['target_mask_bbox_xyxy_exclusive']
        box = [max(0, x0 // 16 - 2), max(0, y0 // 16 - 2), min(grid_w, math.ceil(x1/16)+2), min(grid_h, math.ceil(y1/16)+2)]
        assert box == row['core_token_bbox_xyxy_exclusive']
        indices = [y * grid_w + x for y in range(box[1], box[3]) for x in range(box[0], box[2])]
        assert indices == row['indices'] and len(indices) == row['n_core_tokens']
        assert len(indices) / (grid_h * grid_w) == row['target_active_fraction']
        roi_checks += 1
        assert len(row['duplicate_comparisons']) == len(references)
        for comparison in row['duplicate_comparisons']:
            reference = references[comparison['reference_id']]
            signature = row['signature']
            ah = bin(int(signature['average_hash'], 16) ^ int(reference['average_hash'], 16)).count('1')
            dh = bin(int(signature['difference_hash'], 16) ^ int(reference['difference_hash'], 16)).count('1')
            aspect = abs(math.log((signature['width']/signature['height']) / (reference['width']/reference['height'])))
            exact = signature['raw_sha256'] == reference['raw_sha256'] or signature['rgb_sha256'] == reference['rgb_sha256']
            near = ah <= 6 and dh <= 6 and aspect <= 0.02
            gray = not exact and not near and (ah <= 10 or dh <= 10)
            assert (comparison['average_hash_distance'], comparison['difference_hash_distance']) == (ah, dh)
            assert comparison['exact'] == exact and comparison['near_duplicate'] == near and comparison['gray_zone'] == gray
            assert comparison['verdict'] == 'PASS' and not exact and not near and not gray
            pair_checks += 1
        references['val16/' + row['img_id']] = row['signature']
    assert asset_checks == 48 and roi_checks == 16 and pair_checks == 16 * 11 + 120
    layouts = [json.loads(line) for line in (root / 'data' / 'parquet_layouts.jsonl').read_text().splitlines() if line.strip()]
    forbidden = {}
    for layout in layouts:
        ranges = layout['target_payload_ranges_exclusive']
        if layout['shard'] in forbidden:
            assert forbidden[layout['shard']] == ranges
        forbidden[layout['shard']] = ranges
    sizes = {row['path']: row['size_bytes'] for row in inventory['files']}
    events = [json.loads(line) for line in (root / 'data' / 'network_ranges.jsonl').read_text().splitlines() if line.strip()]
    assert events and all(row['status'] == 'complete' for row in events)
    received = 0
    for event in events:
        assert event['phase'] == 'val16_source_mask_extraction' and event['shard'].startswith('data/dev-')
        start, end = event['start_inclusive'], event['end_exclusive']
        assert event['response_http_status'] == 206
        assert event['received_bytes'] == event['requested_bytes'] == end - start
        assert event['content_range'] == f'bytes {start}-{end-1}/{sizes[event["shard"]]}'
        assert not any(start < upper and end > lower for lower, upper in forbidden[event['shard']])
        assert parent['contract']['revision'] in event['source_url']
        assert hashlib.sha256(event['source_url'].encode()).hexdigest() == event['source_url_sha256']
        received += event['received_bytes']
    assert report['network']['received_bytes'] == received
    assert not report['target_columns_read'] and not report['test_media_read'] and not report['train8_assets_modified']
    assert report['model_calls'] == report['dit_forwards'] == report['training_steps'] == 0
    rho = [row['target_active_fraction'] for row in rows]
    result = dict(status='PASS', frozen_val16_count=16, original_rank_order_preserved=True,
        source_mask_binary_sha256_checks=asset_checks, independent_core_geometry_checks=roi_checks,
        independently_recomputed_duplicate_pairs=pair_checks, prior_reference_count=11,
        within_val16_pairs=120, hard_duplicate_hits=0, gray_zone_hits=0,
        successful_http_ranges=len(events), failed_http_ranges=0, received_bytes=received,
        target_payload_range_overlaps=0, all_ranges_exact_206=True,
        core_fraction=dict(min=min(rho), median=statistics.median(rho), mean=statistics.mean(rho), max=max(rho)),
        target_columns_read=False, test_media_read=False, train8_assets_modified=False,
        model_calls=0, dit_forwards=0, training_steps=0,
        source_case_manifest_sha256=digest(root / 'cases_val16.json'),
        source_asset_manifest_sha256=digest(root / 'data' / 'val16_assets.json'),
        audit_script_sha256=digest(Path(__file__)))
    output = root / 'data' / 'asset_independent_validation.json'
    if output.exists():
        assert read(output) == result
    else:
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
