import argparse
import copy
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
from PIL import Image


HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent
SPEC = importlib.util.spec_from_file_location('magicbrush_prepare_v1_preserved', HERE.with_name('prepare_magicbrush.py'))
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)
ORIGINAL_DECODE = BASE.decode_geometry
ORIGINAL_OPEN = BASE.open_parquet


def decode_geometry(source_bytes, mask_bytes):
    with Image.open(io.BytesIO(mask_bytes)) as image:
        original_mode, original_format, original_size = image.mode, image.format, list(image.size)
        if image.getexif().get(274, 1) != 1:
            raise ValueError('Nontrivial mask EXIF orientation is unsupported')
        rgba = np.asarray(image.convert('RGBA'))
    alpha = rgba[:, :, 3]
    transparent = alpha == 0
    edges = (alpha > 0) & (alpha < 255)
    if np.any(alpha != 255):
        if not transparent.any():
            raise ValueError('A nonopaque alpha mask without a fully transparent edit region is ambiguous')
        positive = alpha < 255
        buffer = io.BytesIO()
        Image.fromarray(positive.astype(np.uint8) * 255).save(buffer, format='PNG')
        source, geometry, binary = ORIGINAL_DECODE(source_bytes, buffer.getvalue())
        geometry.update(mask_positive_rule='alpha < 255 for conservative ROI; alpha == 0 is the explicitly editable region',
            mask_encoding='DALL-E-style RGBA source canvas with transparent edit region',
            rgb_channels_used_for_edit_mask=False,
            alpha_zero_official_edit_pixels=int(transparent.sum()),
            alpha_antialias_edge_pixels=int(edges.sum()),
            alpha_lt255_conservative_roi_pixels=int(positive.sum()),
            conservative_edge_rule_is_research_protocol=True)
        ys, xs = np.nonzero(transparent)
        geometry['alpha_zero_bbox_xyxy_exclusive'] = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    else:
        source, geometry, binary = ORIGINAL_DECODE(source_bytes, mask_bytes)
        geometry.update(mask_encoding='opaque grayscale white-positive mask',
            rgb_channels_used_for_edit_mask=True, alpha_zero_official_edit_pixels=0,
            alpha_antialias_edge_pixels=0, alpha_lt255_conservative_roi_pixels=0,
            conservative_edge_rule_is_research_protocol=True)
    geometry.update(original_mask_mode=original_mode, original_mask_format=original_format,
        original_mask_wh=original_size,
        alpha_histogram={str(index): int(count) for index, count in enumerate(np.bincount(alpha.reshape(-1), minlength=256)) if count},
        decoder_version='alpha_v2',
        official_alpha_source='https://developers.openai.com/cookbook/examples/dalle/how_to_create_dynamic_masks_with_dall-e_and_segment_anything')
    return source, geometry, binary


def effective_contract(parent):
    result = copy.deepcopy(parent['contract'])
    result['geometry'].update(alpha='RGBA alpha==0 explicitly editable; alpha<255 conservative ROI includes antialias edges; opaque grayscale white>0 compatible',
        roi='bbox of alpha<255 for RGBA, otherwise positive opaque grayscale; expanded by 2 image tokens',
        decoder_version='alpha_v2', conservative_alpha_edge_rule_is_research_protocol=True)
    return result


def ensure_json(path, value):
    if path.exists():
        if BASE.load(path) != value:
            raise RuntimeError(f'Existing frozen v2 artifact differs: {path}')
    else:
        BASE.frozen_json(path, value)


def effective_selection(root, parent, child):
    parent_path = root / 'data' / 'frozen_selection.json'
    parent_sha = BASE.sha256(parent_path.read_bytes())
    payload = {key: value for key, value in parent.items() if key not in ('selection_payload_sha256', 'created_at', 'script_sha256')}
    if BASE.sha256(BASE.canonical(payload)) != parent['selection_payload_sha256']:
        raise RuntimeError('Parent frozen selection checksum is invalid')
    amendment = dict(version='alpha_v2', parent_frozen_selection_sha256=parent_sha,
        parent_contract_sha256=parent['contract_sha256'], wrapper_script_sha256=BASE.sha256(HERE.read_bytes()),
        preserved_v1_script_sha256=BASE.sha256(HERE.with_name('prepare_magicbrush.py').read_bytes()),
        unchanged=['revision', 'seed', 'all IDs', 'all metadata', 'all ranks', 'all locators', 'all official group splits'],
        official_edit_region='alpha == 0', conservative_roi_region='alpha < 255 including antialias edges',
        extraction_export_order='explicitly validate unique IDs and restore the original frozen hash-rank order after row-group I/O',
        conservative_edge_threshold_is_research_rule=True,
        observed_initial_sample=dict(img_id='257102', source_wh=[500, 500], mask_wh=[1024, 1024],
            mask_mode='RGBA', alpha_zero_pixels_reported_by_execution_host=147106,
            observation='mask RGB contains the source canvas and alpha erases the instructed cabinet region; no target examined'),
        sources=['https://arxiv.org/html/2306.10012v2#S3.SS2',
            'https://developers.openai.com/cookbook/examples/dalle/how_to_create_dynamic_masks_with_dall-e_and_segment_anything',
            'https://raw.githubusercontent.com/OSU-NLP-Group/MagicBrush/main/evaluation/export_test_set.py'],
        effective_contract=effective_contract(parent), target_images_read=False, replacement_performed=False)
    sidecar = child / 'data' / 'decoder_amendment.json'
    ensure_json(sidecar, amendment)
    result = copy.deepcopy(parent)
    result['contract'] = amendment['effective_contract']
    result['contract_sha256'] = BASE.sha256(BASE.canonical(result['contract']))
    result['parent_frozen_selection_sha256'] = parent_sha
    result['decoder_amendment_sha256'] = BASE.sha256(sidecar.read_bytes())
    result['decoder_amendment'] = 'preprocess_alpha_v2/data/decoder_amendment.json'
    result['script_sha256'] = amendment['wrapper_script_sha256']
    result.pop('selection_payload_sha256', None)
    result.pop('created_at', None)
    stable = {key: value for key, value in result.items() if key != 'script_sha256'}
    result['selection_payload_sha256'] = BASE.sha256(BASE.canonical(stable))
    assert result['selections'] == parent['selections']
    assert result['ranked_candidates'] == parent['ranked_candidates']
    assert result['metadata_shard_checksums'] == parent['metadata_shard_checksums']
    ensure_json(child / 'data' / 'frozen_selection.json', result)
    return result, amendment


def cached_assets(root, parent):
    hashes = {}
    attempts = root / 'data' / 'extraction_attempts.jsonl'
    if attempts.exists():
        for line in attempts.read_text().splitlines():
            for row in json.loads(line).get('partial_records', []):
                if 'source_sha256' in row and 'mask_sha256' in row:
                    hashes[row['img_id']] = row
    status = root / 'data' / 'extraction_status.json'
    if status.exists():
        for row in BASE.load(status).get('partial_records', []):
            if 'source_sha256' in row and 'mask_sha256' in row:
                hashes[row['img_id']] = row
    cache = {}
    for selected in parent['selections']['train8']:
        image_id = selected['img_id']
        source = root / 'inputs' / 'train8' / image_id / 'source.original'
        mask = root / 'inputs' / 'train8' / image_id / 'mask.original'
        previous = hashes.get(image_id)
        if previous and source.exists() and mask.exists():
            raw_source, raw_mask = source.read_bytes(), mask.read_bytes()
            if BASE.sha256(raw_source) != previous['source_sha256'] or BASE.sha256(raw_mask) != previous['mask_sha256']:
                raise RuntimeError(f'Original v1 cached bytes changed for {image_id}')
            cache[image_id] = dict(source_img=dict(bytes=raw_source, path=None),
                mask_img=dict(bytes=raw_mask, path=None))
    return cache


class CachedOrRemoteParquet:
    def __init__(self, handle, root, child, selections, cache):
        self.handle, self.root, self.child = handle, root, child
        self.selections, self.cache = selections, cache

    def read_row_group(self, group, columns, use_threads=False):
        wanted = [row for row in self.selections if row['shard_path'] == self.handle.shard and row['row_group'] == group]
        if wanted and all(row['img_id'] in self.cache for row in wanted):
            shard = BASE.load(self.root / 'data' / 'metadata_shards' / (Path(self.handle.shard).name + '.json'))
            rows = [dict(img_id=row['original_img_id'], turn_index=row['turn_index'], instruction=row['instruction'],
                source_img=None, mask_img=None) for row in shard['rows'] if row['row_group'] == group]
            for row in wanted:
                rows[row['row_in_group']].update(self.cache[row['img_id']])
                BASE.append_json(self.child / 'data' / 'cache_reuse.jsonl', dict(time=BASE.now(), img_id=row['img_id'],
                    source='original failed-v1 attempt bytes with saved SHA256 verified',
                    source_sha256=BASE.sha256(self.cache[row['img_id']]['source_img']['bytes']),
                    mask_sha256=BASE.sha256(self.cache[row['img_id']]['mask_img']['bytes']),
                    new_network_bytes=0, target_column_read=False))
            return pa.Table.from_pylist(rows).select(columns)
        return ORIGINAL_OPEN(self.handle).read_row_group(group, columns=columns, use_threads=use_threads)


def restore_frozen_order(records, chosen):
    expected = [row['img_id'] for row in chosen]
    actual = [row['img_id'] for row in records]
    if len(set(expected)) != len(expected) or len(set(actual)) != len(actual) or set(actual) != set(expected):
        raise RuntimeError('Extracted IDs must match the unique frozen IDs exactly before restoring rank order')
    by_id = {row['img_id']: row for row in records}
    restored = [by_id[image_id] for image_id in expected]
    assert [row['img_id'] for row in restored] == expected
    return restored


def export_root(root, child, parent, amendment, report):
    effective_sha = BASE.sha256((child / 'data' / 'frozen_selection.json').read_bytes())
    sidecar_sha = BASE.sha256((child / 'data' / 'decoder_amendment.json').read_bytes())
    records = copy.deepcopy(restore_frozen_order(report['train8_records'], parent['selections']['train8']))
    for row in records:
        for field in ('image', 'original_mask', 'binary_mask'):
            row[field] = str(Path('preprocess_alpha_v2') / row[field])
        row.update(parent_frozen_selection_sha256=amendment['parent_frozen_selection_sha256'],
            effective_frozen_selection_sha256=effective_sha,
            decoder_amendment='preprocess_alpha_v2/data/decoder_amendment.json', decoder_amendment_sha256=sidecar_sha)
    exported = copy.deepcopy(report)
    exported.update(train8_records=records, parent_frozen_selection_sha256=amendment['parent_frozen_selection_sha256'],
        effective_frozen_selection_sha256=effective_sha, decoder_amendment='preprocess_alpha_v2/data/decoder_amendment.json',
        decoder_amendment_sha256=sidecar_sha, wrapper_script_sha256=BASE.sha256(HERE.read_bytes()),
        decoder_version='alpha_v2', original_failure_records_preserved=True)
    ensure_json(root / 'data' / 'train8_assets.json', exported)
    cases = dict(dataset=BASE.REPO, revision=BASE.REVISION, seed=BASE.SEED,
        split='official_train_first_turn_train8', selection_sha256=amendment['parent_frozen_selection_sha256'],
        parent_frozen_selection_sha256=amendment['parent_frozen_selection_sha256'],
        effective_frozen_selection_sha256=effective_sha, ground_truth_target_used=False,
        decoder_amendment='preprocess_alpha_v2/data/decoder_amendment.json', decoder_amendment_sha256=sidecar_sha,
        ready_for_calibration=report['ready_for_calibration'], cases=records)
    ensure_json(root / 'cases_train8.json', cases)
    return exported


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extract-train8', action='store_true', required=True)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--old-cases', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    parent = BASE.load(root / 'data' / 'frozen_selection.json')
    if not parent['metadata_complete'] or parent['contract']['revision'] != BASE.REVISION:
        raise RuntimeError('A completed original pinned frozen selection is required')
    child = root / 'preprocess_alpha_v2'
    effective, amendment = effective_selection(root, parent, child)
    inventory = BASE.load(root / 'data' / 'repo_inventory.json')
    if inventory['files_sha256'] != parent['inventory_files_sha256']:
        raise RuntimeError('Original inventory does not match the frozen selection')
    ensure_json(child / 'data' / 'repo_inventory.json', inventory)
    cache = cached_assets(root, parent)
    BASE.contract = lambda: effective['contract']
    BASE.decode_geometry = decode_geometry
    BASE.open_parquet = lambda handle: CachedOrRemoteParquet(handle, root, child, effective['selections']['train8'], cache)
    BASE.__file__ = str(HERE)
    old_cases = args.old_cases or root.parent / 'Lightning8前置实验_20261022' / 'cases.json'
    try:
        report = BASE.extract_train8(child, effective, inventory, old_cases)
    except Exception:
        path = child / 'data' / 'train8_assets.json'
        if path.exists():
            report = BASE.load(path)
            if len(report['train8_records']) == 8:
                export_root(root, child, parent, amendment, report)
        raise
    exported = export_root(root, child, parent, amendment, report)
    print(json.dumps(dict(status=exported['status'], decoder_version='alpha_v2',
        ready_for_calibration=exported['ready_for_calibration'], same_train8_ids=True,
        parent_frozen_selection_sha256=amendment['parent_frozen_selection_sha256'],
        model_calls=0, dit_forwards=0, target_images_read=False), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
