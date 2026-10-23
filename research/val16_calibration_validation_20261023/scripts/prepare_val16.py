import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent.parent


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ensure(base, path, value):
    if path.exists():
        if load(path) != value:
            raise RuntimeError(f'Frozen asset contract differs: {path}')
    else:
        base.frozen_json(path, value)


def setup(args):
    source = args.source_stage.resolve()
    module_path = source / 'scripts' / 'prepare_magicbrush_v2.py'
    spec = importlib.util.spec_from_file_location('val16_alpha_v2_decoder', module_path)
    decoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(decoder)
    base = decoder.BASE
    parent_path = source / 'data' / 'frozen_selection.json'
    parent = load(parent_path)
    stable = {key: value for key, value in parent.items() if key not in ('created_at', 'script_sha256', 'selection_payload_sha256')}
    assert base.sha256(base.canonical(stable)) == parent['selection_payload_sha256']
    assert parent['contract']['revision'] == base.REVISION and parent['metadata_complete']
    chosen = parent['selections']['val16']
    assert len(chosen) == 16 and len({row['img_id'] for row in chosen}) == 16
    assert all(row['official_split'] == 'dev' and row['turn_index'] == 1 for row in chosen)
    assert not ({row['img_id'] for row in chosen} & {row['img_id'] for row in parent['selections']['train32']})
    assert not ({row['img_id'] for row in chosen} & {row['img_id'] for row in parent['selections']['test64']})
    inventory_path = source / 'data' / 'repo_inventory.json'
    inventory = load(inventory_path)
    assert inventory['revision'] == base.REVISION
    assert base.sha256(base.canonical(inventory['files'])) == inventory['files_sha256'] == parent['inventory_files_sha256']
    old = base.old_signatures(args.old_cases.resolve())
    references = {f'old_demo/{key}': row['signature'] for key, row in old['signatures'].items()}
    train_path = source / 'data' / 'train8_assets.json'
    train = load(train_path)
    assert train['ready_for_calibration'] and len(train['train8_records']) == 8
    assert [row['img_id'] for row in train['train8_records']] == [row['img_id'] for row in parent['selections']['train8']]
    for row in train['train8_records']:
        raw = (source / row['image']).read_bytes()
        assert base.sha256(raw) == row['source_sha256']
        with Image.open(io.BytesIO(raw)) as image:
            signature = base.image_signature(image, raw)
        assert signature == row['signature']
        references['train8/' + row['img_id']] = signature
    root = args.root.resolve()
    frozen = dict(scope='frozen official-dev Val16 source/mask asset extraction and isolation QC only',
        repo_id=base.REPO, revision=base.REVISION, selection_seed=parent['contract']['seed'],
        parent_frozen_selection_sha256=sha(parent_path), parent_selection_payload_sha256=parent['selection_payload_sha256'],
        parent_inventory_sha256=sha(inventory_path), parent_inventory_files_sha256=inventory['files_sha256'],
        parent_train8_assets_sha256=sha(train_path), selections=chosen,
        decoder_path=str(module_path), decoder_sha256=sha(module_path),
        preserved_prepare_v1_sha256=sha(source / 'scripts' / 'prepare_magicbrush.py'),
        extraction_script_sha256=sha(Path(__file__)),
        geometry_contract=decoder.effective_contract(parent)['geometry'],
        duplicate_contract=parent['contract']['duplicate_check'], old_demo_reference=old,
        reference_signatures=references, reference_count=len(references),
        dev_source_mask_read=True, train8_assets_modified=False, test_media_read=False, target_columns_read=False,
        replacement_performed=False, model_calls=0, dit_forwards=0, training_steps=0)
    assert len(references) == 11
    ensure(base, root / 'data' / 'asset_preregistration.json', frozen)
    base.atomic_json(root / 'data' / 'asset_environment.json', dict(time=base.now(), python=sys.version,
        executable=sys.executable, host=os.uname().nodename,
        packages={name: importlib.metadata.version(name) for name in ['numpy', 'Pillow', 'pyarrow', 'fsspec', 'huggingface_hub']},
        https_proxy_configured=bool(os.environ.get('https_proxy') or os.environ.get('HTTPS_PROXY')),
        gpu_used=False, model_imported=False))
    return root, source, decoder, base, chosen, inventory, frozen


def get_assets(root, base, chosen, inventory):
    shard_map = {row['path']: row for row in inventory['files']}
    grouped = defaultdict(list)
    records = {}
    for row in chosen:
        image_id = row['img_id']
        entry_path = root / 'data' / 'asset_bytes' / (image_id + '.json')
        if entry_path.exists():
            entry = load(entry_path)
            assert entry['locator'] == row
            assert sha(root / entry['image']) == entry['source_sha256']
            assert sha(root / entry['original_mask']) == entry['mask_sha256']
            records[image_id] = entry
        else:
            grouped[(row['shard_path'], row['row_group'])].append(row)
    for (path, group), wanted in sorted(grouped.items()):
        shard = shard_map[path]
        with base.AuditRangeFile(shard['source_url'], shard['size_bytes'], root / 'data' / 'network_ranges.jsonl',
                'val16_source_mask_extraction', path) as handle:
            parquet = base.open_parquet(handle)
            table = parquet.read_row_group(group, columns=base.ASSET_COLUMNS, use_threads=False)
            assert table.column_names == base.ASSET_COLUMNS and 'target_img' not in table.column_names
            for row in wanted:
                selected = table.slice(row['row_in_group'], 1).to_pylist()[0]
                assert str(selected['img_id']) == row['original_img_id']
                assert selected['turn_index'] == 1 and selected['instruction'] == row['instruction']
                raw_source, raw_mask = selected['source_img']['bytes'], selected['mask_img']['bytes']
                assert isinstance(raw_source, bytes) and isinstance(raw_mask, bytes)
                prefix = Path('inputs') / 'val16' / row['img_id']
                image_path, mask_path = prefix / 'source.original', prefix / 'mask.original'
                base.immutable_bytes(root / image_path, raw_source)
                base.immutable_bytes(root / mask_path, raw_mask)
                entry = dict(locator=row, image=str(image_path), original_mask=str(mask_path),
                    source_sha256=base.sha256(raw_source), mask_sha256=base.sha256(raw_mask),
                    source_bytes=len(raw_source), mask_bytes=len(raw_mask), target_column_read=False,
                    columns_read=base.ASSET_COLUMNS, source_url=shard['source_url'], source_url_sha256=shard['source_url_sha256'])
                ensure(base, root / 'data' / 'asset_bytes' / (row['img_id'] + '.json'), entry)
                records[row['img_id']] = entry
                print(json.dumps(dict(phase='val16_bytes', img_id=row['img_id'], completed=len(records), total=16)), flush=True)
    assert len(records) == 16
    return [records[row['img_id']] for row in chosen]


def contact_sheet(root, records):
    canvas = Image.new('RGB', (1600, 1760), 'white')
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(records):
        with Image.open(root / row['image']) as image:
            preview = image.convert('RGB')
        preview.thumbnail((390, 390), Image.Resampling.LANCZOS)
        x, y = (index % 4) * 400 + 5, (index // 4) * 440 + 5
        canvas.paste(preview, (x, y))
        if 'source_bbox_xyxy_float' in row:
            box = row['source_bbox_xyxy_float']
            sx, sy = preview.width / row['source_wh'][0], preview.height / row['source_wh'][1]
            draw.rectangle((x+box[0]*sx, y+box[1]*sy, x+box[2]*sx, y+box[3]*sy), outline='red', width=3)
        draw.text((x, y+395), f'{index+1} ID {row["img_id"]} {row["status"]}', fill='black')
        if 'target_active_fraction' in row:
            draw.text((x, y+413), f'core fraction {row["target_active_fraction"]:.4f}', fill='black')
    path = root / 'figures' / 'val16_source_roi_contact.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return str(path.relative_to(root))


def quality_control(root, decoder, base, chosen, assets, frozen):
    references = copy.deepcopy(frozen['reference_signatures'])
    records = []
    failures, gray = [], []
    for selected, asset in zip(chosen, assets):
        row = dict(selected, **{key: value for key, value in asset.items() if key != 'locator'},
            case_id='mb_val_' + selected['img_id'], prompt=selected['instruction'], status='PASS')
        raw_source = (root / row['image']).read_bytes()
        raw_mask = (root / row['original_mask']).read_bytes()
        try:
            source, geometry, binary = decoder.decode_geometry(raw_source, raw_mask)
            row.update(geometry)
            signature = base.image_signature(source, raw_source)
            comparisons = [base.duplicate_comparison(signature, reference, reference_id) for reference_id, reference in references.items()]
            row.update(signature=signature, duplicate_comparisons=comparisons)
            rejected = [entry for entry in comparisons if entry['verdict'] == 'REJECT']
            questionable = [entry for entry in comparisons if entry['gray_zone']]
            if rejected:
                row['status'] = 'REJECTED_SOURCE_DUPLICATE'
                failures.append(dict(img_id=row['img_id'], reason='source_duplicate', evidence=rejected))
            elif questionable:
                row['status'] = 'SOURCE_REVIEW_REQUIRED'
                gray.append(dict(img_id=row['img_id'], comparisons=questionable))
            references['val16/' + row['img_id']] = signature
            path = Path('inputs') / 'val16' / row['img_id'] / 'mask.target.binary.png'
            base.immutable_bytes(root / path, binary)
            row.update(binary_mask=str(path), binary_mask_sha256=base.sha256(binary))
        except Exception as error:
            row.update(status='GEOMETRY_FAILED', error_type=type(error).__name__, error=str(error))
            failures.append(dict(img_id=row['img_id'], reason='geometry', error=str(error), traceback=traceback.format_exc()))
        records.append(row)
        print(json.dumps(dict(phase='val16_qc', img_id=row['img_id'], status=row['status'],
            core_fraction=row.get('target_active_fraction'))), flush=True)
    assert [row['img_id'] for row in records] == [row['img_id'] for row in chosen]
    figure = contact_sheet(root, records)
    ready = not failures and not gray
    report = dict(status='PASS' if ready else ('BLOCKED_QC' if failures else 'SOURCE_REVIEW_REQUIRED'),
        created_at=base.now(), ready_for_validation=ready, frozen_source_selection_sha256=frozen['parent_frozen_selection_sha256'],
        asset_preregistration_sha256=sha(root / 'data' / 'asset_preregistration.json'),
        val16_records=records, frozen_ids=[row['img_id'] for row in chosen], asset_count=16,
        geometry_pass_count=sum('indices' in row for row in records), hard_failures=failures, gray_zones=gray,
        initial_reference_count=11, within_val16_pair_checks=120,
        duplicate_rule=frozen['duplicate_contract'], contact_sheet=figure,
        model_calls=0, dit_forwards=0, training_steps=0, target_columns_read=False,
        test_media_read=False, dev_source_mask_read=True, train8_assets_modified=False,
        network=base.network_summary(root), replacement_performed=False)
    ensure(base, root / 'data' / 'val16_assets.json', report)
    cases = dict(dataset=base.REPO, revision=base.REVISION, seed=frozen['selection_seed'],
        split='official_dev_frozen_val16_first_turn',
        parent_frozen_selection_sha256=frozen['parent_frozen_selection_sha256'],
        asset_preregistration_sha256=report['asset_preregistration_sha256'],
        ready_for_validation=ready, target_images_used=False, train8_assets_modified=False, cases=records)
    ensure(base, root / 'cases_val16.json', cases)
    lines = ['# Val16 资产与隔离 QC', '', f'状态：{report["status"]}；冻结16张，实际原始源图/掩码16组；几何通过{report["geometry_pass_count"]}组。',
        '', '源图和掩码按固定 revision 的 Parquet 列投影读取；未读取目标图数据列、未读取 Test 图像、未运行 GPU 或拟合。',
        '', f'冻结 ID 顺序：{", ".join(report["frozen_ids"])}。', '',
        f'旧3demo+Train8共11个参照，外加Val16内部120对；硬失败{len(failures)}，近重复灰区{len(gray)}。阈值与父阶段一致，未依据生成质量选样，未替补ID。',
        '', 'RGBA以alpha==0表示明确编辑区域，alpha<255作为本研究纳入抗锯齿边缘的保守ROI；bbox加2token halo。几何通过不保证生成质量或无损。', '',
        '| 序号 | img_id | QC | core/target |', '|---:|---|---|---:|']
    lines.extend(f'| {index+1} | {row["img_id"]} | {row["status"]} | {row.get("target_active_fraction", "N/A")} |' for index, row in enumerate(records))
    (root / '01_Val16资产QC.md').write_text('\n'.join(lines) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-stage', type=Path, required=True)
    parser.add_argument('--old-cases', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    root, source, decoder, base, chosen, inventory, frozen = setup(args)
    destination = root / 'data' / 'val16_assets.json'
    if destination.exists():
        report = load(destination)
        assert report['asset_preregistration_sha256'] == sha(root / 'data' / 'asset_preregistration.json')
        for row in report['val16_records']:
            assert sha(root / row['image']) == row['source_sha256']
            assert sha(root / row['original_mask']) == row['mask_sha256']
            if row.get('binary_mask'):
                assert sha(root / row['binary_mask']) == row['binary_mask_sha256']
        print(json.dumps(dict(status=report['status'], reused=True, asset_count=16)), flush=True)
    else:
        try:
            assets = get_assets(root, base, chosen, inventory)
            report = quality_control(root, decoder, base, chosen, assets, frozen)
        except Exception as error:
            base.append_json(root / 'data' / 'asset_failures.jsonl', dict(time=base.now(), error=str(error),
                traceback=traceback.format_exc(), model_calls=0, target_columns_read=False))
            raise
    print(json.dumps(dict(status=report['status'], ready_for_validation=report['ready_for_validation'],
        asset_count=16, geometry_pass_count=report['geometry_pass_count'],
        hard_failures=len(report['hard_failures']), gray_zones=len(report['gray_zones']),
        cases=str(root / 'cases_val16.json'), model_calls=0, dit_forwards=0)), flush=True)
    assert not report['hard_failures'], 'Frozen Val16 has hard source/geometry QC failures; no replacement permitted'


if __name__ == '__main__':
    main()
