import argparse
import copy
import hashlib
import io
import json
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

import prepare_magicbrush_v2 as subject


def png(value, mode=None):
    handle = io.BytesIO()
    Image.fromarray(value, mode=mode).save(handle, format='PNG')
    return handle.getvalue()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=subject.ROOT / 'data' / 'geometry_alpha_v2_checks.json')
    args = parser.parse_args()
    report = dict(status='RUNNING', scope='synthetic CPU PNG/geometry and contract checks only; no network or model',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        wrapper_sha256=hashlib.sha256(subject.HERE.read_bytes()).hexdigest(),
        model_calls=0, dit_forwards=0, training_steps=0, network_requests=0, checks=[])
    yy, xx = np.indices((64, 64))
    rgb = np.stack([(xx * 3) % 256, (yy * 4) % 256, (xx + yy) % 256], axis=-1).astype(np.uint8)
    source = png(rgb, 'RGB')
    alpha = np.full((64, 64), 255, dtype=np.uint8)
    alpha[10:20, 20:30] = 0
    alpha[9, 20] = 254
    rgba = np.concatenate([rgb, alpha[..., None]], axis=-1)
    mask = png(rgba, 'RGBA')

    def check(name, callback):
        try:
            evidence = callback()
            report['checks'].append(dict(name=name, status='PASS', evidence=evidence))
        except Exception as error:
            report['checks'].append(dict(name=name, status='FAIL', error_type=type(error).__name__,
                error=str(error), traceback=traceback.format_exc()))

    def alpha_edges():
        _, geometry, binary = subject.decode_geometry(source, mask)
        array = np.asarray(Image.open(io.BytesIO(binary)))
        assert geometry['alpha_zero_official_edit_pixels'] == 100
        assert geometry['alpha_antialias_edge_pixels'] == 1
        assert geometry['alpha_lt255_conservative_roi_pixels'] == 101
        assert geometry['raw_mask_bbox_xyxy_exclusive'] == [20, 9, 30, 20]
        assert geometry['alpha_zero_bbox_xyxy_exclusive'] == [20, 10, 30, 20]
        assert geometry['target_mask_bbox_xyxy_exclusive'] == [320, 144, 480, 320]
        assert geometry['core_token_bbox_xyxy_exclusive'] == [18, 7, 32, 22]
        assert array[144, 320] == 255 and array[143, 320] == 0
        assert geometry['rgb_channels_used_for_edit_mask'] is False
        return {key: geometry[key] for key in ('alpha_zero_official_edit_pixels', 'alpha_antialias_edge_pixels',
            'alpha_lt255_conservative_roi_pixels', 'core_token_bbox_xyxy_exclusive')}

    def rgb_ignored():
        changed = rgba.copy()
        changed[:, :, :3] = 255
        _, before, old_binary = subject.decode_geometry(source, mask)
        _, after, new_binary = subject.decode_geometry(source, png(changed, 'RGBA'))
        assert old_binary == new_binary
        assert before['indices'] == after['indices']
        return dict(binary_equal=True, indices_equal=True, arbitrary_non_grayscale_rgb_canvas_supported=True)

    def opaque_compatible():
        gray = (alpha < 255).astype(np.uint8) * 255
        _, old, old_binary = subject.ORIGINAL_DECODE(source, png(gray, 'L'))
        _, opaque, binary = subject.decode_geometry(source, png(gray, 'L'))
        _, rgba_geometry, rgba_binary = subject.decode_geometry(source, mask)
        for key in ('raw_mask_bbox_xyxy_exclusive', 'target_mask_bbox_xyxy_exclusive',
                'core_token_bbox_xyxy_exclusive', 'target_hw', 'target_grid_hw', 'indices'):
            assert old[key] == opaque[key] == rgba_geometry[key]
        assert old_binary == binary == rgba_binary
        assert opaque['mask_encoding'] == 'opaque grayscale white-positive mask'
        return dict(opaque_grayscale_compatible=True, coordinate_bbox_and_indices_unchanged=True)

    def rejects(mask_bytes, source_bytes=source):
        try:
            subject.decode_geometry(source_bytes, mask_bytes)
        except ValueError as error:
            return dict(rejected=True, reason=str(error))
        raise AssertionError('Expected ValueError was not raised')

    def contract_only_geometry():
        contract = subject.BASE.contract()
        parent = dict(contract=copy.deepcopy(contract), selections=dict(train8=[dict(img_id='1')]),
            ranked_candidates={'train': ['1']}, metadata_shard_checksums={'one': 'sha'})
        frozen_before = json.dumps(parent, sort_keys=True)
        effective = subject.effective_contract(parent)
        assert json.dumps(parent, sort_keys=True) == frozen_before
        original_no_geometry = {key: value for key, value in contract.items() if key != 'geometry'}
        effective_no_geometry = {key: value for key, value in effective.items() if key != 'geometry'}
        assert original_no_geometry == effective_no_geometry
        for key in ('target_area', 'dimension_rounding', 'image_token_pixels', 'halo_tokens',
                'aspect_log_distance_max', 'mask_resampling', 'semantic_boundary_learned'):
            assert contract['geometry'][key] == effective['geometry'][key]
        return dict(parent_unchanged=True, only_geometry_contract_changes=True)

    def rank_order():
        frozen_ids = ['405994', '301232', '544811', '253960', '315467', '355272', '257102', '271359']
        chosen = [dict(img_id=image_id) for image_id in frozen_ids]
        records = [dict(img_id=image_id, shard=f'shard-{(index + 2) % 3}', row_group=7-index)
            for index, image_id in enumerate(frozen_ids)]
        records.sort(key=lambda row: (row['shard'], row['row_group']))
        before = copy.deepcopy(records)
        restored = subject.restore_frozen_order(records, chosen)
        assert [row['img_id'] for row in restored] == frozen_ids
        assert records == before
        for invalid in (records[:-1], records + [records[0]]):
            try:
                subject.restore_frozen_order(invalid, chosen)
            except RuntimeError:
                pass
            else:
                raise AssertionError('Missing or duplicate IDs must be rejected')
        return dict(multiple_shard_IO_order_restored_to_frozen_rank=True, original_input_not_mutated=True,
            missing_and_duplicate_IDs_rejected=True, frozen_ids=frozen_ids)

    check('alpha_zero_and_alpha254_edge_included_with_exact_bbox', alpha_edges)
    check('RGBA_RGB_canvas_never_used_as_grayscale_mask', rgb_ignored)
    check('opaque_grayscale_and_existing_coordinate_mapping_compatible', opaque_compatible)
    check('empty_grayscale_rejected', lambda: rejects(png(np.zeros((64, 64), dtype=np.uint8), 'L')))
    check('all_positive_grayscale_rejected', lambda: rejects(png(np.full((64, 64), 255, dtype=np.uint8), 'L')))
    full_alpha = rgba.copy()
    full_alpha[:, :, 3] = 0
    check('all_transparent_full_ROI_rejected', lambda: rejects(png(full_alpha, 'RGBA')))
    opaque_rgb = rgba.copy()
    opaque_rgb[:, :, 3] = 255
    check('opaque_color_source_canvas_rejected_as_ambiguous_mask', lambda: rejects(png(opaque_rgb, 'RGBA')))
    half_alpha = rgba.copy()
    half_alpha[:, :, 3] = 255
    half_alpha[10:20, 20:30, 3] = 254
    check('partial_alpha_without_alpha0_rejected_as_ambiguous', lambda: rejects(png(half_alpha, 'RGBA')))
    nonsquare_source = png(rgb[:32], 'RGB')
    check('source_mask_aspect_mismatch_rejected', lambda: rejects(mask, nonsquare_source))
    check('effective_contract_keeps_ID_policy_and_mapping', contract_only_geometry)
    check('multiple_shard_IO_order_restored_without_ID_changes', rank_order)
    report['status'] = 'PASS' if all(row['status'] == 'PASS' for row in report['checks']) else 'FAIL'
    report['checks_passed'] = sum(row['status'] == 'PASS' for row in report['checks'])
    report['checks_total'] = len(report['checks'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(status=report['status'], checks=report['checks_total'], passed=report['checks_passed'],
        output=str(args.output), model_calls=0, network_requests=0), ensure_ascii=False), flush=True)
    assert report['status'] == 'PASS', 'alpha_v2 CPU geometry checks failed'


if __name__ == '__main__':
    main()
