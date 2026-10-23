import argparse
import hashlib
import io
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from fsspec.spec import AbstractBufferedFile
from huggingface_hub import HfApi, hf_hub_url
from PIL import Image


REPO = 'osunlp/MagicBrush'
REVISION = '6c254560879df7851233ea0fd2ae2a62764d0ef2'
SEED = 2026102303
SPLITS = ('train', 'dev', 'test')
COUNTS = {'train': 32, 'dev': 16, 'test': 64}
ROWS = {'train': 8807, 'dev': 528, 'test': 1053}
SHARDS = {'train': 51, 'dev': 4, 'test': 6}
META_COLUMNS = ['img_id', 'turn_index', 'instruction']
ASSET_COLUMNS = META_COLUMNS + ['source_img', 'mask_img']
ROOT = Path(__file__).resolve().parent.parent


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()


def now():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)


def frozen_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2).encode() + b'\n'
    temporary = path.with_name(path.name + f'.{os.getpid()}.frozen.tmp')
    try:
        with temporary.open('wb') as handle:
            handle.write(payload)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def append_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + '\n')


def contract():
    return dict(schema_version=1, repo_id=REPO, revision=REVISION, seed=SEED,
        requested_counts=COUNTS, selection_turn_index=1,
        cross_split_rule='exclude every img_id occurring in more than one official split; never move groups',
        duplicate_first_turn_rule='exclude img_id with other than one turn_index=1 row within its split',
        ranking='sha256(UTF8(seed|revision|official_split|normalized_coco_img_id)), ascending hexadecimal, then img_id',
        train_nested_prefixes=[8, 16, 32], target_columns_read=False,
        source_extraction_scope='only frozen Train8, no Test or dev media; other Train32 sources remain locked locators',
        duplicate_check=dict(exact='raw bytes SHA256 or dimensions plus decoded RGB SHA256',
            hash_bits=64, reject_both_hash_distance_max=6, reject_aspect_log_distance_max=0.02,
            gray_either_hash_distance_max=10, image_only=True),
        geometry=dict(target_area=1024 * 1024, dimension_rounding='Python round to nearest multiple of 32, same as Qwen calculate_dimensions',
            image_token_pixels=16, roi='bbox of positive visible white mask, expanded by 2 image tokens in every direction',
            halo_tokens=2, aspect_log_distance_max=0.01,
            mask_resampling='NEAREST to generated target canvas, only if source and mask aspect agree',
            alpha='reject any mask with nonopaque alpha; no inferred alpha convention',
            semantic_boundary_learned=False),
        post_freeze_rejection='stop and preserve frozen IDs; no silent replacement or overwriting')


class AuditRangeFile(AbstractBufferedFile):
    def __init__(self, url, size, audit_path, phase, shard, retries=3):
        self.source_url = url
        self.audit_path = Path(audit_path)
        self.phase = phase
        self.shard = shard
        self.retries = retries
        self.target_ranges = []
        self.object_etag = None
        super().__init__(None, url, mode='rb', block_size=1024 * 1024,
            cache_type='none', size=int(size))
        self.details = dict(name=url, size=int(size), type='file')

    def _fetch_range(self, start, end):
        start, end = int(start), min(int(end), self.size)
        if start >= end:
            return b''
        wanted = f'bytes {start}-{end - 1}/{self.size}'
        if any(start < upper and end > lower for lower, upper in self.target_ranges):
            append_json(self.audit_path, dict(time=now(), phase=self.phase, shard=self.shard,
                source_url=self.source_url, status='blocked_before_network',
                start_inclusive=start, end_exclusive=end, reason='Range overlaps target_img column chunk'))
            raise RuntimeError('Requested byte range overlaps target_img payload; refusing before network request')
        for attempt in range(self.retries):
            started = time.perf_counter()
            record = dict(time=now(), phase=self.phase, shard=self.shard,
                source_url=self.source_url, source_url_sha256=sha256(self.source_url.encode()),
                start_inclusive=start, end_exclusive=end, requested_bytes=end - start,
                attempt_1based=attempt + 1, target_column_requested=False)
            try:
                request = urllib.request.Request(self.source_url,
                    headers={'Range': f'bytes={start}-{end - 1}', 'Accept-Encoding': 'identity',
                        'User-Agent': 'MagicBrush-pinned-source-only-research/1.0'})
                with urllib.request.urlopen(request, timeout=180) as response:
                    record.update(status=response.status, content_range=response.headers.get('Content-Range'),
                        content_length=response.headers.get('Content-Length'),
                        content_encoding=response.headers.get('Content-Encoding'),
                        etag=response.headers.get('ETag'),
                        final_host=urllib.parse.urlsplit(response.url).hostname)
                    if response.status != 206 or record['content_range'] != wanted:
                        raise RuntimeError(f'HTTP Range must be exact 206 with {wanted}; received {record}')
                    if record['content_encoding'] not in (None, '', 'identity'):
                        raise RuntimeError('Encoded Range response is not allowed')
                    if record['etag']:
                        if self.object_etag is not None and self.object_etag != record['etag']:
                            raise RuntimeError('Range responses changed ETag within the same pinned object')
                        self.object_etag = record['etag']
                    value = response.read(end - start + 1)
                if len(value) != end - start:
                    raise RuntimeError(f'Range body length mismatch: {len(value)} != {end-start}')
                record.update(status='complete', response_http_status=206, received_bytes=len(value),
                    body_sha256=sha256(value), seconds=time.perf_counter() - started)
                append_json(self.audit_path, record)
                return value
            except Exception as error:
                record.update(status='failed', error_type=type(error).__name__, error=str(error),
                    seconds=time.perf_counter() - started)
                append_json(self.audit_path, record)
                if isinstance(error, RuntimeError) or attempt + 1 == self.retries:
                    raise
                time.sleep(min(2 ** attempt, 4))
        raise RuntimeError('Unreachable Range retry state')


def open_parquet(handle):
    handle.seek(handle.size - 8)
    tail = handle.read(8)
    if len(tail) != 8 or tail[4:] != b'PAR1':
        raise RuntimeError('Not an ordinary Parquet object with PAR1 footer')
    footer_length = int.from_bytes(tail[:4], 'little')
    footer_start = handle.size - 8 - footer_length
    if footer_start < 4:
        raise RuntimeError('Invalid Parquet footer length')
    handle.seek(footer_start)
    footer = handle.read(footer_length)
    metadata = pq.read_metadata(io.BytesIO(b'PAR1' + footer + tail))
    target_ranges = []
    for group in range(metadata.num_row_groups):
        row_group = metadata.row_group(group)
        for column in range(row_group.num_columns):
            chunk = row_group.column(column)
            if chunk.path_in_schema == 'target_img' or chunk.path_in_schema.startswith('target_img.'):
                offsets = [value for value in (chunk.dictionary_page_offset, chunk.data_page_offset) if value is not None and value >= 4]
                if not offsets:
                    raise RuntimeError('Target column chunk has no auditable page offset')
                lower = min(offsets)
                upper = lower + int(chunk.total_compressed_size)
                if upper > footer_start or upper < lower:
                    raise RuntimeError('Target payload boundaries conflict with Parquet footer')
                target_ranges.append((lower, upper))
    if not target_ranges:
        raise RuntimeError('Expected target_img column chunks were not found')
    handle.target_ranges = target_ranges
    audit = dict(time=now(), phase=handle.phase, shard=handle.shard,
        footer_start_inclusive=footer_start, footer_length=footer_length,
        footer_sha256=sha256(footer), footer_tail_sha256=sha256(tail),
        target_payload_ranges_exclusive=[list(value) for value in target_ranges],
        rule='all future HTTP payload ranges must avoid these target_img column chunks',
        metadata_footer_is_not_image_payload=True)
    append_json(handle.audit_path.parent / 'parquet_layouts.jsonl', audit)
    return pq.ParquetFile(handle, metadata=metadata, pre_buffer=False)


def network_summary(root):
    path = root / 'data' / 'network_ranges.jsonl'
    if not path.exists():
        return dict(requests=0, received_bytes=0)
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    complete = [row for row in events if row['status'] == 'complete']
    return dict(audit_file=str(path.relative_to(root)), audit_file_sha256=sha256(path.read_bytes()),
        complete_range_requests=len(complete), failed_range_requests=sum(row['status'] == 'failed' for row in events),
        blocked_range_requests=sum(row['status'] == 'blocked_before_network' for row in events),
        received_bytes=sum(row['received_bytes'] for row in complete),
        by_phase={phase: dict(requests=sum(row['phase'] == phase for row in complete),
            received_bytes=sum(row['received_bytes'] for row in complete if row['phase'] == phase))
            for phase in sorted({row['phase'] for row in complete})},
        scope='actual successful Parquet range bodies; excludes Hub JSON/tree requests and failed-body bytes')


def repo_inventory(root):
    destination = root / 'data' / 'repo_inventory.json'
    if destination.exists():
        saved = load(destination)
        assert saved['revision'] == REVISION and saved['repo_id'] == REPO
        assert saved['files_sha256'] == sha256(canonical(saved['files']))
        return saved
    started = time.perf_counter()
    files = []
    api = HfApi()
    for entry in api.list_repo_tree(REPO, path_in_repo='data', recursive=True,
            expand=False, revision=REVISION, repo_type='dataset'):
        path = entry.path
        match = re.fullmatch(r'data/(train|dev|test)-[0-9]{5}-of-[0-9]{5}(?:-[a-f0-9]+)?\.parquet', path)
        if not match:
            continue
        lfs = getattr(entry, 'lfs', None)
        if lfs is None:
            lfs_sha = None
        elif isinstance(lfs, dict):
            lfs_sha = lfs.get('sha256')
        else:
            lfs_sha = getattr(lfs, 'sha256', None)
        url = hf_hub_url(REPO, path, repo_type='dataset', revision=REVISION)
        files.append(dict(path=path, split=match.group(1), size_bytes=int(entry.size),
            git_blob_id=getattr(entry, 'blob_id', None), lfs_sha256=lfs_sha,
            xet_hash=getattr(entry, 'xet_hash', None), source_url=url,
            source_url_sha256=sha256(url.encode())))
    files.sort(key=lambda row: row['path'])
    assert dict(Counter(row['split'] for row in files)) == SHARDS
    result = dict(repo_id=REPO, revision=REVISION, created_at=now(),
        hub_tree_api=f'https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/data?recursive=true&expand=false',
        api_response_semantics='RepoFile metadata, not image or target bytes; library may paginate',
        files=files, files_sha256=sha256(canonical(files)),
        compressed_parquet_total_bytes=sum(row['size_bytes'] for row in files),
        compressed_parquet_split_bytes={split: sum(row['size_bytes'] for row in files if row['split'] == split) for split in SPLITS},
        seconds=time.perf_counter() - started)
    frozen_json(destination, result)
    return result


def metadata_cache_path(root, shard):
    return root / 'data' / 'metadata_shards' / (Path(shard['path']).name + '.json')


def metadata_shard(root, shard):
    destination = metadata_cache_path(root, shard)
    if destination.exists():
        value = load(destination)
        assert value['revision'] == REVISION and value['shard'] == shard
        assert value['rows_sha256'] == sha256(canonical(value['rows']))
        assert value['columns_read'] == META_COLUMNS
        return value
    started = time.perf_counter()
    rows = []
    with AuditRangeFile(shard['source_url'], shard['size_bytes'], root / 'data' / 'network_ranges.jsonl',
            'metadata_scan', shard['path']) as handle:
        parquet = open_parquet(handle)
        assert set(META_COLUMNS + ['source_img', 'mask_img', 'target_img']).issubset(set(parquet.schema_arrow.names))
        absolute = 0
        groups = []
        for group in range(parquet.num_row_groups):
            table = parquet.read_row_group(group, columns=META_COLUMNS, use_threads=False)
            assert table.column_names == META_COLUMNS
            group_rows = table.to_pylist()
            groups.append(dict(row_group=group, count=len(group_rows), absolute_start=absolute))
            for local, row in enumerate(group_rows):
                raw_id = str(row['img_id'])
                if not raw_id.isdecimal() or not isinstance(row['instruction'], str) or not row['instruction'].strip():
                    raise ValueError(f'Invalid metadata: {shard["path"]}, {group}, {local}')
                rows.append(dict(img_id=str(int(raw_id)), original_img_id=raw_id,
                    turn_index=int(row['turn_index']), instruction=row['instruction'],
                    official_split=shard['split'], shard_path=shard['path'], row_group=group,
                    row_in_group=local, row_in_shard=absolute + local))
            absolute += len(group_rows)
        assert absolute == parquet.metadata.num_rows
    value = dict(revision=REVISION, shard=shard, created_at=now(), columns_read=META_COLUMNS,
        target_column_read=False, image_columns_read=False, row_groups=groups,
        rows=rows, rows_sha256=sha256(canonical(rows)), seconds=time.perf_counter() - started)
    frozen_json(destination, value)
    print(json.dumps(dict(phase='metadata_shard_complete', shard=shard['path'], rows=len(rows)), ensure_ascii=False), flush=True)
    return value


def freeze_selection(root, inventory, metadata):
    expected = contract()
    policy_sha = sha256(canonical(expected))
    destination = root / 'data' / 'frozen_selection.json'
    all_rows = [row for shard in metadata for row in shard['rows']]
    assert dict(Counter(row['official_split'] for row in all_rows)) == ROWS
    split_groups = defaultdict(set)
    for row in all_rows:
        split_groups[row['img_id']].add(row['official_split'])
    conflicts = sorted(key for key, splits in split_groups.items() if len(splits) > 1)
    conflict_set = set(conflicts)
    selected = {}
    ranked = {}
    excluded_first = []
    for split in SPLITS:
        grouped = defaultdict(list)
        for row in all_rows:
            if row['official_split'] == split and row['turn_index'] == 1:
                grouped[row['img_id']].append(row)
        candidates = []
        for image_id, records in grouped.items():
            if image_id in conflict_set:
                continue
            if len(records) != 1:
                excluded_first.append(dict(img_id=image_id, official_split=split,
                    reason='ambiguous first-turn rows', count=len(records)))
                continue
            row = dict(records[0])
            row['selection_rank_sha256'] = sha256(f'{SEED}|{REVISION}|{split}|{image_id}'.encode())
            candidates.append(row)
        candidates.sort(key=lambda row: (row['selection_rank_sha256'], row['img_id']))
        assert len(candidates) >= COUNTS[split]
        ranked[split] = candidates
        selected[split] = candidates[:COUNTS[split]]
    selections = dict(train32=selected['train'], val16=selected['dev'], test64=selected['test'])
    selections.update(train8=selected['train'][:8], train16=selected['train'][:16])
    deterministic = dict(contract=expected, contract_sha256=policy_sha,
        inventory_files_sha256=inventory['files_sha256'], metadata_complete=True,
        metadata_rows=ROWS, metadata_shards=SHARDS,
        metadata_shard_checksums={shard['shard']['path']: shard['rows_sha256'] for shard in metadata},
        cross_split_img_ids_excluded=conflicts, ambiguous_first_turn_groups_excluded=excluded_first,
        selections=selections, ranked_candidates=ranked,
        frozen_test_scope='IDs, instructions and locators only; no test source/mask/target media read',
        content_duplicate_check_scope='not performed during metadata scan; Train8 checked during extraction only',
        ground_truth_target_read=False)
    deterministic['selection_payload_sha256'] = sha256(canonical(deterministic))
    if destination.exists():
        saved = load(destination)
        existing = {key: value for key, value in saved.items() if key not in ('created_at', 'script_sha256')}
        if existing != deterministic:
            raise RuntimeError('Frozen selection differs; use an explicitly new stage directory instead of overwriting')
        return saved
    saved = dict(deterministic, created_at=now(), script_sha256=sha256(Path(__file__).read_bytes()))
    frozen_json(destination, saved)
    atomic_json(root / 'data' / 'metadata_status.json', dict(status='COMPLETE_FROZEN', time=now(),
        frozen_selection_sha256=sha256(destination.read_bytes()), row_counts=ROWS, shard_counts=SHARDS,
        selected_counts={name: len(rows) for name, rows in selections.items()},
        model_calls=0, dit_forwards=0, training_steps=0, target_columns_read=False,
        media_extracted=0, test_media_read=False, network=network_summary(root)))
    return saved


def dimension(width, height):
    ratio = width / height
    target_width = math.sqrt(1024 * 1024 * ratio)
    target_height = target_width / ratio
    return round(target_width / 32) * 32, round(target_height / 32) * 32


def image_signature(image, raw_bytes):
    image = image.convert('RGB')
    gray = image.convert('L')
    average = np.asarray(gray.resize((8, 8), Image.Resampling.LANCZOS), dtype=np.float32)
    difference = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    ahash = ''.join('1' if value else '0' for value in (average >= average.mean()).reshape(-1))
    dhash = ''.join('1' if value else '0' for value in (difference[:, 1:] > difference[:, :-1]).reshape(-1))
    return dict(raw_sha256=sha256(raw_bytes), rgb_sha256=sha256(canonical(list(image.size)) + image.tobytes()),
        width=image.width, height=image.height, average_hash=f'{int(ahash, 2):016x}', difference_hash=f'{int(dhash, 2):016x}')


def duplicate_comparison(signature, reference, reference_id):
    average = (int(signature['average_hash'], 16) ^ int(reference['average_hash'], 16)).bit_count()
    difference = (int(signature['difference_hash'], 16) ^ int(reference['difference_hash'], 16)).bit_count()
    aspect = abs(math.log((signature['width'] / signature['height']) / (reference['width'] / reference['height'])))
    exact = signature['raw_sha256'] == reference['raw_sha256'] or signature['rgb_sha256'] == reference['rgb_sha256']
    near = average <= 6 and difference <= 6 and aspect <= 0.02
    gray = not exact and not near and (average <= 10 or difference <= 10)
    return dict(reference_id=reference_id, exact=exact, near_duplicate=near, gray_zone=gray,
        average_hash_distance=average, difference_hash_distance=difference, aspect_log_distance=aspect,
        verdict='REJECT' if exact or near else ('SOURCE_REVIEW_REQUIRED' if gray else 'PASS'))


def old_signatures(cases_path):
    cases_path = Path(cases_path).resolve()
    cases = load(cases_path)['cases']
    assert len(cases) == 3
    output = {}
    for row in cases:
        path = cases_path.parent / row['image']
        raw = path.read_bytes()
        if row.get('input_sha256') and sha256(raw) != row['input_sha256']:
            raise RuntimeError(f'Old demo bytes changed: {path}')
        with Image.open(io.BytesIO(raw)) as image:
            signature = image_signature(image, raw)
        output[row['case_id']] = dict(signature=signature, path=str(path), source_url=row.get('source_url'))
    return dict(cases_path=str(cases_path), cases_sha256=sha256(cases_path.read_bytes()), signatures=output)


def immutable_bytes(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != value:
            raise RuntimeError(f'Existing asset differs; refusing to replace: {path}')
    else:
        temporary = path.with_name(path.name + f'.{os.getpid()}.asset.tmp')
        try:
            temporary.write_bytes(value)
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def decode_geometry(source_bytes, mask_bytes):
    with Image.open(io.BytesIO(source_bytes)) as handle:
        source_format, source_mode = handle.format, handle.mode
        if handle.getexif().get(274, 1) != 1:
            raise ValueError('Nontrivial source EXIF orientation requires an explicit joint mask transform')
        source = handle.convert('RGB')
    with Image.open(io.BytesIO(mask_bytes)) as handle:
        mask_format, mask_mode = handle.format, handle.mode
        if handle.getexif().get(274, 1) != 1:
            raise ValueError('Nontrivial mask EXIF orientation is unsupported')
        rgba = np.asarray(handle.convert('RGBA'))
    if not np.all(rgba[:, :, 3] == 255):
        raise ValueError('Nonopaque mask alpha has no verified white-region convention')
    if not np.array_equal(rgba[:, :, 0], rgba[:, :, 1]) or not np.array_equal(rgba[:, :, 1], rgba[:, :, 2]):
        raise ValueError('Mask is not grayscale; refusing to guess which colors indicate edits')
    positive = rgba[:, :, 0] > 0
    if not positive.any() or positive.all():
        raise ValueError('Empty or all-positive mask is ineligible for fixed local ROI')
    mask_height, mask_width = positive.shape
    aspect_error = abs(math.log((source.width / source.height) / (mask_width / mask_height)))
    if aspect_error > 0.01:
        raise ValueError(f'Source/mask aspect mismatch {aspect_error}; unverified crop/padding cannot be inferred')
    ys, xs = np.nonzero(positive)
    raw_bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    width, height = dimension(source.width, source.height)
    binary_image = Image.fromarray(positive.astype(np.uint8) * 255).resize((width, height), Image.Resampling.NEAREST)
    binary = np.asarray(binary_image) > 0
    if not binary.any():
        raise ValueError('Mask disappears under the predefined output transform')
    yy, xx = np.nonzero(binary)
    bbox = [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]
    grid_height, grid_width = height // 16, width // 16
    token_box = [max(0, bbox[0] // 16 - 2), max(0, bbox[1] // 16 - 2),
        min(grid_width, math.ceil(bbox[2] / 16) + 2), min(grid_height, math.ceil(bbox[3] / 16) + 2)]
    indices = [y * grid_width + x for y in range(token_box[1], token_box[3]) for x in range(token_box[0], token_box[2])]
    if len(indices) == grid_height * grid_width:
        raise ValueError('Mask bbox plus halo covers all target tokens; local-execution calibration requires inactive tokens')
    buffer = io.BytesIO()
    binary_image.save(buffer, format='PNG')
    info = dict(source_wh=list(source.size), source_mode=source_mode, source_format=source_format,
        original_mask_wh=[mask_width, mask_height], original_mask_mode=mask_mode, original_mask_format=mask_format,
        mask_positive_rule='opaque grayscale channel > 0', source_mask_aspect_log_error=aspect_error,
        raw_mask_bbox_xyxy_exclusive=raw_bbox,
        source_bbox_xyxy_float=[raw_bbox[0] * source.width / mask_width, raw_bbox[1] * source.height / mask_height,
            raw_bbox[2] * source.width / mask_width, raw_bbox[3] * source.height / mask_height],
        target_wh=[width, height], target_hw=[height, width], target_grid_hw=[grid_height, grid_width],
        target_mask_bbox_xyxy_exclusive=bbox, core_token_bbox_xyxy_exclusive=token_box,
        halo_tokens=2, n_target_tokens=grid_height * grid_width, n_core_tokens=len(indices),
        target_active_fraction=len(indices) / (grid_height * grid_width),
        mask_transform='same normalized source canvas, nearest neighbor; no crop/pad inferred',
        roi_protocol='provided annotation bbox plus fixed halo; not a learned semantic boundary', indices=indices)
    return source, info, buffer.getvalue()


def extract_train8(root, selection, inventory, old_cases):
    destination = root / 'data' / 'train8_assets.json'
    selection_sha = sha256((root / 'data' / 'frozen_selection.json').read_bytes())
    if selection['contract'] != contract() or not selection['metadata_complete']:
        raise RuntimeError('Extraction requires the complete pinned frozen metadata selection')
    old = old_signatures(old_cases)
    if destination.exists():
        saved = load(destination)
        if saved['frozen_selection_sha256'] != selection_sha or saved['old_demo_reference'] != old:
            raise RuntimeError('Saved extraction has different frozen inputs')
        expected_ids = [row['img_id'] for row in selection['selections']['train8']]
        if [row['img_id'] for row in saved['train8_records']] != expected_ids:
            raise RuntimeError('Saved Train8 IDs differ from frozen selection')
        for row in saved['train8_records']:
            for path_key, hash_key in [('image', 'source_sha256'), ('original_mask', 'mask_sha256'),
                    ('binary_mask', 'binary_mask_sha256')]:
                if sha256((root / row[path_key]).read_bytes()) != row[hash_key]:
                    raise RuntimeError(f'Saved Train8 asset changed: {row[path_key]}')
        cases_path = root / 'cases_train8.json'
        if not cases_path.exists():
            frozen_json(cases_path, dict(dataset=REPO, revision=REVISION, seed=SEED,
                split='official_train_first_turn_train8', selection_sha256=selection_sha,
                ground_truth_target_used=False, ready_for_calibration=saved['ready_for_calibration'],
                cases=saved['train8_records']))
        if not saved['ready_for_calibration']:
            raise RuntimeError('Existing source-review report is preserved; no replacement or automatic acceptance allowed')
        return saved
    shards = {row['path']: row for row in inventory['files']}
    chosen = selection['selections']['train8']
    records = []
    references = {key: row['signature'] for key, row in old['signatures'].items()}
    groups = defaultdict(list)
    for record in chosen:
        groups[(record['shard_path'], record['row_group'])].append(record)
    try:
        for (shard_path, group), wanted in sorted(groups.items()):
            shard = shards[shard_path]
            with AuditRangeFile(shard['source_url'], shard['size_bytes'], root / 'data' / 'network_ranges.jsonl',
                    'train8_source_mask_extraction', shard_path) as handle:
                parquet = open_parquet(handle)
                table = parquet.read_row_group(group, columns=ASSET_COLUMNS, use_threads=False)
                assert table.column_names == ASSET_COLUMNS and 'target_img' not in table.column_names
                for selected in wanted:
                    case_id = 'mb_train_' + selected['img_id']
                    item = dict(selected, case_id=case_id, status='EXTRACTING', target_column_read=False,
                        columns_read=ASSET_COLUMNS)
                    records.append(item)
                    row = table.slice(selected['row_in_group'], 1).to_pylist()[0]
                    assert str(row['img_id']) == selected['original_img_id']
                    assert int(row['turn_index']) == 1 and row['instruction'] == selected['instruction']
                    source_bytes = row['source_img'].get('bytes')
                    mask_bytes = row['mask_img'].get('bytes')
                    if not isinstance(source_bytes, bytes) or not isinstance(mask_bytes, bytes):
                        raise RuntimeError('Embedded original image bytes are required; external paths are not followed')
                    item.update(source_sha256=sha256(source_bytes), mask_sha256=sha256(mask_bytes),
                        source_size_bytes=len(source_bytes), mask_size_bytes=len(mask_bytes))
                    base = Path('inputs') / 'train8' / selected['img_id']
                    source_path, mask_path = base / 'source.original', base / 'mask.original'
                    immutable_bytes(root / source_path, source_bytes)
                    immutable_bytes(root / mask_path, mask_bytes)
                    item.update(image=str(source_path), original_mask=str(mask_path), prompt=selected['instruction'])
                    source, geometry, binary_bytes = decode_geometry(source_bytes, mask_bytes)
                    signature = image_signature(source, source_bytes)
                    comparisons = [duplicate_comparison(signature, reference, key) for key, reference in references.items()]
                    item.update(signature=signature, duplicate_comparisons=comparisons, **geometry)
                    if any(row['verdict'] == 'REJECT' for row in comparisons):
                        item['status'] = 'REJECTED_SOURCE_DUPLICATE'
                        raise RuntimeError(f'Frozen {case_id} duplicates an old demo or previously extracted Train8 source; no replacement allowed')
                    references[case_id] = signature
                    binary_path = base / 'mask.target.binary.png'
                    immutable_bytes(root / binary_path, binary_bytes)
                    item.update(binary_mask=str(binary_path), binary_mask_sha256=sha256(binary_bytes),
                        status='SOURCE_REVIEW_REQUIRED' if any(row['gray_zone'] for row in comparisons) else 'PASS')
                    print(json.dumps(dict(phase='train8_asset', case_id=case_id, status=item['status'],
                        target_hw=item['target_hw'], core_fraction=item['target_active_fraction']), ensure_ascii=False), flush=True)
        order = {row['img_id']: index for index, row in enumerate(chosen)}
        records.sort(key=lambda row: order[row['img_id']])
        assert len(records) == 8 and [row['img_id'] for row in records] == [row['img_id'] for row in chosen]
        ready = all(row['status'] == 'PASS' for row in records)
        report = dict(status='PASS' if ready else 'SOURCE_REVIEW_REQUIRED', created_at=now(),
            frozen_selection_sha256=selection_sha, contract_sha256=selection['contract_sha256'],
            script_sha256=sha256(Path(__file__).read_bytes()), old_demo_reference=old,
            train8_records=records, requested_extracted_sources=8, extracted_sources=len(records),
            target_columns_read=False, test_media_read=False, dev_media_read=False,
            train16_remaining_media_read=False, train32_remaining_media_read=False,
            full_dataset_content_dedup_complete=False, source_only_gray_zone_review_required=not ready,
            ready_for_calibration=ready, model_calls=0, dit_forwards=0, training_steps=0,
            network_granularity='Parquet row-group projected columns; may fetch source/mask bytes of unselected rows in the same group, but never the target column')
        if destination.exists():
            previous = load(destination)
            if previous['frozen_selection_sha256'] != selection_sha or previous['old_demo_reference'] != old:
                raise RuntimeError('Existing extraction report has different frozen inputs')
            old_records = previous['train8_records']
            if old_records != records:
                raise RuntimeError('Existing extraction results differ; refusing to overwrite')
        else:
            frozen_json(destination, report)
        cases_path = root / 'cases_train8.json'
        cases = dict(dataset=REPO, revision=REVISION, seed=SEED, split='official_train_first_turn_train8',
            selection_sha256=selection_sha, ground_truth_target_used=False,
            ready_for_calibration=ready, cases=records)
        if cases_path.exists():
            if load(cases_path) != cases:
                raise RuntimeError('Existing cases_train8.json differs')
        else:
            frozen_json(cases_path, cases)
        atomic_json(root / 'data' / 'extraction_status.json', dict(status=report['status'], time=now(),
            ready_for_calibration=ready, extracted_sources=8, model_calls=0, dit_forwards=0, training_steps=0,
            test_generated=False, target_columns_read=False, network=network_summary(root)))
        if not ready:
            raise RuntimeError('Train8 source-only near-duplicate gray zones require review; frozen IDs are preserved')
        return report
    except Exception as error:
        failure = dict(status='BLOCKED_EXTRACTION', time=now(), frozen_selection_sha256=selection_sha,
            error_type=type(error).__name__, error=str(error), partial_records=records,
            no_replacement_performed=True, ready_for_calibration=False,
            target_columns_read=False, test_media_read=False, model_calls=0, dit_forwards=0, training_steps=0)
        atomic_json(root / 'data' / 'extraction_status.json', failure)
        append_json(root / 'data' / 'extraction_attempts.jsonl', failure)
        raise


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--metadata-only', action='store_true')
    mode.add_argument('--extract-train8', action='store_true')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--old-cases', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    (root / 'data').mkdir(parents=True, exist_ok=True)
    if args.metadata_only:
        inventory = repo_inventory(root)
        metadata = [metadata_shard(root, shard) for shard in inventory['files']]
        selection = freeze_selection(root, inventory, metadata)
        print(json.dumps(dict(status='COMPLETE_FROZEN', selection_sha256=sha256((root / 'data' / 'frozen_selection.json').read_bytes()),
            counts={key: len(value) for key, value in selection['selections'].items()}, model_calls=0,
            target_columns_read=False, test_media_read=False), ensure_ascii=False), flush=True)
    else:
        selected_path = root / 'data' / 'frozen_selection.json'
        if not selected_path.exists():
            raise RuntimeError('Run --metadata-only until all 61 shards are complete and the selection is frozen first')
        selection = load(selected_path)
        expected_sha = selection['selection_payload_sha256']
        payload = {key: value for key, value in selection.items() if key not in ('selection_payload_sha256', 'created_at', 'script_sha256')}
        assert sha256(canonical(payload)) == expected_sha
        inventory = repo_inventory(root)
        if inventory['files_sha256'] != selection['inventory_files_sha256']:
            raise RuntimeError('Inventory differs from frozen selection')
        old_cases = args.old_cases or root.parent / 'Lightning8前置实验_20261022' / 'cases.json'
        report = extract_train8(root, selection, inventory, old_cases)
        print(json.dumps(dict(status=report['status'], ready_for_calibration=report['ready_for_calibration'],
            train8_assets=str(root / 'data' / 'train8_assets.json'), model_calls=0), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
