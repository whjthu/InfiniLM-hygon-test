"""Stage both frozen45->800 views in the same prospective training install, no encode."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from exposures import exposure_indices


def need(ok, code):
    if not ok:
        raise ValueError(code)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def pinned(pin, check, cap=262144):
    check(); p = Path(pin['path']); before = p.lstat()
    need(p.is_file() and not p.is_symlink() and 0 < before.st_size <= cap, 'BOUNDED_QUALIFIED_ARTIFACT')
    if pin.get('bytes') is not None:
        need(before.st_size == pin['bytes'], 'QUALIFIED_ARTIFACT_SIZE')
    raw = p.read_bytes(); after = p.lstat()
    need((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
         (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) and
         len(raw) == before.st_size and sha(raw) == pin['SHA256'], 'QUALIFIED_ARTIFACT_SHA')
    return raw, dict(path=str(p), bytes=len(raw), SHA256=sha(raw))


def write(path, raw):
    with path.open('xb') as f:
        f.write(raw)


def build_views(root, source, plan, check):
    need(root.is_absolute() and not root.exists(), 'NEW_PAIR_VIEWS_NO_REPLAY')
    root.mkdir(mode=0o700)
    indices = exposure_indices(); counts = Counter(indices)
    need(len(indices) == 800 and len(counts) == 45 and set(counts.values()) == {17, 18}, 'FIXED800_SCHEDULE')
    results = {}
    for arm in ('domain', 'generic'):
        c = plan['input_arms'][arm]
        qraw, _ = pinned(dict(c['qualification'], path=str(source / c['qualification']['file'])), check)
        q = json.loads(qraw)
        if arm == 'domain':
            need(q['case'] == 'T03.sft' and q['native_preprocessing_qualified'] is True and
                 q['completed_train'] == 45 and q['completed_dev'] == 10 and q['actual_updates'] == 0,
                 'DOMAIN45_ALREADY_NATIVE_QUALIFIED')
        else:
            need(q['independent_passed'] is True and q['state'] == 'T03_GENERIC45_NATIVE_CPU_INDEPENDENT_METADATA_PASS'
                 and q['completed_train'] == 45 and q['missing_train'] == q['failed_train'] == q['new_dev_encodes'] == 0,
                 'GENERIC45_ACTUAL_INDEPENDENT_QUALIFICATION')
        raw, canonical_pin = pinned(c['canonical45'], check)
        binding_raw, _ = pinned(c['data_binding'], check)
        native_raw, native_pin = pinned(c['native_identities'], check)
        binding = json.loads(binding_raw); full_native = json.loads(native_raw)
        native = [row for row in full_native if row['split'] == 'train']
        need(len(full_native) == c['native_total_records'] and len(native) == 45 and
             [row['ordinal'] for row in native] == list(range(45)), 'ACTUAL45_NATIVE_REFERENCE')
        need(binding['passed'] is True and binding['files']['train-swift.jsonl']['SHA256'] == sha(raw) and
             binding['files']['train-swift.jsonl']['bytes'] == len(raw) and binding['files']['train-swift.jsonl']['rows'] == 45
             and binding['dev_gold_read'] is False and binding['blind_inputs_gold_or_media_read'] is False,
             'ORIGINAL_TRAIN_ONLY_BINDING')
        lines = raw.splitlines(keepends=True)
        need(len(lines) == 45 and all(line.endswith(b'\n') for line in lines), 'FULL45_LINES')
        rows = [json.loads(line) for line in lines]; checked_media = set()
        for row, reference in zip(rows, native):
            check()
            need(set(row) == {'id', 'messages', 'images'} and len(row['messages']) == 2 and
                 row['messages'][0]['role'] == 'user' and row['messages'][1]['role'] == 'assistant' and
                 row['messages'][1]['content'].startswith('FINAL: ') and len(row['images']) == 1,
                 'CANONICAL_FINAL_SUPERVISION_ONLY')
            need(sha(row['id'].encode()) == reference['source_id_SHA256'] and reference['labels_None'] is False and
                 reference['full_input_retained'] is True and reference['supervised_EOS_count'] > 0,
                 'CANONICAL_NATIVE_FULL_IDENTITY')
            image = Path(row['images'][0]); media_root = Path(binding['target_data'])
            need(image.is_relative_to(media_root), 'IMAGE_ESCAPED_QUALIFIED_SCOPE')
            relative = image.relative_to(media_root).as_posix()
            need(relative.startswith('train/media/') and relative in binding['media'], 'NO_DEV_OR_BLIND_TRAINING')
            if relative not in checked_media:
                pinned(dict(binding['media'][relative], path=str(image)), check, cap=1048576)
                checked_media.add(relative)
        need(len({row['id'] for row in rows}) == 45, 'UNIQUE45')
        output = root / arm; output.mkdir(mode=0o700)
        view_raw = b''.join(lines[i] for i in indices)
        write(output / 'train-800.jsonl', view_raw)
        ledger_raw = canonical(dict(arm=arm, view_SHA256=sha(view_raw), original_index_by_physical_row=indices))
        write(output / 'exposure-ledger-private.json', ledger_raw)
        # Full native45 references stay in R. Per-occurrence IDs/labels are identity candidates until real forward observation.
        occurrences = []
        for physical, i in enumerate(indices):
            r = native[i]
            occurrences.append(canonical(dict(physical_row=physical, canonical_ordinal=i,
                source_id_SHA256=r['source_id_SHA256'], canonical_line_SHA256=sha(lines[i]),
                native_record_SHA256=sha(canonical(r)), collated_fields_SHA256=sha(canonical(r['collated_fields'])),
                CPU_effective_supervised_tokens=r.get('effective_supervised_tokens'),
                GPU_padding_rank_cursor_or_input_match_qualified=False)))
        occurrence_raw = b''.join(occurrences)
        write(output / 'occurrence800-private.jsonl', occurrence_raw)
        def actual(name, body):
            return dict(path=str(output / name), bytes=len(body), SHA256=sha(body))
        known = all('effective_supervised_tokens' in r for r in native)
        result = dict(arm=arm, unique_rows=45, physical_rows=800,
            canonical45=canonical_pin, native45=native_pin,
            training800=actual('train-800.jsonl', view_raw),
            occurrences=actual('exposure-ledger-private.json', ledger_raw),
            occurrence_ledger=actual('occurrence800-private.jsonl', occurrence_raw),
            original_index_sequence_SHA256=sha(canonical(indices)), counts=[counts[i] for i in range(45)],
            CPU_full_input_tokens_weighted800=sum(counts[i] * r['full_length'] for i, r in enumerate(native)),
            CPU_effective_supervised_tokens_weighted800=sum(counts[i] * r['effective_supervised_tokens'] for i, r in enumerate(native)) if known else None,
            actual_GPU_supervised_token_exposures=None, actual_updates=0, new_encodes=0,
            new_dev_inputs_read=0, gold_or_blind_inputs_read=False, actual_training_qualified=False)
        write(output / 'view-binding-private.json', canonical(result)); results[arm] = result
    need(results['domain']['original_index_sequence_SHA256'] == results['generic']['original_index_sequence_SHA256'],
         'BOTH_ARMS_SAME800_PHYSICAL_SCHEDULE')
    write(root / 'pair-views-private.json', canonical(results))
    return results
