"""Reuse the common Swift messages/images interface; train gold only, no training loop."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

D = Path(__file__).resolve().parent


def need(ok, why):
    if not ok:
        raise ValueError(why)


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n').encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def pinned(path, pin, cap=2 * 1024**2):
    need(path.is_file() and not path.is_symlink() and 0 < pin['bytes'] <= cap, 'bounded regular source required')
    before = path.stat(); raw = path.read_bytes(); after = path.stat()
    need((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
         == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
         and len(raw) == pin['bytes'] and digest(raw) == pin['SHA256'], 'source identity changed')
    return raw


def verify_source():
    manifest = json.loads((D / 'SOURCE_FILES.json').read_bytes())
    for name, pin in manifest['files'].items():
        need(not Path(name).is_absolute() and '..' not in Path(name).parts, 'source member path differs')
        pinned(D / name, pin)


def join(rows, labels, media, target, split):
    """Order-preserving join, no answer-based selection; dev never receives gold."""
    count = 45 if split == 'train' else 10
    need(split in {'train', 'dev'} and len(rows) == count and len({r['id'] for r in rows}) == count,
         'full split denominator required')
    root = PurePosixPath(target)
    need(root.is_absolute() and '..' not in root.parts and 'blind' not in root.parts, 'target data root differs')
    gold = {r['id']: r for r in labels} if labels is not None else None
    need((split == 'train' and gold is not None and len(labels) == len(gold) == count and set(gold) == {r['id'] for r in rows})
         or (split == 'dev' and labels is None), 'train-only gold/full membership required')
    result = []
    for row in rows:
        need(set(row) == {'id', 'chart_group', 'source_group', 'messages', 'images'}
             and len(row['messages']) == len(row['images']) == 1, 'source input schema differs')
        user = row['messages'][0]
        need(set(user) == {'role', 'content'} and user['role'] == 'user'
             and user['content'].startswith('<image>\n') and user['content'].count('<image>') == 1,
             'native user/image prompt differs')
        name = row['images'][0]; path = PurePosixPath(name)
        need(path.parts[:2] == (split, 'media') and not path.is_absolute() and '..' not in path.parts
             and name in media and path.suffix == '.png', 'foreign split/image rejected')
        messages = [dict(user)]
        if split == 'train':
            label = gold[row['id']]
            need(label['data_status'] == 'QUALIFIED' and label['chart_id'] == row['chart_group']
                 and label['group'] == row['source_group'] and user['content'] == '<image>\n' + label['question'],
                 'train gold/source join differs')
            answer = label['answer']
            pattern = r'-?(?:0|[1-9][0-9]*)\.[0-9]{2}' if label['answer_type'] == 'decimal2' else r'E[0-9]{2}(?: E[0-9]{2})*'
            need(label['answer_type'] in {'decimal2', 'labels'} and isinstance(answer, str)
                 and len(answer) <= 64 and re.fullmatch(pattern, answer), 'frozen answer format differs')
            messages.append(dict(role='assistant', content='FINAL: ' + answer))
        result.append(dict(id=row['id'], messages=messages, images=[str(root / path)]))
    need(len({r['images'][0] for r in result}) == (9 if split == 'train' else 2), 'all split media required')
    return result


def main(a):
    verify_source()
    claim = json.loads((D / 'CLAIM.json').read_bytes())
    need(claim['root_GO'] is True and claim['CPU_GO'] is True, 'root publication/CPU admission required')
    plan = json.loads((D / 'PLAN.json').read_bytes())
    audit = json.loads(pinned(a.audit, plan['dataset_audit']))
    need(audit['passed'] is True and audit['all_gold_derived_from_displayed_Decimal'] is True
         and audit['all_inputs_answer_free'] is True and audit['crosssplit_event_or_group_overlap'] == 0
         and audit['split_QA'] == {'train': 45, 'dev': 10, 'blind': 25}, 'independent dataset audit not qualified')
    manifest = json.loads(pinned(a.dataset / 'manifest-private.json', plan['dataset_manifest']))
    need(manifest['failed_charts'] == 0 and manifest['group_closure_before_QA'] is True, 'dataset source closure differs')
    charts = [r for r in manifest['charts'] if r['split'] in {'train', 'dev'}]
    media = {r['image']['name']: r['image'] for r in charts}
    need(len(media) == 11 and all(r['status'] == 'QUALIFIED' for r in charts), 'full11 train/dev media required')
    need(not a.output.exists(), 'new adapter namespace already exists; no replay')
    a.output.mkdir(mode=0o700)
    result = dict(passed=False, planned_train_rows=45, planned_dev_rows=10, training_GO=False, GPU_GO=False)
    try:
        inputs = {}
        for split in ['train', 'dev']:
            raw = pinned(a.dataset / split / 'inputs.jsonl', plan['inputs'][split])
            inputs[split] = [json.loads(x) for x in raw.splitlines()]
        labels = [json.loads(x) for x in pinned(a.dataset / 'train/labels-private.jsonl', plan['train_labels']).splitlines()]
        pins = {}
        for split in ['train', 'dev']:
            rows = join(inputs[split], labels if split == 'train' else None, media, a.target_data, split)
            raw = b''.join(canonical(r) for r in rows); name = split + '-swift.jsonl'
            (a.output / name).write_bytes(raw); pins[name] = dict(bytes=len(raw), SHA256=digest(raw), rows=len(rows))
        for name, pin in media.items():
            pinned(a.dataset / name, pin, 1024**2)
        result.update(passed=True, state='TRAIN45_SUPERVISED_DEV10_ANSWER_FREE_ADAPTER_ONLY',
                      dataset_manifest_SHA256=plan['dataset_manifest']['SHA256'], dataset_audit_SHA256=plan['dataset_audit']['SHA256'],
                      files=pins, media=media, target_data=a.target_data, literal_special_tokens_added=False,
                      dev_gold_read=False, blind_inputs_gold_or_media_read=False, native_qualified=False)
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error)[:200]); raise
    finally:
        (a.output / 'data-binding-private.json').write_bytes(canonical(result))
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ['audit', 'dataset', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--target-data', required=True)
    raise SystemExit(main(p.parse_args()))
