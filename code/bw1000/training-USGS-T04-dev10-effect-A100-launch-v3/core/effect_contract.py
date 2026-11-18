"""Dev10 identity/EOS/record contracts. No model, gold, network or device imports."""
import hashlib
import json
import re
from pathlib import Path

ARMS = ('initial', 'domain25', 'generic25')
CAP = 256


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode()


def pinned(path, pin, maximum=2 * 1024**2):
    path = Path(path)
    before = path.lstat()
    need(path.is_file() and not path.is_symlink() and before.st_size == pin['bytes']
         and before.st_size <= maximum, 'EXACT_REGULAR_FILE_REQUIRED')
    data = path.read_bytes()
    after = path.lstat()
    need((before.st_ino, before.st_size, before.st_mtime_ns) ==
         (after.st_ino, after.st_size, after.st_mtime_ns)
         and sha(data) == pin['SHA256'], 'PIN_OR_STABLE_STAT_MISMATCH')
    return data


def eos_observation(ids, eos_ids):
    need(isinstance(ids, list) and 0 < len(ids) <= CAP
         and all(type(x) is int and x >= 0 for x in ids), 'RAW_GENERATED_IDS_REQUIRED')
    need(isinstance(eos_ids, list) and eos_ids
         and all(type(x) is int and x >= 0 for x in eos_ids), 'EOS_IDS_REQUIRED')
    # Conservative: reaching the independent cap is not accepted even if final ID is EOS.
    cap_hit = len(ids) >= CAP
    natural = ids[-1] in eos_ids and not cap_hit and not any(x in eos_ids for x in ids[:-1])
    return dict(natural_EOS=natural, cap_hit=cap_hit, generated_tokens=len(ids),
                generated_ids_SHA256=sha(canonical(ids)), last_token_is_EOS=ids[-1] in eos_ids)


def scoring_prediction(row):
    value = dict(row)
    text = value.get('text')
    match = re.fullmatch(r'(?:FINAL: )?(-?\d+\.\d{2}|E\d{2}(?: E\d{2})*)', text.strip()) \
        if isinstance(text, str) and '\n' not in text and '\r' not in text else None
    value['text'] = match.group(1) if match else 'INVALID_FINAL_FORMAT'
    value['strict_FINAL_format'] = bool(match and text.strip().startswith('FINAL: '))
    return value


def load_dev(plan, data, identities):
    rows = [json.loads(line) for line in pinned(Path(data) / 'dev-swift.jsonl', plan['dev_inputs']).splitlines()]
    refs = json.loads(pinned(identities, plan['native_identities']))
    refs = [r for r in refs if r['split'] == 'dev']
    need(len(rows) == len(refs) == len({r['id'] for r in rows}) == 10, 'ALL_TEN_DEV_SLOTS_REQUIRED')
    for ordinal, (row, ref) in enumerate(zip(rows, refs)):
        need(set(row) == {'id', 'messages', 'images'} and ref['ordinal'] == ordinal
             and sha(row['id'].encode()) == ref['source_id_SHA256']
             and len(row['messages']) == 1 and row['messages'][0]['role'] == 'user'
             and len(row['images']) == 1 and ref['labels_None'] is True
             and ref['full_input_retained'] is True, 'DEV_INPUT_OR_NATIVE_REF_MISMATCH')
        expected = {str(Path(data) / name): pin for name, pin in plan['dev_media'].items()}
        need(row['images'][0] in expected, 'DEV_IMAGE_OUTSIDE_FROZEN_TWO')
        pinned(row['images'][0], expected[row['images'][0]])
    return rows, refs


def require_ready(plan):
    need(plan['root_GO'] is True and plan['GPU_GO'] is True, 'ROOT_ADMISSION_REQUIRED')
    need(plan['generic25'] is not None, 'GENERIC25_NOT_READY_NO_ACTUAL_ARTIFACT')
    need(plan['generic25']['actual_updates'] == 25 and plan['generic25']['actual_terminal_qualified'] is True
         and plan['generic25']['saved_adapter_source_qualified'] is True,
         'GENERIC25_REAL_TERMINAL_AND_STATE_REQUIRED')
    need(plan['generic25']['initial_named_SHA256'] == plan['initial_named_SHA256'],
         'PAIRED_INITIAL_TRAINABLES_DIFFER')
    need(plan['domain_original_training_accepted'] is False, 'DO_NOT_WASH_ORIGINAL_DOMAIN_FAILURE')


def dump(path, value):
    with Path(path).open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
