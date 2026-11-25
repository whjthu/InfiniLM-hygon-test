"""Issue and independently validate one disposition per original prediction slot."""
import json
import os

from effect_contract import ARMS, eos, need, pin, read

STATUSES = {'COMPLETED', 'ERROR', 'MISSING'}


def failed_slot(row, arm, ordinal, status, error_type=None):
    return dict(id=row['id'], arm=arm, ordinal=ordinal, status=status, reason=error_type,
                text=None, natural_EOS=False, cap_hit=False)


def seal_arm(output, arm, rows, predictions):
    complete = []
    for ordinal, row in enumerate(rows):
        complete.append(predictions.get(ordinal, failed_slot(row, arm, ordinal, 'MISSING')))
    path = output / (arm + '-predictions-private.jsonl')
    with path.open('xb') as stream:
        for row in complete:
            stream.write(json.dumps(row, allow_nan=False, ensure_ascii=False).encode() + b'\n')
        stream.flush()
        os.fsync(stream.fileno())
    counts = {status.lower(): sum(row['status'] == status for row in complete)
              for status in STATUSES}
    return dict(planned=36, attempted=counts['completed'] + counts['error'], **counts,
                file=dict(name=path.name, **pin(read(path, cap=4 << 20))))


def verify_slots(folder, final, expected_ids):
    need(len(expected_ids) == len(set(expected_ids)) == 36, 'explicit order requires exact36')
    need(set(final['arms']) == set(ARMS), 'all three arm ledgers required')
    all_rows, pins, arms = {}, {}, {}
    for arm in ARMS:
        entry = final['arms'][arm]
        name = arm + '-predictions-private.jsonl'
        need(entry['file']['name'] == name, 'exact sealed slot filename')
        raw = read(folder / name, entry['file'], cap=4 << 20)
        rows = [json.loads(line) for line in raw.splitlines()]
        need([row['id'] for row in rows] == expected_ids, 'ordered original blind36 must match')
        for ordinal, row in enumerate(rows):
            need(row['arm'] == arm and row['ordinal'] == ordinal and row['status'] in STATUSES
                 and type(row['natural_EOS']) is type(row['cap_hit']) is bool, 'disposition context differs')
            if row['status'] == 'COMPLETED':
                need(entry['actual476_pre_forward_equal'] is True
                     and eos(row['raw_generated_token_ids'], row['raw_EOS_ids']) ==
                     {key: row[key] for key in ('natural_EOS', 'cap_hit', 'generated_tokens',
                                                'generated_ids_SHA256')}, 'native EOS or arm proof differs')
            else:
                need(row['text'] is None and not row['natural_EOS'] and not row['cap_hit']
                     and 'raw_generated_token_ids' not in row, 'failed slot contains a prediction')
        counts = {status.lower(): sum(row['status'] == status for row in rows) for status in STATUSES}
        need(entry['planned'] == 36 and entry['attempted'] == counts['completed'] + counts['error']
             and all(entry[key] == value for key, value in counts.items()), 'slot denominators differ')
        arms[arm] = dict(**counts, natural_EOS=sum(row['natural_EOS'] for row in rows),
                         cap=sum(row['cap_hit'] for row in rows))
        pins[name] = pin(raw)
        all_rows[arm] = rows
    return all_rows, pins, arms
