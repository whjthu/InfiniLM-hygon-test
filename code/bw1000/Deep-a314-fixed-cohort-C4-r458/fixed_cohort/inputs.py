"""Consume the four exact CPU945 private records, without tokenization."""
import hashlib
import json
from pathlib import Path

ENCODING_SHA = '04c1ffa27726cc89a177814d2e8ed4838fed51052fee715a33851b9b65c8dd5a'
GOLD_SHA = 'c350be0128d1304ecbe5ae0d9ddd53d2d2137faac37f733b60bf0ba7cdb08a38'
ORDINALS = (0, 1, 128, 129)
compact = lambda x: json.dumps(x, ensure_ascii=False, separators=(',', ':')).encode()
digest = lambda x: hashlib.sha256(x).hexdigest()


def load_four(encoding_path, gold_path):
    raw = Path(encoding_path).read_bytes()
    assert digest(raw) == ENCODING_SHA
    enc = json.loads(raw)
    assert enc['passed'] and enc['completed'] and enc['actual_CPU'] == '910407.945'
    assert enc['actual_encodings'] == 256 and enc['actual_Deep_codec']
    gold_raw = Path(gold_path).read_bytes()
    assert digest(gold_raw) == GOLD_SHA
    gold = {c['id']: c for c in json.loads(gold_raw)['cases']}
    records = []
    for row, ordinal in enumerate(ORDINALS):
        meta = enc['cases'][ordinal]
        assert meta['selected_ordinal'] == ordinal
        assert meta['suite'] == ('gsm8k' if row < 2 else 'mmlu')
        assert meta['shot_count'] == (8 if row < 2 else 5)
        assert meta['output_budget'] == 4096 and meta['actual_thinking'] is False
        private = Path(meta['private_path']).read_bytes()
        assert digest(private) == meta['private_SHA256']
        p = json.loads(private)
        assert p['case_id'] == meta['case_id']
        body, ids = p['body'], p['prompt_token_ids']
        assert digest(compact(body)) == meta['request_body_SHA256']
        assert digest(compact(ids)) == meta['prompt_IDs_SHA256']
        assert digest(p['prompt'].encode()) == meta['prompt_SHA256']
        assert len(ids) == meta['prompt_tokens'] and len(ids) + 4096 <= 32768
        assert all(type(t) is int and 0 <= t < 129280 for t in ids)
        assert body['max_tokens'] == 4096
        assert body['chat_template_kwargs']['enable_thinking'] is False
        case = gold[meta['case_id']]
        assert digest(compact(case)) == meta['source_case_SHA256']
        records.append(dict(row=row, ordinal=ordinal, case_id=p['case_id'],
            prompt_token_ids=tuple(ids), body=body, gold=case,
            max_new_tokens=4096, prompt_IDs_SHA256=meta['prompt_IDs_SHA256'],
            body_SHA256=meta['request_body_SHA256']))
    return records
