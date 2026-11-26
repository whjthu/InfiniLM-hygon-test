"""Independently score the original sealed Gemma 96-question confirmation locally."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import runpy


REPO = Path(__file__).resolve().parent.parent
LOCAL = REPO / 'local-only'
PREDICTIONS = LOCAL / 'gemma96-original3682-authorized-288-predictions-once'
SCORE = PREDICTIONS / 'first-score'
BUNDLE = LOCAL / 'gemma-usgs-confirmatory96-prelabel-render-49253df040cf-original'
SCOPE = '480d9f01d2784d0a870cb049dbaec61f'
RECEIPT = 'c5030451ba9fa005702e7ac233546b56bb9ec60016b166904e315b616381cfcf'
PREREG = 'b5a9a308dd50c2a940da9f9eb2a6b2bfc23872cecd37ab0b5a07816d6790f1a6'
SOURCES = {
    'geometry': ('code/bw1000/gemma-usgs-confirmatory96-prelabel-render-v1/build.py',
                 'a45a3dba03f087f33a48f5b50008ea4862ba08fa5efdafbc083b1a3a771c13fc'),
    'parser': ('code/bw1000/training-gemma-usgs72-blind36-4e86f89e971b-v4/core/response_protocol.py',
               '37d0bdc4f085b0210740dba227706d1416753d85c57bf37c722cc662f21eba2f'),
    'statistics': ('tools/gemma_confirmatory96_statistics.py',
                   'c29b4e98e85efc37d9a891ea2f4fc9c60623a15e9371ce499736735560fe7f89'),
}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def pin(raw):
    return dict(bytes=len(raw), SHA256=sha256(raw).hexdigest())


def read(path, expected, cap):
    with path.open('rb') as stream:
        raw = stream.read(cap + 1)
    require(0 < len(raw) <= cap and pin(raw)['SHA256'] == expected, f'Frozen bytes differ for {path.name}')
    return raw


def save(path, data):
    raw = (json.dumps(data, sort_keys=True, allow_nan=False, ensure_ascii=False) + '\n').encode()
    with path.open('xb') as stream:
        stream.write(raw)
    return pin(raw)


def main():
    modules = {}
    for name, (path, digest) in SOURCES.items():
        read(REPO / path, digest, 65_536)
        modules[name] = runpy.run_path(str(REPO / path))
    prereg = read(REPO / 'coordination/tasks/training-t02-gemma-usgs-confirmatory96-preregistration-v1.json', PREREG, 65_536)
    receipt = json.loads(read(PREDICTIONS / 'receipt.json', RECEIPT, 4096))
    require(receipt['scope'] == SCOPE and receipt['total'] == 288 and receipt['gold_transferred'] is False,
            'Original prediction transfer scope differs')
    stats = modules['statistics']
    arms = stats['ARMS']
    names = {f'{arm}-predictions-private.jsonl' for arm in arms}
    require(set(receipt['prediction_files']) == names, 'Original prediction filenames differ')
    manifest = json.loads(read(BUNDLE / 'INPUT_FILES_PRIVATE.json',
                               '9e53e81251a722680c7d15c607f27377740fe234bd400971b9513864508e2ec7', 65_536))
    inputs_raw = read(BUNDLE / 'inputs.jsonl', '63962a41f124a50e1a2b38fa513cc2aaefe4dcb632162ba06d2070a462baa408', 65_536)
    require(manifest['inputs'] == pin(inputs_raw) and manifest['blind_scoring_GO'] is False,
            'Original input-only manifest differs')
    inputs = [json.loads(line) for line in inputs_raw.splitlines()]
    ids = [v['id'] for v in inputs]
    require(len(set(ids)) == len(ids) == stats['SLOTS_PER_ARM'], 'Original input denominator differs')
    predictions = {}
    for arm in arms:
        filename = f'{arm}-predictions-private.jsonl'
        expected = receipt['prediction_files'][filename]
        raw = read(PREDICTIONS / filename, expected['SHA256'], 4 << 20)
        require(pin(raw) == expected, 'Original sealed prediction length differs')
        rows = [json.loads(line) for line in raw.splitlines()]
        require([v['id'] for v in rows] == ids and all(
            v['arm'] == arm and v['ordinal'] == index and v['status'] == 'COMPLETED'
            and v['natural_EOS'] is True and v['cap_hit'] is False
            for index, v in enumerate(rows)), 'Original sealed prediction or disposition differs')
        predictions[arm] = rows

    SCORE.mkdir(mode=0o700)
    intent = save(SCORE / 'intent.json', dict(scope=SCOPE, original_prediction_receipt_SHA256=RECEIPT,
                                            preregistration=pin(prereg), denominator=288,
                                            issued_utc=datetime.now(timezone.utc).isoformat()))
    geometry = modules['geometry']
    charts, body_pins, _ = geometry['source_and_history']()
    require(body_pins == manifest['exact_saved_body_pins'], 'Original official body set differs')
    contracts = geometry['loaded']('contracts')
    require([(v['id'], v['group']) for v in charts] == [(v['id'], v['group']) for v in manifest['charts']],
            'Original frozen chart order differs')
    gold = []
    for chart, witness in zip(charts, manifest['charts'], strict=True):
        shown = contracts['displayed_rows'](chart)
        require(pin(geometry['canonical'](shown))['SHA256'] == witness['displayed_rows_SHA256'],
                'Frozen rendered values differ')
        gold.extend(contracts['questions'](chart))
    require([(v['id'], v['chart_id'], v['group'], v['kind'], v['event_ids'], v['question']) for v in gold]
            == [(v['id'], v['chart_id'], v['group'], v['kind'], v['event_ids'],
                 v['messages'][0]['content'].removeprefix('<image>\n')) for v in inputs],
            'Derived label identity or frozen question differs')
    parse = modules['parser']['extract_content']
    slots, diagnostics = [], []
    for index, label in enumerate(gold):
        row = dict(id=label['id'], kind=label['kind'], group=label['group'], arms={})
        detail = dict(id=label['id'], arms={})
        for arm in arms:
            decoded = parse(predictions[arm][index]['text'], label)
            correct = decoded['valid'] and decoded['normalized'] == label['answer']
            row['arms'][arm] = dict(correct=correct, disposition='completed')
            detail['arms'][arm] = dict(correct=correct, parse_valid=decoded['valid'],
                                       strict_FINAL=decoded['strict_FINAL_format'], reason=decoded['reason'])
        slots.append(row)
        diagnostics.append(detail)
    result = stats['aggregate'](slots)
    result['parse_diagnostics'] = {arm: dict(
        parsed=sum(v['arms'][arm]['parse_valid'] for v in diagnostics),
        strict_FINAL=sum(v['arms'][arm]['strict_FINAL'] for v in diagnostics)) for arm in arms}
    for baseline in ('base', 'question25'):
        comparable = [v for v in diagnostics if all(v['arms'][arm]['parse_valid'] for arm in ('domain25', baseline))]
        differences = [int(v['arms']['domain25']['correct']) - int(v['arms'][baseline]['correct']) for v in comparable]
        result['comparisons']['domain25_vs_' + baseline]['both_decodable'] = dict(
            denominator=len(differences), wins=differences.count(1), losses=differences.count(-1), ties=differences.count(0))
    result.update(state='ORIGINAL_3682_GEMMA_CONFIRMATORY96_FIRST_OFFLINE_SCORE',
                  authorized_transfer_receipt_SHA256=RECEIPT, frozen_preregistration_SHA256=PREREG,
                  prediction_files=receipt['prediction_files'], original_inner_seal=receipt['original_inner_seal'],
                  original_outer_seal=receipt['original_outer_seal'], scoring_intent=intent,
                  official_bodies_rebound=len(body_pins), label_count=len(gold), predictions_scored=288,
                  gold_labels=save(SCORE / 'labels-private.json', gold),
                  private_row_scores=save(SCORE / 'row-scores-private.json', diagnostics),
                  original_training_steps=25, this_scope_new_optimizer_updates=0)
    summary_pin = save(SCORE / 'summary.json', result)
    print(json.dumps(dict(summary=summary_pin, arms=result['arms'], comparisons=result['comparisons'],
                          confirmatory_effect_pass=result['confirmatory_effect_pass'], formal_delivery_credit=0)))


if __name__ == '__main__':
    main()
