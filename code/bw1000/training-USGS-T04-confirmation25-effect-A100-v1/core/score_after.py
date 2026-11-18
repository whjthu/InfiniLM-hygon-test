"""Separate post-generation process: seal all predictions before opening confirmation gold.

Uses the unchanged published displayed-value scorer. No model/device/network.
Private per-slot scores and raw predictions never go into Git.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from effect_contract import ARMS, dump, eos_observation, need, pinned, scoring_prediction

D = Path(__file__).resolve().parent


def main(a):
    plan = json.loads((D / 'PLAN.json').read_bytes())
    final = json.loads(pinned(a.run / 'generation-finalized.json', dict(bytes=a.finalized_bytes, SHA256=a.finalized_sha)))
    need(final['arms'] == list(ARMS) and final['confirmation_denominator_each'] == 25
         and final['input_SHA256'] == plan['confirmation_inputs']['SHA256']
         and final['all_prediction_files_closed_before_gold'] is True
         and final['confirmation_gold_read'] is False and final['confirmation_inputs_and_media_read'] is True,
         'PREDICTIONS_MUST_BE_FINALIZED_BEFORE_GOLD')
    # A separate actual terminal consumer must pin/validate all75 outputs and owned post first.
    terminal=json.loads(pinned(a.terminal,dict(bytes=a.terminal_bytes,SHA256=a.terminal_sha)))
    need(terminal['independent_passed'] is True and terminal['planned_generations']==75
         and type(terminal['observed_predictions']) is int and 0<=terminal['observed_predictions']<=75
         and terminal['confirmation_gold_read'] is False
         and terminal['generation_finalized_SHA256']==a.finalized_sha,'ACTUAL_INDEPENDENT_CONFIRMATION75_TERMINAL_REQUIRED')
    predictions = {}
    # All three finished or partial prediction files are pinned/read before gold.
    # A failed arm with no file contributes twentyfive missing slots, with no retry/backfill.
    for arm in ARMS:
        record = final['arm_results'][arm]
        need(record['planned'] == 25 and 0 <= record['completed'] <= record['attempted'] <= 25,
             'FULL_ARM_DENOMINATOR_REQUIRED')
        values = []
        if 'file' in record:
            need(record['file'] == arm + '-predictions-private.jsonl', 'EXACT_ARM_FILE_REQUIRED')
            values = [json.loads(line) for line in pinned(a.run / record['file'], record).splitlines()]
        need(len(values) == record['completed'], 'PARTIAL_PREDICTION_COUNT_MISMATCH')
        for row in values:
            need(row['arm'] == arm and row['native_field_identity_verified'] is True, 'NATIVE_INPUT_OR_ARM_MISMATCH')
            observed = eos_observation(row['raw_generated_token_ids'], row['raw_EOS_ids'])
            need(all(row[k] == v and type(row[k]) is type(v) for k, v in observed.items()),
                 'NATURAL_EOS_OR_CAP_METADATA_NOT_DERIVED_FROM_ACTUAL_IDS')
        predictions[arm] = [scoring_prediction(row) for row in values]
    need(sum(len(v) for v in predictions.values())==terminal['observed_predictions'],'INDEPENDENT_OBSERVED75_DENOMINATOR')
    if not (final['passed'] is True and terminal['engineering_completed'] is True
            and terminal['all75_prediction_files_pinned_and_verified'] is True
            and all(len(predictions[arm]) == 25 for arm in ARMS)):
        # Seventyfive real outputs must be durably sealed before inspecting any gold.
        # Missing attempts remain in each twentyfive-slot denominator without opening labels.
        result = dict(state='PREDICTION_FAILURE_NO_GOLD_OPENED', confirmation_gold_read=False, confirmation_opened=False,
            arms={arm: dict(denominator=25, real_predictions=len(predictions[arm]),
                missing=25-len(predictions[arm]), correct=None, accuracy=None) for arm in ARMS},
            domain_original_training_accepted=False, Model_GO=False)
        dump(a.output, result); print(json.dumps(result)); return
    need(final['all_predictions_finalized_before_gold'] is True, 'ALL_SEVENTYFIVE_PREDICTIONS_SEAL_REQUIRED')
    # This is the first point where any label file may be read.
    manifest = json.loads(pinned(a.dataset / 'manifest-private.json', plan['dataset_manifest']))
    label_pin = manifest['outputs']['confirmation/labels-private.jsonl']
    need(label_pin == plan['confirmation_labels'], 'FROZEN_CONFIRMATION_LABEL_IDENTITY_CHANGED')
    labels = [json.loads(line) for line in pinned(a.dataset / 'confirmation/labels-private.jsonl', label_pin).splitlines()]
    need(len(labels) == 25, 'ALL_TWENTYFIVE_GOLD_SLOTS_REQUIRED')
    for name, pin in plan['frozen_scorer'].items():
        pinned(D / 'frozen_scorer' / name, pin)
    sys.path.insert(0, str(D / 'frozen_scorer'))
    spec = importlib.util.spec_from_file_location('published_USGS_score_once', D / 'frozen_scorer/score_once.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    results = {arm: module.score(labels, predictions[arm]) for arm in ARMS}
    # Pair by immutable chart and event-closure groups, using only the pre-QA split plan.
    raw=(a.dataset/'declaration-private.json').read_bytes()
    need(__import__('hashlib').sha256(raw).hexdigest()==manifest['declaration_SHA256'],'FROZEN_SPLIT_PLAN_SHA')
    split=json.loads(raw);chart_group={c['id']:c['group'] for c in split['charts'] if c['split']=='confirmation'}
    need(len(chart_group)==5 and len(set(chart_group.values()))==5,'FROZEN_FIVE_CHART_FIVE_GROUP_DENOMINATOR')
    paired={}
    for arm,record in results.items():
        by_chart={};by_group={}
        for row in record['rows']:
            chart=row['chart_id'];need(chart in chart_group,'NO_FOREIGN_CONFIRMATION_CHART')
            for dest,key in [(by_chart,chart),(by_group,str(chart_group[chart]))]:
                item=dest.setdefault(key,dict(denominator=0,correct=0));item['denominator']+=1;item['correct']+=int(row['correct'])
        paired[arm]=dict(by_chart=by_chart,by_closure_group=by_group)

    for arm in ARMS:
        need(results[arm]['denominator'] == 25, 'NEVER_DROP_FAILED_OR_MISSING_CONFIRMATION_SLOTS')
    private = dict(metric='Frozen displayed-value task accuracy; optional single FINAL prefix removed identically in all arms',
        arms=results, paired_chart_group=paired, domain_original_training_accepted=False, confirmation_opened=True, confirmation_gold_read=True,
        formal17=False, Model_GO=False, domain_capability_qualified=False)
    dump(a.output, private)
    print(json.dumps(dict(arms={arm: {k: v for k, v in result.items() if k not in ['rows']}
        for arm, result in results.items()}, domain_original_training_accepted=False, confirmation_opened=True,
        strict_FINAL_counts={arm: sum(row['strict_FINAL_format'] for row in predictions[arm]) for arm in ARMS}), allow_nan=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('run', 'dataset', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--terminal',type=Path,required=True);p.add_argument('--terminal-bytes',type=int,required=True);p.add_argument('--terminal-sha',required=True)
    p.add_argument('--finalized-bytes', type=int, required=True); p.add_argument('--finalized-sha', required=True)
    main(p.parse_args())
