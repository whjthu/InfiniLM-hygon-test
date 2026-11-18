"""Separate post-generation process: seal all predictions before opening dev gold.

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
    need(final['arms'] == list(ARMS) and final['dev_denominator_each'] == 10
         and final['input_SHA256'] == plan['dev_inputs']['SHA256']
         and final['all_prediction_files_closed_before_gold'] is True
         and final['dev_gold_read'] is False and final['blind_inputs_gold_or_media_read'] is False,
         'PREDICTIONS_MUST_BE_FINALIZED_BEFORE_GOLD')
    predictions = {}
    # All three finished or partial prediction files are pinned/read before gold.
    # A failed arm with no file contributes ten missing slots, with no retry/backfill.
    for arm in ARMS:
        record = final['arm_results'][arm]
        need(record['planned'] == 10 and 0 <= record['completed'] <= record['attempted'] <= 10,
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
    if not (final['passed'] is True and all(len(predictions[arm]) == 10 for arm in ARMS)):
        # Thirty real outputs must be durably sealed before inspecting any gold.
        # Missing attempts remain in each ten-slot denominator without opening labels.
        result = dict(state='PREDICTION_FAILURE_NO_GOLD_OPENED', dev_gold_read=False, blind_opened=False,
            arms={arm: dict(denominator=10, real_predictions=len(predictions[arm]),
                missing=10-len(predictions[arm]), correct=None, accuracy=None) for arm in ARMS},
            domain_original_training_accepted=False, Model_GO=False)
        dump(a.output, result); print(json.dumps(result)); return
    need(final['all_predictions_finalized_before_gold'] is True, 'ALL_THIRTY_PREDICTIONS_SEAL_REQUIRED')
    # This is the first point where any label file may be read.
    manifest = json.loads(pinned(a.dataset / 'manifest-private.json', plan['dataset_manifest']))
    label_pin = manifest['outputs']['dev/labels-private.jsonl']
    need(label_pin == plan['dev_labels'], 'FROZEN_DEV_LABEL_IDENTITY_CHANGED')
    labels = [json.loads(line) for line in pinned(a.dataset / 'dev/labels-private.jsonl', label_pin).splitlines()]
    need(len(labels) == 10, 'ALL_TEN_GOLD_SLOTS_REQUIRED')
    for name, pin in plan['frozen_scorer'].items():
        pinned(D / 'frozen_scorer' / name, pin)
    sys.path.insert(0, str(D / 'frozen_scorer'))
    spec = importlib.util.spec_from_file_location('published_USGS_score_once', D / 'frozen_scorer/score_once.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    results = {arm: module.score(labels, predictions[arm]) for arm in ARMS}
    for arm in ARMS:
        need(results[arm]['denominator'] == 10, 'NEVER_DROP_FAILED_OR_MISSING_DEV_SLOTS')
    private = dict(metric='Frozen displayed-value task accuracy; optional single FINAL prefix removed identically in all arms',
        arms=results, domain_original_training_accepted=False, blind_opened=False,
        formal17=False, Model_GO=False, domain_capability_qualified=False)
    dump(a.output, private)
    print(json.dumps(dict(arms={arm: {k: v for k, v in result.items() if k not in ['rows']}
        for arm, result in results.items()}, domain_original_training_accepted=False, blind_opened=False,
        strict_FINAL_counts={arm: sum(row['strict_FINAL_format'] for row in predictions[arm]) for arm in ARMS}), allow_nan=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('run', 'dataset', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--finalized-bytes', type=int, required=True); p.add_argument('--finalized-sha', required=True)
    main(p.parse_args())
