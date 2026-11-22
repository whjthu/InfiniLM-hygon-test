"""Bounded metadata-only original terminal consumer; no model or old run reads."""
import argparse
import json
from pathlib import Path
import re
import time

from split_contract import LABEL, REQUEST, PARENT, NODE, TRAIN, sha, need, verify_steps
from view_schedule import VIEW_BATCHES, validate_observed_views, view_order_sha256


def consume(root, actual, independent_post):
    end = time.monotonic()+40
    pins, total = {}, 0
    def doc(name, cap=1<<20):
        nonlocal total
        need(time.monotonic()<end, 'reader deadline exhausted')
        path=root/name
        need(path.is_file() and path.resolve(strict=True)==path, 'bounded unaliased original required')
        with path.open('rb') as stream:
            raw=stream.read(cap+1)
        total+=len(raw)
        need(0<len(raw)<=cap and total<=6<<20, 'reader byte bound exceeded')
        pins[name]=dict(bytes=len(raw),SHA256=sha(raw))
        return json.loads(raw)
    need(root.name==LABEL, 'wrong original namespace')
    intent=doc('intent.json')
    need((intent['request'],intent['parent'],intent['node'])==(REQUEST,PARENT,NODE) and
         intent['deadline']-intent['epoch']==1800, 'wrong original intent/epoch')
    outcome=doc('outcome.json')
    need(outcome['request']==REQUEST and outcome['publication']==intent['publication'], 'outcome context differs')
    result=dict(request=REQUEST,publication=intent['publication'],qualified=False,
        planned_updates=25,planned_predictions=4,planned_metric_pairs=2,actual_updates=0,missing_updates=25,
        actual_prediction_slots=0,missing_prediction_slots=4,independent_train_pairs=7,derived_views=VIEW_BATCHES,
        source_manifest_SHA256=intent['source_manifest_SHA256'],claim_SHA256=intent['claim_SHA256'],
        candidate_relative_holdout_only=True,global_fresh_blind=False,formal_CO3D=False,
        formal_performance=False,model_GO=False,files=pins)
    def count_partial():
        nonlocal total
        steps_name = 'arms/continuous/train/steps.ndjson'
        steps = root / steps_name
        if not steps.is_file():
            return
        need(steps.resolve(strict=True) == steps, 'original partial step path differs')
        with steps.open('rb') as stream:
            raw = stream.read((1 << 20) + 1)
        total += len(raw)
        need(len(raw) <= 1 << 20 and total <= 6 << 20, 'partial updates exceeded cap')
        rows = [json.loads(line) for line in raw.splitlines()]
        need([r['global_optimizer_step'] for r in rows] == list(range(1, len(rows) + 1)) and len(rows) <= 25,
             'partial step prefix differs')
        result.update(actual_updates=len(rows), missing_updates=25-len(rows))
        pins[steps_name] = dict(bytes=len(raw), SHA256=sha(raw))
        name = 'arms/continuous/evaluation/predictions-sealed-private.json'
        if (root/name).is_file():
            predictions = doc(name)
            need(predictions.get('gold_read') is False and type(predictions.get('rows')) is list
                 and len(predictions['rows']) <= 4, 'partial prediction denominator differs')
            count = len(predictions['rows'])
            result.update(actual_prediction_slots=count, missing_prediction_slots=4-count)

    for phase in ('pre','gpu','post'):
        if phase not in outcome['stages']:
            count_partial()
            result.update(state='PARTIAL_OR_FAILED',missing_phase=phase,logical_read_bytes=total)
            return result
        row=actual[phase]
        need(type(row) is list and len(row)==7 and re.fullmatch(PARENT+r'\.[0-9]+',row[0]) and
             row[1]==LABEL+'-'+phase and row[5]==NODE and
             outcome['stages'][phase]['row']==row, 'original exact phase accounting differs')
        if row[2:4]!=['COMPLETED','0:0']:
            count_partial()
            result.update(state='ACTUAL_FAILED',failed_phase=phase,accounting=actual,logical_read_bytes=total)
            return result
    before,after=doc('before.json'),doc('after.json')
    clean=independent_post['nodes'][NODE]
    need(before['pci']==after['pci'] and len(before['pci'])==8 and before['boot_id']==after['boot_id'] and
         all(x['full_fd_inventory_performed'] is True and x['kfd_holder_count']==0 for x in (before,after)) and
         clean['parent']==PARENT and set(clean['PCIs'])==set(before['pci']) and
         clean['active_steps']==[] and clean['KFD_clean'] is clean['PID_clean'] is True and
         clean['device_probe_observed'] is True and
         re.fullmatch(PARENT+r'\.[0-9]+',clean['device_probe_step']) and
         int(clean['device_probe_step'].split('.')[-1])>int(actual['gpu'][0].split('.')[-1]),
         'independent same-parent device/process release not proven')
    combined=doc('native-result.json')
    need(pins['native-result.json']['SHA256']==outcome['result_SHA256'] and outcome['passed'] is True,
         'native/outcome seal differs')
    receipt=doc('arms/continuous/receipt.json')
    need(combined['receipts']=={'continuous':dict(SHA256=pins['arms/continuous/receipt.json']['SHA256'],value=receipt)} and
         receipt['epoch']==intent['epoch'] and receipt['deadline']==intent['deadline']-100 and
         receipt['executed_optimizer_steps']==25, 'original native receipt differs')
    verify_steps(receipt['steps'])
    native=doc('arms/continuous/native-result.json')
    need(pins['arms/continuous/native-result.json']['SHA256']==receipt['result_SHA256'] and
         native['metrics']['global_optimizer_step']==native['metrics']['executed_optimizer_steps']==25 and
         native['metrics']['cached_valid_batches']==VIEW_BATCHES and native['metrics']['sequence_count']==1 and
         native['checks']['camera_head_weights_changed'] is True and
         native['checks']['frozen_aggregator_unchanged'] is True, 'positive-update/frozen-backbone proof missing')
    cohort=native['metrics']['domain_geometry']
    need(receipt['unique_training_pairs']==cohort['verified_pairs']==7 and
         receipt['derived_training_views']==cohort['derived_views']==VIEW_BATCHES and
         receipt['view_order_SHA256']==cohort['view_order_SHA256']==view_order_sha256() and
         cohort['physical_recordings']=={TRAIN:7} and cohort['heldout_geometry_loaded_by_training'] is False,
         'original physical training pairs or view order differ')
    validate_observed_views(cohort['rows'])
    metric=doc('arms/continuous/evaluation/metric-private.json')
    predictions=doc('arms/continuous/evaluation/predictions-sealed-private.json')
    need(pins['arms/continuous/evaluation/predictions-sealed-private.json']==metric['prediction_seal'] and
         len(predictions['rows'])==4 and predictions['gold_read'] is False and
         metric['summary']['pairs']==2 and metric['summary']['sequences']==2 and
         metric['planned_predictions']==4 and metric['planned_pairs']==2 and
         receipt['selection_SHA256']==metric['selection_SHA256'], 'all sealed cross-sequence metric slots required')
    result.update(state='FOURTEEN_VIEW25_AND_RECORDING_HELDOUT_METRIC_COMPLETE',qualified=True,
        actual_updates=25,missing_updates=0,actual_prediction_slots=4,missing_prediction_slots=0,metric_pairs=2,metric=metric['summary'],
        independent_post_step=clean['device_probe_step'],post_clean=True,logical_read_bytes=total)
    need(len(json.dumps(result))<16384, 'bounded public summary exceeded')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--actual-accounting',type=Path,required=True)
    parser.add_argument('--independent-post',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(consume(args.root,json.loads(args.actual_accounting.read_bytes()),
        json.loads(args.independent_post.read_bytes())),sort_keys=True,allow_nan=False))
