"""Bounded metadata-only original terminal consumer; no model or old run reads."""
import argparse
import json
from pathlib import Path
import re
import time

from split_contract import LABEL, REQUEST, PARENT, NODE, sha, need, verify_steps


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
        planned_updates=25,planned_predictions=4,planned_metric_pairs=2,actual_updates=None,
        source_manifest_SHA256=intent['source_manifest_SHA256'],claim_SHA256=intent['claim_SHA256'],
        candidate_relative_holdout_only=True,global_fresh_blind=False,formal_CO3D=False,
        formal_performance=False,model_GO=False,files=pins)
    for phase in ('pre','gpu','post'):
        if phase not in outcome['stages']:
            result.update(state='PARTIAL_OR_FAILED',missing_phase=phase,logical_read_bytes=total)
            return result
        row=actual[phase]
        need(type(row) is list and len(row)==7 and re.fullmatch(PARENT+r'\.[0-9]+',row[0]) and
             row[1]==LABEL+'-'+phase and row[5]==NODE and
             outcome['stages'][phase]['row']==row, 'original exact phase accounting differs')
        if row[2:4]!=['COMPLETED','0:0']:
            path=root/'arms/continuous/train/steps.ndjson'
            if path.is_file():
                with path.open('rb') as stream:
                    raw=stream.read((1<<20)+1)
                need(len(raw)<=1<<20, 'partial updates exceeded cap')
                rows=[json.loads(line) for line in raw.splitlines()]
                need([r['global_optimizer_step'] for r in rows]==list(range(1,len(rows)+1)) and len(rows)<=25,
                     'partial step prefix differs')
                result['actual_updates']=len(rows)
                pins['arms/continuous/train/steps.ndjson']=dict(bytes=len(raw),SHA256=sha(raw))
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
         native['metrics']['cached_valid_batches']==7 and native['metrics']['sequence_count']==1 and
         native['checks']['camera_head_weights_changed'] is True and
         native['checks']['frozen_aggregator_unchanged'] is True, 'positive-update/frozen-backbone proof missing')
    metric=doc('arms/continuous/evaluation/metric-private.json')
    predictions=doc('arms/continuous/evaluation/predictions-sealed-private.json')
    need(pins['arms/continuous/evaluation/predictions-sealed-private.json']==metric['prediction_seal'] and
         len(predictions['rows'])==4 and predictions['gold_read'] is False and
         metric['summary']['pairs']==2 and metric['summary']['sequences']==2 and
         metric['planned_predictions']==4 and metric['planned_pairs']==2 and
         receipt['selection_SHA256']==metric['selection_SHA256'], 'all sealed cross-sequence metric slots required')
    result.update(state='SEVEN_PAIR25_AND_RECORDING_HELDOUT_METRIC_COMPLETE',qualified=True,
        actual_updates=25,actual_prediction_slots=4,metric_pairs=2,metric=metric['summary'],
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
