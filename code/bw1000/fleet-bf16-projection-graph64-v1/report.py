"""Read the original eight-card resident GEMM comparison and independent busy samples."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics


ORDER = ('eager','graph','graph','eager')
NANO = 10**9
SAMPLE_OFFSETS = (1,3,5,7)


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def records(root, name, pins, jsonl=False):
    path = root/name
    need(path.is_file() and path.resolve(strict=True)==path, name+' path differs')
    with path.open('rb') as source:
        raw = source.read(131073)
    need(0<len(raw)<=131072,name+' size differs')
    pins[name] = sha256(raw).hexdigest()
    return [json.loads(line) for line in raw.splitlines()] if jsonl else json.loads(raw)


def compare(root):
    report = dict(expected_cards=8,expected_arms=32,expected_busy_card_samples=128,raw_SHA256={},
                  completed_arms=0,valid_cards=0,actual_busy_card_samples=0,passed=False,failures=[])
    pins = report['raw_SHA256']
    ready = {}
    try:
        intent = records(root,'intent.json',pins)
        node=records(root,'terminal.json',pins)
        need(all(intent[key]==node[key] for key in ('request','node','step','grid','selected_PCI'))
             and intent['planned_cards']==node['planned_cards']==8 and node['expected_arms']==32,'original placement or arm denominator differs')
        pci = intent['selected_PCI']
        grid = intent['grid']
        need(len(set(pci))==len(pci)==8 and [a[0] for a in grid]==list(ORDER)
             and all(start==grid[0][1]+offset*NANO and end-start==8*NANO
                     for (_,start,end),offset in zip(grid,(0,12,24,36),strict=True)), 'original grid or PCI differs')
        report['original_step'] = intent['step']
    except (OSError,ValueError,KeyError,TypeError) as error:
        report['failures'].append(str(error)[:120])
        return report

    for idx in range(8):
        try:
            card = records(root,f'card-{idx}.json',pins)
            shape=card['geometry']
            need(card['card']==idx and card['PCI']==pci[idx] and (shape['M'],shape['K'],shape['N'])==(184,4096,1024)
                 and shape['input_strides']==[4096,1] and shape['weight_strides']==shape['output_strides']==[1024,1]
                 and card['normal_release'] is True and card['error'] is None
                 and card['changed_and_restored_input_replay_passed'] is True
                 and type(card['capture_ns']) is int and card['capture_ns']>0 and len(card['arms'])==4,
                 'original owner, matrix or graph proof differs')
            for arm_index,(mode,scheduled,expected_end) in enumerate(grid):
                arm = records(root,f'card-{idx}-arm-{arm_index}.json',pins)
                need(card['arms'][arm_index]==arm and arm['index']==arm_index and arm['mode']==mode
                     and arm['passed'] is arm['full_BF16_equality'] is arm['CPU_first16'] is True
                     and arm['scheduled_start_epoch_ns']==scheduled and arm['scheduled_end_epoch_ns']==expected_end
                     and scheduled<=arm['actual_start_epoch_ns']<=scheduled+500_000_000
                     and expected_end-150_000_000<=arm['actual_end_epoch_ns']
                     <=arm['collection_end_epoch_ns']<=expected_end+500_000_000,
                     'arm timing or output proof differs')
                need(arm['wall_ns']==arm['collection_end_epoch_ns']-arm['actual_start_epoch_ns']>0
                     and type(arm['groups']) is int and arm['groups']>0
                     and arm['gemm_calls']==64*arm['groups'] and arm['useful_rows']==184*arm['gemm_calls']
                     and arm['h2d_payload_bytes']==arm['d2h_payload_bytes']==0
                     and math.isfinite(arm['compute_stream_ms']) and arm['compute_stream_ms']>0
                     and arm['captured_gemms_per_replay']==(64 if mode=='graph' else 0),
                     'arm completion count or byte boundary differs')
                report['completed_arms']+=1
            ready[card['PCI']]=card
        except (OSError,ValueError,KeyError,TypeError) as error:
            report['failures'].append(f'card-{idx} '+str(error)[:90])
    report['valid_cards']=len(ready)
    def rate(arm):
        return arm['gemm_calls']*NANO/arm['wall_ns']

    report['per_card_graph_over_eager_pairs']={key:[
        rate(card['arms'][b])/rate(card['arms'][a]) for a,b in ((0,1),(3,2))]
        for key,card in ready.items()}
    if len(ready)==8:
        ratios=[value for row in report['per_card_graph_over_eager_pairs'].values() for value in row]
        report['median_graph_over_eager_across_card_pairs']=statistics.median(ratios)
        report['capture_seconds_median']=statistics.median(card['capture_ns']/NANO for card in ready.values())
    report['median_GEMM_per_card_per_s'] = {
        mode:(statistics.median(rate(arm)
                                for card in ready.values() for arm in card['arms'] if arm['mode']==mode)
              if ready else None) for mode in ('eager','graph')}

    blocks={}
    try:
        for sample in records(root,'busy.jsonl',pins,jsonl=True):
            key=(sample['arm'],sample['planned_epoch_ns'])
            need(key not in blocks,'duplicate busy timestamp')
            blocks[key]=sample
    except (OSError,ValueError,KeyError,TypeError) as error:
        report['failures'].append('busy '+str(error)[:95])
    driver={'eager':[],'graph':[]}
    if len(ready)==8:
        for i,(mode,scheduled,_) in enumerate(grid):
            first=max(card['arms'][i]['actual_start_epoch_ns'] for card in ready.values())
            last=min(card['arms'][i]['actual_end_epoch_ns'] for card in ready.values())
            for sec in SAMPLE_OFFSETS:
                sample=blocks.get((i,scheduled+sec*NANO),{})
                values=sample.get('busy_percent')
                observed_begin=sample.get('observed_begin_epoch_ns')
                observed_end=sample.get('observed_end_epoch_ns')
                if (sample.get('mode')==mode and isinstance(values,dict) and set(values)==set(pci)
                        and all(type(n) is int and 0<=n<=100 for n in values.values())
                        and type(observed_begin) is type(observed_end) is int
                        and first<=observed_begin<=observed_end<=last):
                    driver[mode].extend(values.values())
    report['actual_busy_card_samples']=sum(map(len,driver.values()))
    report['driver_busy_by_arm']={mode:dict(samples=len(values),mean=statistics.mean(values) if values else None)
                                  for mode,values in driver.items()}
    report['passed']=(not report['failures'] and report['completed_arms']==32 and report['valid_cards']==8
                      and report['actual_busy_card_samples']==128 and node['completed_arms']==32 and node['valid_cards']==8)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('original_node',type=Path)
    args = parser.parse_args()
    report=compare(args.original_node.resolve(strict=True))
    print(json.dumps(report,sort_keys=True,allow_nan=False))
    return int(not report['passed'])


if __name__ == '__main__':
    raise SystemExit(main())
