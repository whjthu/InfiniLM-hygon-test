"""Check raw per-card records for one numbered shard, retaining partial evidence."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from statistics import median
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from device import BURST, ORDER, TOKENS, WINDOW, TIMING_SHA, load, need


ACTIVE = {'RUNNING','PENDING','CONFIGURING','COMPLETING'}


def read(path, hashes):
    need(path.resolve(strict=True)==path and path.is_file(),'terminal evidence path alias')
    with path.open('rb') as stream:
        raw=stream.read(65537)
    need(0<len(raw)<=65536,'terminal evidence size')
    hashes[path.name]=sha256(raw).hexdigest()
    return json.loads(raw)


def node_result(folder, node, step, scope, gold_in, gold_out):
    hashes={}
    seen=set()
    pairs=[]
    completed_arms=driver_samples=0
    errors=[]
    try:
        intent=read(folder/'intent.json',hashes)
        need(intent['step']==step and intent['node']==node and intent['request']==scope['request']
             and set(intent['PCI'])==set(scope['PCI']) and intent['planned_cards']==8,'wrong node intent')
        try:
            terminal=read(folder/'terminal.json',hashes)
            need(terminal['step']==step and terminal['node']==node and terminal['request']==scope['request'],
                 'wrong terminal node or step')
        except (OSError,KeyError,TypeError,ValueError) as error:
            terminal=None
            errors.append('node terminal: '+str(error)[:155])
        for card in range(8):
            try:
                result=read(folder/f'card-{card}.json',hashes)
                need(result['card']==card and result['PCI'] in scope['PCI'] and result['PCI'] not in seen
                     and result['planned_arms']==4 and len(result['rows'])<=4,'wrong physical card or arm denominator')
                seen.add(result['PCI'])
                rates=[]
                for index, row in enumerate(result['rows']):
                    if row['state']!='COMPLETE':
                        break
                    need((row['position'],row['heads_per_block'])==(index,ORDER[index]+1)
                         and type(row['groups']) is int and row['groups']>0
                         and row['full_submissions']==BURST*row['groups']
                         and row['completed_rows']==TOKENS*4*row['full_submissions']
                         and WINDOW<=row['wall_seconds']<=WINDOW+.5
                         and math.isfinite(row['gpu_event_ms']) and row['gpu_event_ms']>0
                         and row['output_SHA256']==gold_out and row['free_bytes']>=8*1024**3,
                         'actual useful row/event/bitwise condition failed')
                    samples=row['driver']
                    need(len(samples)==WINDOW-1 and all(sample['planned_offset_s']==ordinal
                         and ordinal<=sample['observed_offset_s']<ordinal+.5
                         and type(sample['busy_percent']) is int and 0<=sample['busy_percent']<=100
                         for ordinal,sample in enumerate(samples,1)),'actual driver sample gate failed')
                    completed_arms+=1
                    driver_samples+=len(samples)
                    rates.append(row['completed_rows']/row['wall_seconds'])
                if result['complete'] is True:
                    need(len(rates)==4 and result['input_SHA256']==gold_in,'incomplete claimed card')
                    pairs.append((rates[1]/rates[0],rates[2]/rates[3]))
            except (OSError, KeyError, TypeError, ValueError) as error:
                errors.append(f'card-{card}: {str(error)[:160]}')
        if terminal is not None:
            need((terminal['valid_cards'],terminal['completed_arms'],terminal['actual_pairs'])==
                 (len(pairs),completed_arms,2*len(pairs)), 'terminal counters differ from eight raw card records')
    except (OSError,KeyError,TypeError,ValueError) as error:
        errors.append(str(error)[:160])
    complete=len(pairs)==8 and not errors
    medians=[median(pair[index] for pair in pairs) for index in range(2)] if pairs else []
    return dict(node=node,step=step,request=scope['request'],complete=complete,planned_cards=8,
                valid_cards=len(pairs),planned_arms=32,
                completed_arms=completed_arms,actual_driver_samples=driver_samples,actual_pairs=2*len(pairs),
                device_qualified=complete,sharding_candidate=complete and min(medians)>=1.03,
                pair_order_medians=medians,raw_SHA256=hashes,errors=errors)


def reconcile(campaign, spec):
    expected=spec['scopes']
    label=spec['label']
    rows=spec['accounting_rows']
    counts=len(expected)
    result=dict(label=label,planned_nodes=counts,planned_cards=8*counts,planned_arms=32*counts,
                actual_pairs=0,actual_driver_samples=0,device_qualified=False,
                independent_clean_post_required=True,replay_allowed=False,nodes=[])
    need(1<=counts<=4 and len(rows)<=1,'invalid shard denominator or ambiguous accounting')
    if not rows:
        return dict(result,state='UNKNOWN')
    row=rows[0]
    need(len(row)==7 and row[1]==label and re.fullmatch(r'1103606\.\d+',row[0])
         and set(spec['expanded_accounting_nodes'])==set(expected),'actual accounting scope differs')
    state=row[2].split()[0]
    if state in ACTIVE:
        return dict(result,state='OBSERVED_ACTIVE')
    need(row[6] in (f'cpu={12*counts},gres/dcu={8*counts},mem={16*counts}G,node={counts}',
                     f'cpu={12*counts},gres/dcu={8*counts},mem={16384*counts}M,node={counts}'),
         'actual allocation resource differs')
    reference=load('timing.py',TIMING_SHA)
    gold_in,gold_out=reference.fixed_hashes(TOKENS)
    observed=[node_result(campaign/f'result-{node}',node,row[0],scope,gold_in,gold_out)
              for node,scope in sorted(expected.items())]
    return dict(result,state=state,actual_accounting=row,nodes=observed,
                actual_pairs=sum(item['actual_pairs'] for item in observed),
                actual_driver_samples=sum(item['actual_driver_samples'] for item in observed),
                device_qualified=state=='COMPLETED' and row[3]=='0:0'
                                 and all(item['device_qualified'] for item in observed))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--actual',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(reconcile(args.campaign.resolve(strict=True),json.loads(args.actual.read_bytes())),
                     sort_keys=True,allow_nan=False))
