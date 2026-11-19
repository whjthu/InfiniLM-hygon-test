"""Independently verify the original two-step AdamW group-width sweep."""
import datetime
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

ROOT=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/fleet-qwen-adamw-size-sweep-v1-7a0d5a94fb424deebc7787ecb8a2ef06')
REQUEST='7a0d5a94fb424deebc7787ecb8a2ef06'
PARENTS={'1087656':(39,'1087656.444'),'1103606':(39,'1103606.244')}
WINDOWS={120:'busy.json',240:'busy-240.json',380:'busy-380.json',
         520:'busy-520.json',660:'busy-660.json'}
HASHES={}


def read(path, max_bytes=32768):
    path=ROOT/path
    with path.open('rb') as stream:
        raw=stream.read(max_bytes+1)
    assert 0<len(raw)<=max_bytes,(path.name,'bound')
    HASHES[path.relative_to(ROOT).as_posix()]=hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def main():
    time=datetime.datetime.now(datetime.timezone.utc).isoformat()
    intent=read('intent.json')
    assert intent['request']==REQUEST and intent['planned_cards']==624
    columns='JobIDRaw,JobName%100,State%40,ExitCode,ElapsedRaw,AllocTRES%160'
    accounts={}
    for parent,(count,step) in PARENTS.items():
        raw=subprocess.check_output(['sacct','-n','-P','-j',parent,'--format='+columns],timeout=12,text=True)
        rows=[line.split('|') for line in raw.splitlines() if line.split('|')[0]==step]
        assert len(rows)==1 and rows[0][1]==intent['labels'][parent]
        accounts[parent]=rows[0]
    if any(r[2].split()[0] in ('RUNNING','COMPLETING','PENDING','CONFIGURING') for r in accounts.values()):
        return dict(utc=time,state='ACTIVE',accounting=accounts)
    terminal=read('dispatch-terminal.json')
    assert terminal['request']==REQUEST
    assert terminal['accounts']=={p:[r] for p,r in accounts.items()} and terminal['returncodes']=={p:0 for p in PARENTS}
    planned={i:[] for i in WINDOWS}
    failures=[]
    ratios={32:[],128:[]}
    wall={32:[],128:[]}
    winner_counts={32:0,64:0,128:0}
    card_count=0
    statuses={}
    seen=set()
    for parent,(count,step) in PARENTS.items():
        nodes=intent['parents'][parent]
        assert len(nodes)==len(set(nodes))==count and not (set(nodes)&seen)
        seen.update(nodes)
        for node in nodes:
            base=Path('nodes')/node
            try:
                inv=read(base/'inventory.json')
                pins=set(inv['PCI_KFD'])
                post=read(base/'terminal.json')
                assert (inv['node'],inv['step'],post['node'],post['step'])==(node,step,node,step)
                assert len(pins)==len({v['gpu_id'] for v in inv['PCI_KFD'].values()})==8
                assert post['post_kfd_holders']==0 and post['error'] is None and post['exits']=={str(i):0 for i in range(8)}
                node_pci=set()
                per_node=[]
                for card in range(8):
                    row=read(base/f'card-{card}.json')
                    assert row['node']==node and row['step']==step and row['card']==card and row['error'] is None
                    assert row['pci'] in pins and row['pci'] not in node_pci
                    node_pci.add(row['pci'])
                    result,=row['results']
                    assert result['changed_tensors']==result['tensors']==504 and result['elements']==43646976
                    numeric=result['numeric']
                    sizes=(32,64,128)
                    assert len(numeric)==9 and {(r['step'],r['arm']) for r in numeric}=={(step,size) for step in range(1,4) for size in sizes}
                    assert all(set(r['max_abs'])=={'parameter','exp_avg','exp_avg_sq'} and
                               all(0<=delta<=5e-6 for delta in r['max_abs'].values()) for r in numeric)
                    blocks=result['blocks']
                    ordinal=result['card_index']
                    assert type(ordinal) is int and 0<=ordinal<count*8 and ordinal%8==card
                    offset=ordinal%3
                    order=sizes[offset:]+sizes[:offset]
                    assert [b['group_size'] for b in blocks]==list(order+order[::-1])
                    assert all(b['timed_updates']>=20 and b['event_p50_ms']>0 and b['event_p95_ms']>=b['event_p50_ms']
                               and b['updates_per_second']>0 and 85<=b['wall_seconds']<=95
                               and b['groups']==(504+b['group_size']-1)//b['group_size'] for b in blocks)
                    events={s:statistics.mean(b['event_p50_ms'] for b in blocks if b['group_size']==s) for s in sizes}
                    updates={s:statistics.mean(b['updates_per_second'] for b in blocks if b['group_size']==s) for s in sizes}
                    per_node.append((ordinal,events,updates))
                assert node_pci==pins
                post_ordinals=[entry[0] for entry in per_node]
                assert len(post_ordinals)==8 and len({o//8 for o in post_ordinals})==1
                statuses[node]=(parent,post_ordinals[0]//8,per_node)
                for offset,name in WINDOWS.items():
                    try:
                        sample=read(base/name)
                    except FileNotFoundError:
                        continue
                    assert sample['node']==node and sample['step']==step and len(sample['rows'])==3
                    for point,row in enumerate(sample['rows']):
                        scheduled=intent['epoch']+offset+point*2
                        assert row['point']==point and scheduled<=row['opened_epoch']<=row['closed_epoch']<=scheduled+1
                        assert set(row['busy'])==pins and all(type(x) is int and 0<=x<=100 for x in row['busy'].values())
                    planned[offset].extend(x for row in sample['rows'] for x in row['busy'].values())
            except (AssertionError,FileNotFoundError,ValueError,KeyError,TypeError) as exc:
                failures.append(dict(node=node,error=type(exc).__name__))
    for parent,(count,_) in PARENTS.items():
        ordinals=[ordinal for p,ordinal,_ in statuses.values() if p==parent]
        if sorted(ordinals)!=list(range(count)):
            failures.append(dict(parent=parent, error='INCOMPLETE_UNIQUE_NODE_ORDINALS'))
    for _,_,per_node in statuses.values():
        for _,events,updates in per_node:
            card_count+=1
            winner_counts[max(updates,key=updates.get)]+=1
            for mode in ratios:
                ratios[mode].append(events[mode]/events[64])
                wall[mode].append(updates[mode]/updates[64])
    n=card_count
    success=all(r[2:4]==['COMPLETED','0:0'] for r in accounts.values()) and n==624 and len(statuses)==78 and not failures and not terminal['errors']
    return dict(utc=time,state='TERMINAL_PASS' if success else 'TERMINAL_WITH_GAPS',
                accounting_columns=columns.split(','),accounting=accounts,allocated_cards=640,
                verified_cards=n,worker_post_clean_nodes=len(statuses),failures=failures,
                launch_epoch=intent['epoch'],normalized_median_event_ms_vs_64={s:statistics.median(v) for s,v in ratios.items() if v},
                normalized_median_updates_per_second_vs_64={s:statistics.median(v) for s,v in wall.items() if v},
                fastest_wall_update_rate_cards=winner_counts,
                windows=[dict(offset_seconds=offset,samples=len(a),covered_cards=len(a)//3,
                              mean_busy_percent=statistics.mean(a) if a else None) for offset,a in planned.items()],
                private_inputs=len(HASHES),private_input_manifest_SHA256=hashlib.sha256(
                    json.dumps(HASHES,sort_keys=True,separators=(',',':')).encode()).hexdigest())


if __name__=='__main__':
    print(json.dumps(main(),allow_nan=False),flush=True)
