"""Independently verify the original two-step AdamW grouping experiment."""
import datetime
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

ROOT=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/fleet-qwen-adamw-group-size-v1-8ca6c3ac4f6a497ea825c533b11e7f8d')
REQUEST='8ca6c3ac4f6a497ea825c533b11e7f8d'
PARENTS={'1087656':(39,'1087656.441'),'1103606':(38,'1103606.237')}
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
    assert intent['request']==REQUEST and intent['planned_cards']==616
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
    planned={i:[] for i in WINDOWS}
    failures=[]
    ratios=[]
    wall=[]
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
                for card in range(8):
                    row=read(base/f'card-{card}.json')
                    assert row['node']==node and row['step']==step and row['card']==card and row['error'] is None
                    assert row['pci'] in pins and row['pci'] not in node_pci
                    node_pci.add(row['pci'])
                    result,=row['results']
                    assert result['changed_tensors']==result['tensors']==504 and result['elements']==43646976
                    assert len(result['numeric'])==6 and all(0<=n<=5e-6 for row_num in result['numeric'] for n in row_num['max_abs'].values())
                    blocks=result['blocks']
                    assert [b['group_size'] for b in blocks]==[504,64,64,504]
                    assert all(b['timed_updates']>=20 and b['event_p50_ms']>0 and b['updates_per_second']>0 for b in blocks)
                    assert node_pci<=pins
                    ratios.append(statistics.mean(b['event_p50_ms'] for b in blocks[1:3])/
                                  statistics.mean(blocks[i]['event_p50_ms'] for i in (0,3)))
                    wall.append(statistics.mean(b['updates_per_second'] for b in blocks[1:3])/
                                statistics.mean(blocks[i]['updates_per_second'] for i in (0,3)))
                assert node_pci==pins
                statuses[node]=True
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
    n=len(ratios)
    success=all(r[2:4]==['COMPLETED','0:0'] for r in accounts.values()) and n==616 and len(statuses)==77 and not failures and not terminal['errors']
    return dict(utc=time,state='TERMINAL_PASS' if success else 'TERMINAL_WITH_GAPS',
                accounting_columns=columns.split(','),accounting=accounts,allocated_cards=640,
                verified_cards=n,worker_post_clean_nodes=len(statuses),failures=failures,
                launch_epoch=intent['epoch'],events_chunked_over_unpartitioned_median=statistics.median(ratios) if n else None,
                updates_per_second_chunked_over_unpartitioned_median=statistics.median(wall) if n else None,
                windows=[dict(offset_seconds=offset,samples=len(a),covered_cards=len(a)//3,
                              mean_busy_percent=statistics.mean(a) if a else None) for offset,a in planned.items()],
                private_inputs=len(HASHES),private_input_manifest_SHA256=hashlib.sha256(
                    json.dumps(HASHES,sort_keys=True,separators=(',',':')).encode()).hexdigest())


if __name__=='__main__':
    print(json.dumps(main(),allow_nan=False),flush=True)
