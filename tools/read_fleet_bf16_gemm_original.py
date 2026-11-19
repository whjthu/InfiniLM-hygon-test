"""Read only the original fleet BF16 run's named receipts and accounting."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import subprocess


HELPER = Path('C:/Users/WangHaojie/.codex/visualizations/2026/10/12/01a130bd-3b97-73c2-8ecf-eda5151f1ba7/glm-paired-readonly-connection.py')
HELPER_SHA = '3f00e0eac9b40eb3fee7d788776389bc22c3e87d20de0441312f04a2566178d1'
REPO = Path(__file__).resolve().parents[1]
LABEL = 'fleet-bf16-gemm-uniformity-v1'
REQUEST = '2f160677a4834c0c868611655a8c0817'
STEPS = {'1087656':'1087656.370','1103606':'1103606.197'}
REMOTE = '''
import collections,datetime,hashlib,json,math,pathlib,statistics,subprocess,time

R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')/(LABEL+'-'+REQUEST)
receipt_pins={}
def read(path):
    if not path.is_file():
        return None
    with path.open('rb') as f:
        raw=f.read(262145)
    if len(raw)>262144:
        raise ValueError('BOUNDED_RECEIPT')
    receipt_pins[path.relative_to(R).as_posix()]=hashlib.sha256(raw).hexdigest()
    return json.loads(raw)

def samples(rows, lower, upper):
    return [row for row in rows if lower<=row['epoch']<=upper and len(row['busy'])==8
            and all(isinstance(x,int) and 0<=x<=100 for x in row['busy'].values())]

def busy_summary(by_node):
    values=[value for rows in by_node.values() for row in rows for value in row['busy'].values()]
    return dict(nodes=len(by_node),cards=8*len(by_node),samples=len(values),
                mean_percent=statistics.mean(values) if values else None,
                samples_strictly_over80=sum(v>80 for v in values),
                first_epoch=min((r['epoch'] for rows in by_node.values() for r in rows),default=None),
                last_epoch=max((r['epoch'] for rows in by_node.values() for r in rows),default=None))

intent=read(R/'intent.json')
assert intent and intent['request']==REQUEST and intent['cards']==640
expected=[n for hostlist in intent['parents'].values() for n in hostlist]
assert len(expected)==len(set(expected))==80
now=time.time()
ready={}; live={}; done={}; unavailable=[]; faults=collections.Counter()
accounts={}
cols='JobIDRaw,JobName%100,State%40,ExitCode,ElapsedRaw,AllocTRES%160'
for parent,step in STEPS.items():
    p=subprocess.run(['sacct','-n','-P','-j',step,'--format='+cols],capture_output=True,text=True,check=True,timeout=8)
    accounts[parent]=[row for line in p.stdout.splitlines() if len(row:=line.split('|'))==6 and
                      row[0]==step and row[1]==intent['labels'][parent]]
    for node in intent['parents'][parent]:
        def load(kind):
            row=read(R/'nodes'/(kind+'-'+node+'.json'))
            if row is not None and (row['step']!=step or row['node']!=node):
                raise ValueError('NODE_AND_STEP_BINDING')
            return row
        row=load('ready')
        if row is not None and row['visible']==8:
            ready[node]=row['epoch']
        row=load('live')
        if row is not None and (sample:=samples(row['samples'],now-22,now+2)):
            live[node]=sample
        row=load('final')
        if row is not None:
            assert row['planned_cards']==8 and row['launch_epoch']==intent['epoch']
            done[node]=row
            for error in row['error']:
                faults[error['type']]+=1

shapes={(m,4096,n) for m in (1,16,128,512) for n in (1024,4096,8192)}
reference=[]; rates_by_node={}; paired=collections.defaultdict(list); qualified_cards=0
for node,row in done.items():
    if len({c['device'] for c in row['cards']})!=len(row['cards']):
        raise ValueError('DUPLICATE_LOGICAL_DCU')
    for card in row['cards']:
        assert 0<=card['device']<8 and card['peak_allocated_bytes']>0
        steady=card['steady']
        assert steady['shape']==[4096,4096,4096] and len(steady['steady'])==12
        assert all(v['gemm_calls']>0 and v['wall_seconds']>=8 for v in steady['steady'])
        values=[v['completed_TFLOPS'] for v in steady['steady']]
        assert all(math.isfinite(v) and v>0 for v in values)
        measurements=card['measurements']
        assert {tuple(r['shape']) for r in measurements}==shapes and len(measurements)==len(shapes)
        for m in measurements:
            assert m['sample_dots']>0 and all(len(seq)==9 and all(math.isfinite(v) and v>0 for v in seq)
                                              for seq in m['samples_ms'].values())
            paired['x'.join(map(str,m['shape']))].append(m['median_ms']['reuse']/m['median_ms']['allocate'])
        median=statistics.median(values)
        reference.append(median)
        rates_by_node.setdefault(node,[]).append(median)
        qualified_cards+=1

joint={}; bounds=None
if len(ready)==80 and len(done)==80:
    earliest=max(ready.values())+3
    latest=min(ready.values())+90
    if earliest<=latest:
        bounds=[earliest,latest]
        for node,row in done.items():
            if s:=samples(row['driver_busy'],earliest,latest):
                joint[node]=s
low_nodes=[dict(node=node,median_TFLOPS=statistics.median(vals)) for node,vals in rates_by_node.items() if len(vals)==8]
low_nodes.sort(key=lambda entry:entry['median_TFLOPS'])
result=dict(observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),request=REQUEST,
    epoch=intent['epoch'],expected_cards=640,accounting_columns=cols.split(','),actual_steps=accounts,
    ready_nodes=len(ready),initialized_cards=len(ready)*8,current_recent_samples=busy_summary(live),
    final_nodes=len(done),completed_cards=qualified_cards,error_counts=dict(faults),
    candidate_common_busy_window=bounds,common_busy=busy_summary(joint),clock_sync='NOT_INDEPENDENTLY_MEASURED',
    sustained_reference_TFLOPS=dict(min=min(reference),median=statistics.median(reference),max=max(reference)) if reference else None,
    minimum_node_rates=low_nodes[:5],paired_reuse_over_allocate_median={k:statistics.median(v) for k,v in sorted(paired.items())},
    original_dispatch_terminal=read(R/'dispatch-terminal.json'))
result['receipt_pins_sorted_SHA256']=hashlib.sha256(json.dumps(receipt_pins,sort_keys=True).encode()).hexdigest()
result['receipt_pins_count']=len(receipt_pins)
print(json.dumps(result,allow_nan=False))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', choices=('live-full.json','terminal-original.json'))
    arg = parser.parse_args()
    if hashlib.sha256(HELPER.read_bytes()).hexdigest() != HELPER_SHA:
        raise ValueError('FROZEN_CONNECTION_HELPER')
    ssh = runpy.run_path(str(HELPER))['ssh']
    preamble = f'{LABEL=};{REQUEST=};{STEPS=}\n'
    result = subprocess.run([ssh[0],'-o','StrictHostKeyChecking=yes',*ssh[1:],
                             '/usr/local/python3.12/bin/python3 -I -S -B -'],
                            cwd=REPO,input=(preamble+REMOTE).encode(),capture_output=True,timeout=48)
    if result.returncode:
        print(json.dumps(dict(returncode=result.returncode,stderr_bytes=len(result.stderr),
                              stderr_SHA256=hashlib.sha256(result.stderr).hexdigest())))
        return 1
    if len(result.stdout)>32768:
        raise ValueError('BOUNDED_SUMMARY')
    data=json.loads(result.stdout)
    if arg.output=='terminal-original.json' and data['original_dispatch_terminal'] is None:
        print(json.dumps({'terminal':None,'actual_steps':data['actual_steps']}))
        return 2
    with (REPO/'local-only'/LABEL/arg.output).open('xb') as out:
        out.write((json.dumps(data,indent=2,allow_nan=False)+'\n').encode())
    print(json.dumps(data))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
