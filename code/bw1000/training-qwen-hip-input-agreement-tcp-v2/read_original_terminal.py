import hashlib,json,pathlib,subprocess,time
base=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')
label='training-qwen-hip-input-agreement-tcp-v2'
root=base/label
def small(name):
 try:
  with (root/name).open('rb') as f: raw=f.read(131073)
 except FileNotFoundError:return None
 assert len(raw)<=131072
 return dict(sha256=hashlib.sha256(raw).hexdigest(),value=json.loads(raw))
p=subprocess.run(['sacct','-n','-P','-j','1087656','--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList'],capture_output=True,text=True,timeout=9)
rows=[r.split('|') for r in p.stdout.splitlines() if '|'+label+'-' in r]
i=small('intent.json')
t=small('outcome.json')
def marks(suffix):
 return sum((root/'ranks'/f'{r}-{suffix}.json').is_file() for r in range(8))
log=root/'gpu.log'
raw=log.read_bytes() if log.is_file() and log.stat().st_size<1048576 else b''
types={k:raw.count(k.encode()) for k in ('Init COMPLETE','Watchdog caught','Socket Timeout','Traceback','RuntimeError','Timed out','srun: error','RCCL version')}
from collections import Counter
assert t and p.returncode==0
v=t['value']
assert i and v['request_id']==i['value']['request_id']
items=[]
port_set=set()
timings=[]
hashes=[]
for rank in range(8):
 result=small(f'ranks/{rank}.json')
 before=small(f'ranks/{rank}-before-init.json')
 after=small(f'ranks/{rank}-after-init.json')
 start_end=[[small(f'ranks/{rank}-vote-{j}-{kind}.json') for kind in ('start','end')] for j in range(9)]
 vals=[before['value'],after['value'],*[m['value'] for pair in start_end for m in pair]]
 assert all(item['rank']==rank for item in vals)
 assert all(a['monotonic_ns']<=b['monotonic_ns'] for a,b in zip(vals,vals[1:]))
 expected=[dict(rejecting_rank=rejector,local_vote=int(rank!=rejector),agreed=rejector is None)
           for rejector in [None,*range(8)]]
 assert [pair[1]['value']['vote'] for pair in start_end]==expected
 assert all(m['value']['index']==j for j,pair in enumerate(start_end) for m in pair)
 x=result['value']
 assert x['rank']==rank and x['node']=='a01r4n16' and x['step']=='1087656.206'
 assert x['backend']=='nccl' and x['process_group_destroyed'] and x['observations']==expected
 port_set.add(before['value']['master_port'])
 assert before['value']['launcher']=='torchrun' and before['value']['rendezvous']=='env://'
 items.append(x)
 timings.append(dict(rank=rank,init_ns=vals[1]['monotonic_ns']-vals[0]['monotonic_ns'],
   vote_ns=[pair[1]['value']['monotonic_ns']-pair[0]['value']['monotonic_ns'] for pair in start_end]))
 hashes.append(dict(rank=rank,result=result['sha256'],before=before['sha256'],after=after['sha256'],
   vote_marks=[[m['sha256'] for m in pair] for pair in start_end]))
assert len(port_set)==1
snapshots=[small(f'{name}/a01r4n16.json') for name in ('pre','gpu-gate','post')]
sv=[s['value'] for s in snapshots]
lease=small('leased.json')
launch=small('launcher.json')
assert launch['value']['max_restarts']==0 and launch['value']['rendezvous']=='tcp'
assert len({s['boot_id'] for s in sv})==1 and all(s['kfd_holder_count']==0 for s in sv)
assert len(set(x['pci'] for x in items))==len(set(x['kfd'] for x in items))==8
assert all(lease['value'][x['pci']]==x['kfd'] for x in items)
assert all(s['pci']==sorted(lease['value']) for s in sv)
assert [s['step'] for s in sv]==['1087656.205','1087656.206','1087656.207']
assert rows==[v['phases'][k]['terminal']['row'] for k in ('pre','gpu','post')]
assert all(r[2:4]==['COMPLETED','0:0'] for r in rows)
assert hashlib.sha256(raw).hexdigest()==v['phases']['gpu']['log_sha256']
providers={(x['torch'],x['hip'],x['torch_source_sha256'],x['distributed_source_sha256'],tuple(x['mapped_rccl'])) for x in items}
assert len(providers)==1
torch,hip,torch_sha,dist_sha,libs=next(iter(providers))
assert libs
private_reader_summary=dict(independent_component_passed=True,ranks=8,votes=72,before_after_vote_marks=160,
 snapshot_hashes=[s['sha256'] for s in snapshots],lease_sha256=lease['sha256'],launcher_sha256=launch['sha256'],
 torch=torch,hip=hip,torch_sha256=torch_sha,distributed_sha256=dist_sha,rccl_sonames=[pathlib.Path(x).name for x in libs],
 rank_hashes=hashes,diagnostic_intervals_ns=timings,original_collector_visible={
   name:bool(v.get(name)) for name in ('pre','gate','post','assigned','launcher')},
 original_collector_rank_count=sum(x is not None for x in v['rank_results']),
 original_collector_before_count=sum(x['before'] is not None for x in v['rank_marks']),
 original_collector_vote_ends=sum(pair[1] is not None for x in v['rank_marks'] for pair in x['votes']),
 original_collector_component_passed=v['component_passed'])
ranks=sum((root/'ranks'/f'{i}.json').is_file() for i in range(8))
print(json.dumps(dict(observed=time.time(),accounting_rc=p.returncode,rows=rows,namespace=root.is_dir(),
 request_id=i['value']['request_id'] if i else None,
 outcome_sha256=t['sha256'] if t else None,component_passed=t['value']['component_passed'] if t else None,
 rank_receipts=ranks, audit=private_reader_summary,log_bytes=len(raw),log_SHA256=hashlib.sha256(raw).hexdigest(),diagnostic_counts=types,launcher_started=(root/'launcher.json').is_file(),rank_before_init=marks('before-init'),rank_after_init=marks('after-init'),rank_first_vote=marks('vote-0-end'),rank_final_vote=marks('vote-8-end'))))
