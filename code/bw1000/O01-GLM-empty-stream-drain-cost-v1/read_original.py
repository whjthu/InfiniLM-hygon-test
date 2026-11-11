
"""Independent stdlib-only audit of original SDK drain receipts; no GPU API imports."""
from pathlib import Path
import hashlib,json,statistics,subprocess,sys
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'results/O01-GLM-empty-stream-drain-cost-v1'
H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
raw=(D/'outcome.json').read_bytes();out=json.loads(raw);task=json.loads(sys.argv[1]);assert out['claim_commit']=='e645d05388d5543406c5362bc843f18a1335e8c3'
assert out['keeper_passed'] and out['error'] is None and len(out['runs'])==3
assert out['deadline']==out['epoch']+360 and out['wall_seconds']<360
for n in task['remote_files']:assert H(D/'src'/n)==task['source_SHA256'][n]
originals={n:dict(bytes=(D/n).stat().st_size,SHA256=H(D/n)) for n in ('outcome.json','intent.json','assigned.json','result.json')}
intent=json.loads((D/'intent.json').read_bytes());assert intent['epoch']==out['epoch'] and intent['deadline']==out['deadline']
actual=subprocess.check_output(['sacct','-n','-P','-j','1087656','--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw'],text=True,timeout=15)
lines=[x.split('|') for x in actual.splitlines() if '|O01-GLM-empty-stream-drain-cost-v1' in x]
assert len(lines)==3 and {x[0] for x in lines}=={'1087656.125','1087656.126','1087656.127'}
assert all(x[2:4]==['COMPLETED','0:0'] for x in lines)
for r in out['runs']:
 assert r['returncode']==0 and not r['forced_cleanup'] and len(r['accounting'])==1
 assert r['accounting'][0] in lines
before=out['before'];after=out['after'];a=out['assigned']
assert before['node']==after['node']==a['node']==task['node']=='f12r3n17'
assert before['boot_id']==after['boot_id'] and before['pci']==after['pci']
assert len(set(before['pci']))==8 and before['kfd_holder_count']==after['kfd_holder_count']==0
assert after['known_owned_worker_alive'] is False
assert after['known_owned_worker_PID']==a['PID'] and after['known_owned_worker_startticks']==a['startticks']
assert a['visible_cards']==1 and a['PCI'] in before['pci'] and a['step']=='1087656.126'
v=json.loads((D/'result.json').read_bytes());assert out['result.json']['SHA256']==H(D/'result.json')
assert v['step']==a['step'] and v['PCI']==a['PCI'] and v['completed'] and v['stream_destroyed']
assert v['provider_SHA256']==task['SDK_provider_SHA256']
w=v['windows'];assert len(w)==v['planned_windows']==v['completed_windows']==150
assert v['measured_windows']==120 and v['SDK_timed_calls']==115200
order=task['freeze']['order'];summaries=[]
for i,multiple in enumerate(order):
 z=w[i*25:(i+1)*25];assert len(z)==25
 for j,x in enumerate(z):
  assert x['arm']==i and x['index']==j and x['warm']==(j<5)
  assert x['syncs_per_unit']==multiple and x['units']==512 and x['actual_calls']==512*multiple
  assert type(x['wall_ns']) is int and x['wall_ns']>0 and type(x['thread_CPU_ns']) is int and x['thread_CPU_ns']>=0
 med=statistics.median(x['wall_ns']/512 for x in z[5:]);assert med==v['arms'][i]['median_wall_ns_per_unit'];summaries.append(med)
assert sum(x['actual_calls'] for x in w)==115200
pairs=[]
for i in range(3):
 idx=2*i;one=idx if order[idx]==1 else idx+1;two=idx if order[idx]==2 else idx+1
 d=summaries[two]-summaries[one];assert d==v['pairs'][i]['extra_empty_call_ns'];pairs.append(d)
assert statistics.median(pairs)==v['median_extra_empty_call_ns']
assert v['memory_before']==v['memory_after'] and v['memory_before']['free']>=task['memory_floor_bytes']
result=dict(independent_passed=True,accounting_columns=['JobIDRaw','JobName','State','ExitCode','ElapsedRaw'],accounting=lines,
 original_files=originals,source_SHA256=task['source_SHA256'],planned_windows=150,completed_windows=150,measured_windows=120,
 SDK_timed_calls=115200,median_extra_empty_call_ns=statistics.median(pairs),pair_extra_ns=pairs,arms=v['arms'],raw_timing_windows=w,
 same_original906_SDK_provider=True,SDK_provider_SHA256=v['provider_SHA256'],node=a['node'],PCI=a['PCI'],visible_cards=1,
 epoch=out['epoch'],deadline=out['deadline'],controller_wall_seconds=out['wall_seconds'],device_window_seconds=v['wall_seconds'],
 stream_destroyed=True,pre_post_same_boot_and8PCI=True,KFD_pre_post0=True,known_owned_worker_gone=True,
 memory_before=v['memory_before'],memory_after=v['memory_after'],Model_GO=False,performance_win_proven=False,
 decision='Stop treating removal of the second already-empty SDK sync as an explanation for the original millisecond projection gap. No Native change promoted; pending work, Runtime wrapper, Tensor retirement and enqueue idle remain unresolved.',
 limits=v['limits'],reader_SHA256=H(Path(__file__)))
p=D/'independent-result.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps(result))
