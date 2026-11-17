"""Only new supervisor pure-stdlib schema/receipt/bootstrap boundaries; no child spawn."""
import ast
import copy
import io
import json
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
import bounded_GPU_supervisor as s
from deadline_contract import schedule


def run():
    passed=[]
    def good(name,fn):
        fn();passed.append(name)
    def bad(name,fn):
        try:fn()
        except (ValueError,KeyError):passed.append(name);return
        raise AssertionError(name+' unexpectedly accepted')
    c=dict(mode='fresh25',arm='domain',pair_id='a'*32)
    row=dict(rank=0,mode='fresh25',arm='domain',pair_id='a'*32,completed_microbatches=100,
             updates=25,completed_step=25,recovery_exact=None,formal17=False,
             performance_qualified=False,effect_qualified=False)
    good('fresh100micro25update',lambda:s.rank_result(row,0,c))
    r=dict(row,mode='resume20to25',completed_microbatches=20,updates=5,recovery_exact=True)
    good('resume20micro5update',lambda:s.rank_result(r,0,dict(c,mode='resume20to25')))
    bad('missing_rank_not_complete',lambda:s.rank_result(dict(row,rank=1),0,c))
    bad('25step_mislabeled_5update',lambda:s.rank_result(dict(row,updates=5),0,c))
    bad('resume_not_exact',lambda:s.rank_result(dict(r,recovery_exact=False),0,dict(c,mode='resume20to25')))
    bad('effect_not_inherited',lambda:s.rank_result(dict(row,effect_qualified=True),0,c))
    bad('duplicate_json',lambda:s.decode(b'{"x":1,"x":2}'))
    bad('nonfinite_json',lambda:s.decode(b'{"x":NaN}'))
    good('real_json_roundtrip',lambda: s.need(s.decode((json.dumps(row)+'\n').encode())==row,'ROUNDTRIP'))
    good('same_origin_budget',lambda:s.need(schedule(1000,1001)==dict(epoch=1000,native_deadline=1820,supervisor_stop=1850,deadline=1900),'BUDGET'))
    bad('expired_original_epoch',lambda:schedule(1000,1900))
    bad('future_original_epoch',lambda:schedule(1000,999))
    # Parse the real bootstrap and its exact assembled replacement; no exec/os calls.
    assembled=s.BOOTSTRAP.replace("os.execv(sys.executable,[sys.executable]+sys.argv[1:])",
        "digest=sys.stdin.buffer.readline(66).strip().decode('ascii'); assert len(digest)==64 and set(digest)<=set('0123456789abcdef'); args=sys.argv[1:]; args[args.index('--usgs-domain-config-sha256')+1]=digest; os.execv(sys.executable,[sys.executable]+args)")
    t=ast.parse(assembled)
    good('bootstrap_actual_newline',lambda:s.need(b'GO\n' in [n.value for n in ast.walk(t) if isinstance(n,ast.Constant)],'BOOTSTRAP_NEWLINE'))
    good('bootstrap_exec_preserves_pid',lambda:s.need(sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='execv' for n in ast.walk(t))==1,'ONE_EXEC'))
    a=SimpleNamespace(request_id='a'*32,epoch=1000,publication='b'*40,scope=PurePosixPath('/new/scope'))
    cfg=dict(request_id=a.request_id,epoch=1000,source_publication=a.publication,mode='fresh25',arm='domain',case='T04.sft',root_GO=True,GPU_GO=True,device='cuda',formal17=False,output='/new/scope/witness',train_output='/new/scope/train',deadline=1820,maximum_seconds=820,world_size=8,max_steps=25,physical_rows=800)
    good('template_exact_origin',lambda:s.template_gate(cfg,a))
    bad('template_reset_deadline',lambda:s.template_gate(dict(cfg,deadline=1900),a))
    bad('template_gpu_false',lambda:s.template_gate(dict(cfg,GPU_GO=False),a))
    return dict(passed=True,checks=passed,count=len(passed),Torch_imports=0,child_spawns=0,remote=0,GPU=0)


def run_entry_delta():
    rows=[dict(stage=x) for x in ['planned','owned-cleanup-contract','fresh-admission','runtime','running','exited']]
    rows[1]['actual']=dict(held_session_identity_verified=True,cleanup_helpers_verified=True,resource_release_permitted=False)
    rows[-1]['returncode']=0
    s.entry_stages(rows);checks=['new6stages']
    candidates=[]
    candidates.append(('old5stages',rows[:1]+rows[2:]))
    for key,value in [('held_session_identity_verified',False),('cleanup_helpers_verified',False),('resource_release_permitted',True),('held_session_identity_verified',1)]:
        other=copy.deepcopy(rows);other[1]['actual'][key]=value;candidates.append((key+repr(value),other))
    other=copy.deepcopy(rows);other[-1]['returncode']=1;candidates.append(('entry_rc1',other))
    for name,other in candidates:
        try:s.entry_stages(other)
        except (ValueError,KeyError):checks.append(name)
        else:raise AssertionError(name+' accepted')
    return dict(passed=True,count=len(checks),checks=checks,Torch_imports=0,child_spawns=0,GPU=0,remote=0,old17_repeated=False)


def run_resource_delta():
    def state():return dict(samples=0,sampled_group_peak_RSS_bytes=0,soft_limit_exceeded=False)
    registry={10:dict(PID=10,startticks=123)}
    r=state();assert s.affinity_contract({2,3})==[2,3]
    identity=dict(PID=10,startticks=123,state='R')
    value=s.checked_owned_rss(registry[10],identity,lambda:b'Name:\tpython\nVmRSS:\t1024 kB\n',lambda:identity)
    assert value==1024*1024
    assert s.sample_group_rss(registry,r,reader=lambda _:value) and r['samples']==1 and r['sampled_group_peak_RSS_bytes']==value
    r=state();assert not s.sample_group_rss(registry,r,reader=lambda _:s.RSS_SOFT_BYTES+1) and r['soft_limit_exceeded']
    calls=[]
    def forbidden():calls.append('status');raise AssertionError('unowned status read')
    value=s.checked_owned_rss(registry[10],dict(PID=10,startticks=124,state='R'),forbidden,forbidden)
    assert value is None and calls==[]
    return dict(passed=True,count=3,checks=['matched_normal_and_affinity','above_256GiB_soft_stop','foreign_startticks_no_status_read'],Torch_imports=0,child_spawns=0,GPU=0,remote=0,old17_or_7_repeated=False)


if __name__=='__main__':
    print(json.dumps(run(),sort_keys=True))
