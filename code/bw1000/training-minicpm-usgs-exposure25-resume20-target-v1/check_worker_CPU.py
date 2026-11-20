"""New CPU-only boundaries with explicit process/ownership doubles, no devices."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import time

import worker as w


class OwnedDouble:
    def __init__(self,fail=False):self.cleanups=0;self.fail=fail
    def snapshot(self):return {777001:dict(PID=777001,PPID=77,PGID=777001,startticks=123,state='R')}
    def refresh(self,registry,root):return {}
    def cleanup(self,registry,root,grace):
        self.cleanups+=1
        if self.fail:raise OSError('fixture cleanup failure')
        return dict(owned_processes_remaining=0,termination_needed=False)


class ChildDouble:
    pid=777001
    def __init__(self,rc):self.returncode=rc;self.waits=0
    def poll(self):return self.returncode
    def wait(self,timeout):self.waits+=1;return self.returncode


def config():
    return dict(schema='MINICPM_USGS_TARGET_WORKER_CONFIG_V1',request=w.REQUEST,run_root=str(w.RUN),
      core_directory=str(w.RUN/'src'),framework_directory=str(w.RUN/'framework'),parent=w.PARENT,node=w.NODE,
      port=w.PORT,root_GO=True,training_GO=True,GPU_GO=True,core_manifest_SHA256=w.CORE_SHA,
      epoch=1000,deadline=3400,native_deadline=2980,boot_id='fixture-only')


def main():
    checks=[]
    def good(name,fn):fn();checks.append(dict(name=name,passed=True))
    def reject(name,fn):
        try:fn()
        except (ValueError,FileExistsError,FileNotFoundError):checks.append(dict(name=name,passed=True));return
        raise AssertionError('failure was not rejected: '+name)
    c=config();good('valid_original2400_1980_context',lambda:w.check_config(c,1200))
    for field,value,name in [('epoch',float('nan'),'NaN'),('epoch',1300,'future'),('deadline',3401,'extended_whole'),('native_deadline',2981,'extended_native'),('training_GO',False,'GO_false')]:
        t=dict(c);t[field]=value;reject(name,lambda t=t:w.check_config(t,1200))
    reject('expired_native',lambda:w.check_config(c,2980))
    pcis=['0000:%02x:00.0'%i for i in range(1,9)]
    before=dict(node=w.NODE,boot_id='b',pci=pcis,kfd_holder_count=0,full_fd_inventory_performed=True,
        holder_inventory_basis='kernel_KFD_plus_sameUID_FD_complete',monotonic_ns=100)
    current=dict(before,full_fd_inventory_performed=False,holder_inventory_basis='kernel_KFD_proc_only',monotonic_ns=1000)
    good('qualified_HOST_and_kernel_only_snapshot',lambda:w.check_snapshots(before,current,64))
    for key,value,name in [('boot_id','changed','boot_changed'),('kfd_holder_count',1,'known_KFD_holder'),('monotonic_ns',45_000_000_101,'stale_snapshot'),('pci',pcis[:-1]+pcis[:1],'duplicate_PCI')]:
        t=dict(current);t[key]=value;reject(name,lambda t=t:w.check_snapshots(before,t,64))
    reject('short_CPU_allocation',lambda:w.check_snapshots(before,current,63))
    plan={'command':['/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-qwen-a68952c-swift-initial-CPU-v1/framework/vlm/runtime/swift_entry.py']}
    good('exact_isolated_entry_normalization',lambda:w.normalized_expected(copy.deepcopy(plan),w.RUN))
    reject('unknown_entry_not_normalized',lambda:w.normalized_expected({'command':['/unknown/entry.py']},w.RUN))
    budget=lambda:w.Budget(1000,3400,2980,wall=lambda:1200,monotonic=lambda:0)
    native_plan={'command':['NO_ACTUAL_TRAINING'], 'environment_overrides':{}}
    with tempfile.TemporaryDirectory(prefix='minicpm-worker-cpu-') as temp:
        root=Path(temp)
        r=root/'nonzero';r.mkdir();owned=OwnedDouble();child=ChildDouble(7)
        result=w.run_arm('continuous',r,native_plan,c,budget(),owned,popen=lambda *a,**k:child,sleep=lambda x:None)
        assert result['passed'] is False and result['rc']==7 and owned.cleanups==child.waits==1
        assert len(result['ranks'])==8 and all(x['planned_micro']==100 and x['completed_micro'] is None for x in result['ranks'])
        checks.append(dict(name='nonzero_child_reaped_cleanup_and8_missing_denominators_retained',passed=True))
        r=root/'cleanupfail';r.mkdir();owned=OwnedDouble(True);child=ChildDouble(7)
        result=w.run_arm('continuous',r,native_plan,c,budget(),owned,popen=lambda *a,**k:child,sleep=lambda x:None)
        assert child.waits==1 and owned.cleanups==1 and result['cleanup_known'] is False and result['cleanup_errors']
        checks.append(dict(name='cleanup_exception_does_not_skip_child_reap_or_outcome',passed=True))
        r=root/'spawnfail';r.mkdir();owned=OwnedDouble()
        def fail(*a,**k):raise OSError('fixture Popen failed')
        result=w.run_arm('continuous',r,native_plan,c,budget(),owned,popen=fail,sleep=lambda x:None)
        assert result['attempted_child'] is False and result['error']['type']=='OSError' and owned.cleanups==1
        checks.append(dict(name='spawn_failure_still_cleanup_and_save_outcome',passed=True))
        r=root/'existing';r.mkdir();(r/'continuous').mkdir()
        reject('existing_arm_never_overwritten',lambda:w.run_arm('continuous',r,native_plan,c,budget(),OwnedDouble(),popen=fail))
        r=root/'checkpoint';p=r/'continuous/train/checkpoint-20';p.mkdir(parents=True)
        for n in w.CHECKPOINT_NAMES:(p/n).write_bytes(b'CPU_FIXTURE_NOT_TENSOR')
        (p/'trainer_state.json').write_text(json.dumps(dict(global_step=20,max_steps=25,epoch=.8)),'utf-8')
        inventory=w.checkpoint_inventory(r,'continuous',20,budget());assert inventory['member_count']==14 and inventory['actual_tensor_deserialization_here'] is False
        checks.append(dict(name='actual14file_inventory_uses_args_json_no_tensor_loader',passed=True))
        (p/'rng_state_7.pth').unlink()
        reject('missing_rank7_RNG_rejected',lambda:w.checkpoint_inventory(r,'continuous',20,budget()))
        p=r/'partial/continuous/train/minicpm-usgs-witness/rank-00';p.mkdir(parents=True)
        (p/'progress.json').write_text(json.dumps(dict(outputs=7,completed_steps=[1])),'utf-8')
        ranks=w.rank_metadata('continuous',r/'partial');assert ranks[0]['completed_micro']==7 and ranks[0]['completed_updates']==1 and not ranks[0]['complete']
        assert all(v['completed_micro'] is None for v in ranks[1:])
        checks.append(dict(name='partial7micro_and1update_not_reset_to_zero',passed=True))
    spec=importlib.util.spec_from_file_location('minicpm_helper_test',Path(w.__file__).with_name('owned_processes.py'));helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    args=[b'python',b'--output_dir',str(w.RUN/'continuous/train').encode(),b'minicpm_usgs_continuous_witness']
    assert helper.native_command(args,w.RUN/'continuous') is True
    assert helper.native_command([*args[:-1],b'qwen_treasury_continuous_witness'],w.RUN/'continuous') is False
    checks.append(dict(name='ownership_match_is_exact_Mini_scope_and_callback',passed=True))
    G=Path(__file__).resolve().parents[3]
    out=G/'local-only/training-minicpm-usgs-exposure25-resume20-target-v1-CPU-checks'
    out.mkdir(exist_ok=False)
    result=dict(schema='MINICPM_TARGET_WORKER_NEW_CPU_BOUNDARIES_V1',passed=True,checks=checks,
                checks_passed=len(checks),fixture_only=True,process_and_ownership_doubles=True,
                actual_native_children=0,actual_optimizer_updates=0,real_tensor_bodies=0,
                actual_source_SHA256=hashlib.sha256(Path(w.__file__).read_bytes()).hexdigest(),
                HTTP=0,SSH=0,Torch=0,GPU=0,target_qualified=False)
    (out/'result.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n','utf-8')
    print(json.dumps({k:result[k] for k in ['passed','checks_passed','fixture_only','actual_native_children','actual_optimizer_updates','target_qualified']}))


if __name__=='__main__':main()
