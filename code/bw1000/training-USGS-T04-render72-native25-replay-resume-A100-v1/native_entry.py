"""New opt-in around frozen404 native Swift planner; no training loop replacement."""
import argparse,json,os,resource,signal,sys,time
from pathlib import Path,PurePosixPath
from common import D,P,need,pinned,obj,save,left,verify_source,native_inputs

def main(a):
    verify_source();c=obj(a.config);need(c['phase'] in ('continuous','replay2','resume') and P['root_GO'] is P['GPU_GO'] is True,'EXPLICIT_NEW_OPTIN')
    left(c);signal.signal(signal.SIGALRM,lambda *_:os._exit(124));signal.setitimer(signal.ITIMER_REAL,left(c))
    need(os.environ['CUDA_VISIBLE_DEVICES']==P['expected_GPU']['UUID'] and sorted(os.sched_getaffinity(0))==P['CPU_affinity'],'FIXED_ONE_GPU_CPU')
    parent=c['supervisor'];f=(Path('/proc')/str(parent['PID'])/'stat').read_text().rsplit(') ',1)[1].split();need(os.getppid()==parent['PID'] and int(f[19])==parent['startticks'] and os.uname().nodename==parent['host'],'HELD_OWNED_PARENT')
    resource.setrlimit(resource.RLIMIT_FSIZE,(P['capture_file_cap_bytes'],P['capture_file_cap_bytes']))
    _,_,data=native_inputs(c);sys.path.insert(0,str(Path(P['framework'])/'tools'));sys.path.insert(0,P['framework'])
    import hygon_vlm_recipe as recipe
    plan=recipe.build_launch('T04.sft',(D/'T04-original-launch.json').read_bytes(),PurePosixPath('/home/whj/nvidia-baseline/env-vlm/bin/swift'),PurePosixPath('/mnt/data/nvidia-baseline-20261010/inputs/models'),PurePosixPath(str(data)),PurePosixPath('/home/whj/nvidia-baseline'),PurePosixPath(c['train_output']),P['master_port'])
    cmd=plan['command']
    def group(flag,values):
        i=cmd.index(flag)+1;j=i
        while j<len(cmd) and not cmd[j].startswith('--'):j+=1
        cmd[i:j]=values
    group('--callbacks',['native_render72_diagnostic']);group('--external_plugins',[P['framework']+'/vlm/plugins/llava_onevision.py',str(D/'native_plugin.py')])
    if c['phase']=='resume':cmd+=['--resume_from_checkpoint',c['checkpoint20'],'--resume_only_model','false','--ignore_data_skip','false','--load_args','false']
    env=dict(os.environ)
    for key in recipe.RANK_ENV:env.pop(key,None)
    env.update(plan['environment_overrides']);env.update(CUDA_VISIBLE_DEVICES=P['expected_GPU']['UUID'],NPROC_PER_NODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',RENDER72_CONFIG=str(a.config),PYTHONPATH=str(D)+':'+P['framework'],PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
    need(cmd[cmd.index('--max_steps')+1]=='25' and cmd[cmd.index('--gradient_checkpointing')+1]=='false' and cmd[cmd.index('--attn_impl')+1]=='eager','ORIGINAL_MATH_FLAGS')
    save(Path(c['phase_root'])/'launch.json',dict(case='T04.sft',phase=c['phase'],command=cmd,environment_overrides={k:v for k,v in env.items() if k in plan['environment_overrides'] or k in ['CUDA_VISIBLE_DEVICES','NPROC_PER_NODE','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONPATH']},original8_reference=False,actual_world_size=1))
    os.execve(cmd[0],cmd,env)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);raise SystemExit(main(p.parse_args()))
