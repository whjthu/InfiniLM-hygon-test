"""Changed 12-query consumer only; real Native and device leaves recorded."""
from pathlib import Path
import ast,hashlib,importlib.util,json,sys
def run():
 d=Path(__file__).resolve().parent;g=d.parents[3];parent=g/'code/bw1000/GLM-V-on-demand-prefill-r400'
 for name,old in [('checkpoint_loader.py','checkpoint_loader.py'),('original_consumer_r400.py','consumer.py')]:assert(d/name).read_bytes()==(parent/old).read_bytes()
 for p in d.glob('*.py'):ast.parse(p.read_bytes());compile(p.read_bytes(),str(p),'exec')
 sys.path.insert(0,str(d));import consumer,controller
 result=consumer.recording_contract();assert result['passed']
 cmd=controller.gpu_command('/not-created-CPU-root');assert all(v in cmd for v in ['--jobid=910374','-c32','a02r4n07','--gres=dcu:8','00:10:00'])
 assert consumer.MODES==(0,1,0,1,1,0)and len(consumer.CASES)==2
 for case in consumer.CASES:
  ids=consumer.selections(case);assert len(ids)==len(set(ids))==2048 and min(ids)>=0 and max(ids)==32767
 # Shared real binding/consumer methods, not execution of a fake whole module.
 assert consumer.NativeLeaf.__mro__[1].__module__=='original_consumer_r400'
 shell=(d/'GPU_inside.sh').read_text();assert 'GLM_SELECTED_KVB_PLAN_R426=0'in shell and 'INFINILM_OWNER_CPU_WIDTH_R2392=4'in shell
 source=(d/'GPU_run.py').read_text();assert 'modern_glm_fp8_selected_plan_stats_r426'not in source and 'glm_attention_value_reassociate_r431'in source
 return dict(passed=True,actual_changed_consumer=result,original_checkpoint_loader_bytes=True,original_query_leaf_bytes=True,actual_command_producer=True,actual_host_fullCLI_preflight_pending=True,Native=0,GPU=0,Slurm=0,compile=0,old_suites_rerun=False)
if __name__=='__main__':print(json.dumps(run(),indent=2))
