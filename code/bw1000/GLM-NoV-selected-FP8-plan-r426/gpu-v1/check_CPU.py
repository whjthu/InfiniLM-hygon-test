"""Only newsequence/sourcebindings; originalloader andoldmath CPU not repeated."""
from pathlib import Path
import ast,hashlib,importlib.util,json,subprocess,sys
def run():
 D=Path(__file__).resolve().parent;G=D.parents[3];P=G/'code/bw1000/GLM-V-on-demand-prefill-r400';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 assert (D/'checkpoint_loader.py').read_bytes()==(P/'checkpoint_loader.py').read_bytes()
 assert (D/'original_consumer_r400.py').read_bytes()==(P/'consumer.py').read_bytes()
 for p in D.glob('*.py'):ast.parse(p.read_bytes());compile(p.read_bytes(),str(p),'exec')
 sys.path.insert(0,str(D));sp=importlib.util.spec_from_file_location('actual426consumer',D/'consumer.py');C=importlib.util.module_from_spec(sp);sp.loader.exec_module(C);proof=C.recording_contract();assert proof['passed']
 sp=importlib.util.spec_from_file_location('actual426controller',D/'controller.py');T=importlib.util.module_from_spec(sp);sp.loader.exec_module(T)
 cmd=T.gpu_command('/not-created-CPU-root');assert'-c32'in cmd and'--gres=dcu:8'in cmd and'00:10:00'in cmd and'a02r3n03'in cmd
 shell=(D/'GPU_inside.sh').read_text();assert'GLM_SELECTED_KVB_PLAN_R426=0'in shell and'GLM_V_ON_DEMAND_R403=1'in shell and'570'in shell
 source=(D/'controller.py').read_text();assert"for sig,seconds"not in source and"close_owned(LABEL,p,actual"in source and"signal_owned('KILL')"in source and"terminal_unknown_or_unclosed"in source
 launch=(D/'launch_once.py').read_text();assert'sys.path.insert(0,str(Path(__file__).resolve().parent))'in launch and"if preflight_only:return dict"in launch
 return dict(passed=True,actual_new_consumer=proof,checkpoint_loader_original_bytes=True,source_math_NativeLeaf_original_bytes=True,original_numeric_alert=.005,original_numeric_diagnostic_stop=.02,local_controller_command_producer=True,cleanup_function_reused_from_CPU_checked418_except_parentID=True,actual_host_fullCLI_preflight_pending=True,Native=0,GPU=0,Slurm=0,SDK=False)
if __name__=='__main__':print(json.dumps(run(),indent=2))
