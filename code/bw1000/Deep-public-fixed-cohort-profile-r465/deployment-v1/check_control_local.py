"""Actual new command/env and changed-client main; no Slurm/Native execution."""
from pathlib import Path
import hashlib,importlib.util,json,sys,tempfile
D=Path(__file__).resolve().parent;P=D.parent/'source-v1'
sys.path[:0]=[str(D),str(P),str(D.parents[1]/'public-Deep-native-provider-r427/source-v3')]
import controller as C,client
assert Path(client.__file__).resolve()==(D/'client.py').resolve()
def main():
 m=dict(config_SHA256=hashlib.sha256((D/'config.json').read_bytes()).hexdigest(),provider_source_SHA256='cb0e092859a2804611a20673910b9289a9bf64e16e3d93abfe82d3a3d56e03ec',files_SHA256={'client.py':hashlib.sha256((D/'client.py').read_bytes()).hexdigest()})
 gpu=C.gpu_command();argv=C.client_command('910374.recording',m,1.);env=C.gpu_environment({},1.,m)
 assert '--jobid=910374'in gpu and '--gres=dcu:8'in gpu and gpu[gpu.index('-w')+1]=='a02r4n05'and gpu[gpu.index('-t')+1]=='04:10:00'
 plan=Path(argv[argv.index('--private-plan')+1]);assert plan.name=='pilot-private.json'and plan.parent.name=='Deep465-public-fixed-cohort-CPU-v1'and float(argv[argv.index('--deadline-unix')+1])==13801.
 assert '--output'in argv and '--fixtures'not in argv and '--output-dir'not in argv
 assert env['INFINILM_OWNER_CPU_WIDTH_R2392']=='4'and env['WAVE_INSIDE_R2323']=='Deep-public-fixed-cohort-profile-r465/deployment-v1/inside.sh'
 spec=importlib.util.spec_from_file_location('cpu465parent',P/'check_local_CPU.py');cpu=importlib.util.module_from_spec(spec);spec.loader.exec_module(cpu);assert Path(cpu.client.__file__).resolve()==(D/'client.py').resolve()
 with tempfile.TemporaryDirectory()as folder:
  root=Path(folder);(root/'generation_config.json').write_text('{"eos_token_id":1}');q=cpu.client_case(root)
 assert q['passed']and q['HTTP']==64 and q['all64_wrong_not_early_stop']and q['normal_join']
 return dict(passed=True,actual_gpu_command_noexecution=True,actual_client_command_noexecution=True,actual_environment_noexecution=True,actual_changed_client_main=q,Native=False,SDK=False,GPU=False,Slurm=False)
if __name__=='__main__':print(json.dumps(main()))
