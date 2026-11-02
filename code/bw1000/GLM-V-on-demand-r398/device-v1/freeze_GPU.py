from pathlib import Path
import json,hashlib,sys,ast,tempfile,subprocess
from read_CPU import read
import launch_once
D=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(observed):
 cpu=read(observed);a=json.loads(Path(observed).read_bytes())['receipt']['data']
 for p in D.glob('*.py'):ast.parse(p.read_text());compile(p.read_bytes(),str(p),'exec')
 calls=[]
 def recorded(args,**kw):
  calls.append((args,kw));p=subprocess.Popen([sys.executable,'-c','pass'],stdin=subprocess.DEVNULL,stdout=kw['stdout'],stderr=kw['stderr'],start_new_session=True);p.wait(timeout=10);return p
 with tempfile.TemporaryDirectory(prefix='r398-launch-')as name:
  root=Path(name);(root/'logs').mkdir();p,args=launch_once.start_keeper(root,root/'recorded-go.json',popen=recorded)
  assert p.returncode==0 and len(calls)==1 and args==launch_once.keeper_command(root,root/'recorded-go.json')and'--gres=none'in args and'-c2'in args and'00:15:00'in args
  try:launch_once.start_keeper(root,root/'recorded-go.json',popen=recorded)
  except FileExistsError:pass
  else:raise AssertionError('keeper duplicate accepted')
  assert len(calls)==1
 contract=dict(passed=True,actual_CPU_reader=cpu,actual_keeper_Popen=True,no_Slurm=True,GPU=0,Native=False,duplicate_keeper_rejected=True)
 (D/'GPU-launch-local-CPU-result.json').write_text(json.dumps(contract,indent=2)+'\n',encoding='utf8')
 files=list(json.loads((D/'CPU_SOURCE_MANIFEST.json').read_bytes())['files_SHA256'])+['CPU_SOURCE_MANIFEST.json','read_CPU.py','launch_once.py','freeze_GPU.py','GPU-launch-local-CPU-result.json']
 m=dict(schema=1,files_SHA256={n:sha(D/n)for n in files},CPU_receipt='/refactor/candidates/GLM-V-on-demand-r398-device-v1-CPU/CPU-build.json',CPU_receipt_SHA256=cpu['CPU_receipt_SHA256'],library=a['library'],library_SHA256=cpu['library_SHA256'],CPU_source_manifest_SHA256=cpu['source_manifest_SHA256'],namespace_wrapper_SHA256='b6cd3ac7aa1829a39434d8dc186deed6c993323d1ac894727072701570c0825d',GPU_GO=False,Model_GO=False,prepared=True,final_root_read_pending=True,resource_claim=dict(parent=910371,node='a02r3n03',cards=8,TP=8,cores=32,mem_MiB=64000,GPU_total_s=600,inner_gate_s=540,shell_s=570,keeper_parent=910407,keeper_node='a14r2n20',keeper_cores=2,keeper_mem_MiB=2048,keeper_s=900,attempts=1),request_scope=dict(wholeModel_requests=0,attention_query_calls_perowner=12,warm_perowner=4,formal_perowner=8,seed_calls_perowner=774,untimed_directV_perowner=2,checkpoint_single_layer_loads_perowner=1,profile=0),numeric_policy=dict(alert_NRMSE=.005,diagnostic_stop_NRMSE=.02,bitwise_required=False,Model_quality_accepted=False,save_before_diagnostic_stop=True),memory_peak='UNKNOWN; physical snapshots and logical temporary bounds separately, no loadedfit',invariants=['No indexer/FFN/fullModel','Original32K/C1/TP8/BF16FP8 attention math','One changedconsumerTU;789caller and allotherobjects reused','Original8GiB loadedservice admission unchanged','No replay/timeout extension/weightreload/grid'])
 p=D/'GPU_SOURCE_MANIFEST.json';p.write_text(json.dumps(m,indent=2)+'\n',encoding='utf8');print(json.dumps(dict(GPU_manifest_SHA256=sha(p),files=len(files),GPU_GO=False,Model_GO=False)))
if __name__=='__main__':main(sys.argv[1])
