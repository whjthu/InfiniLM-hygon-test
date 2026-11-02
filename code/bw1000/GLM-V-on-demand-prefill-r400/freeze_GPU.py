from pathlib import Path
import hashlib,json,sys,ast,tempfile,subprocess
from read_CPU import read
import launch_once_gate_v2 as launch
D=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(observed):
 cpu=read(observed);q=json.loads(Path(observed).read_bytes())['receipt']['data'];overlays=['GPU_run_gate_v2.py','GPU_inside_gate_v2.sh','controller_gate_v2.py','resource_probe_gate_v2.py','launch_once_gate_v2.py']
 for n in overlays:assert(D/n).read_bytes()==(D/'gate-only-v2'/n).read_bytes()
 for p in D.glob('*.py'):ast.parse(p.read_text());compile(p.read_bytes(),str(p),'exec')
 calls=[]
 def recorded(args,**kw):
  calls.append(args);p=subprocess.Popen([sys.executable,'-c','pass'],stdin=subprocess.DEVNULL,stdout=kw['stdout'],stderr=kw['stderr'],start_new_session=True);p.wait(timeout=10);return p
 with tempfile.TemporaryDirectory(prefix='r400-launch-')as temp:
  root=Path(temp);(root/'logs').mkdir();p,args=launch.start_keeper(root,root/'go.json',popen=recorded);assert p.returncode==0 and len(calls)==1 and args==launch.keeper_command(root,root/'go.json')and'00:15:00'in args
  assert 'controller_gate_v2.py'in str(args)
  try:launch.start_keeper(root,root/'go.json',popen=recorded)
  except FileExistsError:pass
  else:raise AssertionError('duplicatekeeper')
 result=dict(passed=True,actual_CPU802_reader=cpu,actual_keeper_Popen=True,no_Slurm=True,GPU=0,Native=False,duplicatekeeper_rejected=True)
 (D/'GPU-launch-local-CPU-result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
 files=list(json.loads((D/'CPU_SOURCE_MANIFEST.json').read_bytes())['files_SHA256'])+['CPU_SOURCE_MANIFEST.json']+overlays+['read_CPU.py','freeze_GPU.py','GPU-launch-local-CPU-result.json']
 m=dict(schema=1,files_SHA256={n:sha(D/n)for n in files},CPU_receipt='/refactor/candidates/GLM-V-on-demand-prefill-r400-gate-v2-CPU/CPU-build.json',CPU_receipt_SHA256=cpu['receipt_SHA256'],library=q['library'],library_SHA256=cpu['library_SHA256'],CPU_source_manifest_SHA256=cpu['source_manifest_SHA256'],original_CPU800_source_manifest_SHA256='fe58b9c3745fd1ee1a46ecbd7f2873fa13b1d4ff90fe7c7e44fe527742e90c47',namespace_wrapper_SHA256='b6cd3ac7aa1829a39434d8dc186deed6c993323d1ac894727072701570c0825d',GPU_GO=True,prepared=False,final_root_read_pending=False,Model_GO=False,resource_claim=dict(parent=910371,node='a02r3n03',cards=8,TP=8,cores=32,mem_MiB=64000,GPU_total_s=600,inner_gate_s=540,shell_s=570,keeper_parent=910407,keeper_node='a14r2n20',keeper_cores=2,keeper_mem_MiB=2048,keeper_s=900,attempts=1),request_scope=dict(wholeModel_requests=0,cases=['prefill512-union32768'],query_calls_perowner=6,warm_perowner=2,formal_perowner=4,seed_calls_perowner=384,untimed_V_perowner=1,checkpoint_layer=38,checkpoint_loads_perowner=1,profile=0),selectedV_counts=dict(type='HOST completed originalexpand_cached returns, notGPUkernel/events',OFF=0,ON=4,diagnostic=4,max_block_rows=8192),numeric_policy=dict(alert_NRMSE=.005,diagnostic_stop_NRMSE=.02,bitwise_required=False,global_Model_quality_accepted=False,primary_persisted_before_diagnostic=True),activation_basis='Root preauthorized fixed8192remainingprefill source; actualCPU800chunk+consumer passed; exact2privateW gate-only802 qualified sameunchangedlibrary. No additionalwholeCPU/provider/Model.',execution_entry='launch_once_gate_v2.py',controller_entry='controller_gate_v2.py',GPU_entry='GPU_run_gate_v2.py',cardcost_authority='ActualSlurmGPUelapsed*8; owner_service_residency_s contains originalepoch preflight/queue and is not exactallocatedcardseconds',memory_peak='UNKNOWN; snapshots/logicalbounds separately, no loadedfit',scope_limits=['No decodeU2048 rerun','No old729replay','No guard/provider/kernel/precision change','Oldcaller.o uncompiled, onlynewconsumercomponent qualification','One newlabel/residency only, no timeout extension','Original8GiBloadedadmission unchanged, Model_GO=false'])
 p=D/'GPU_SOURCE_MANIFEST.json';p.write_text(json.dumps(m,indent=2)+'\n',encoding='utf8');print(json.dumps(dict(SHA256=sha(p),files=len(files),GPU_GO=True,Model_GO=False)))
if __name__=='__main__':main(sys.argv[1])
