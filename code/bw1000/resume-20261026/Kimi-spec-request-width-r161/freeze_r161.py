from pathlib import Path
import json,hashlib,subprocess,sys
D=Path(__file__).resolve().parent
out=D/'CPU-contract-r161.json'
if not out.exists():
 z=subprocess.run([sys.executable,'-B',str(D/'CPU_contract_r161.py')],text=True,capture_output=True,check=True)
 out.write_bytes((json.dumps(json.loads(z.stdout),indent=2)+'\n').encode())
manifest={'revision':'r161','source_only':True,'default_OFF':True,'GPU_started':False,'decision':'STOP_NO_DEDICATED_GPU_LOAD','code_parent_commit':'c09ee44b51d6cabd80465c9a5cc0864055243787','shared_445_449_cohort_SHA256':'520d58765aef2e325b15da8a404defea19d79fdd269bbbbe9f4f2eaad0648381','shared_445_449_protocol_SHA256':'bd126c491af3668d2b69d796ea2aad6b71d308c5f701915df029685edbc80d38','root_trace_path':'Kimi-acceptance-cycles-root-r9473-a.json','actual449_report_SHA256':'d68e6f737b11ad5f6ac27e3b8304c20e318af74f65bd78801d51858705d6c7db','no_token_identity_gate_added':True,'actualGPU_performance':'UNKNOWN','files_SHA256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in D.iterdir()if p.is_file()and p.name not in ['source-manifest-r161.json','freeze_r161.py']}}
(D/'source-manifest-r161.json').write_bytes((json.dumps(manifest,indent=2)+'\n').encode());print(json.dumps({'manifest_SHA256':hashlib.sha256((D/'source-manifest-r161.json').read_bytes()).hexdigest(),'decision':manifest['decision'],'files':len(manifest['files_SHA256'])}))
