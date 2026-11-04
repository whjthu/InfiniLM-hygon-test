"""Only newly composed whole_cli consumer, real tempfile paths/recorded child."""
from pathlib import Path
import argparse,base64,copy,hashlib,importlib.util,json,subprocess,sys,tempfile
from unittest.mock import patch
D=Path(__file__).resolve().parent
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--original-seal',type=Path,required=True);p.add_argument('--publication',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 assert H(a.original_seal)=='b5b0907f45dec1b2736c907e9a12c099dbff10025df5b127ef5c95ea203f42bb'
 sealed=json.loads(a.original_seal.read_bytes());data={}
 for path,v in sealed['files'].items():
  raw=base64.b64decode(v['base64']);assert len(raw)==v['bytes']and hashlib.sha256(raw).hexdigest()==v['SHA256'];data[path]=raw
 original=json.loads(data['results/Deep451-HOST-instrumented-metadata-v3/result.json']);assert original['returncode']==0 and original['actual_schema_valid']and original['actual_original_CLI_returned']and original['child_wall_elapsed_s']<90
 spec=importlib.util.spec_from_file_location('Deep451_actual_new_preflight',D/'preflight.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 mf=H(D/'SOURCE_MANIFEST.json');m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());cases=[]
 for foreign in [False,True]:
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);folder=root/'results/Deep451-deployment-v4-metadata'/('node-'+original['HOST_node']+'-pid'+str(original['HOST_PID']))
   if foreign:folder=root/'results/foreign-path'/folder.name
   folder.mkdir(parents=True);q=copy.deepcopy(original);q.update(source_manifest_SHA256=mf,publication=a.publication)
   for name in ['stdout','stderr']:
    target=folder/name;target.write_bytes(data['results/Deep451-HOST-instrumented-metadata-v3/'+name]);q[name]['path']=str(target)
   target=folder/'result.json';target.write_text(json.dumps(q));seen=[]
   def recording_run(command,**kwargs):
    assert command==[sys.executable,'-B',str(D/'metadata_probe.py'),'--source-sha',mf,'--publication',a.publication]
    assert kwargs==dict(capture_output=True,text=True,timeout=130);seen.append(command)
    return subprocess.CompletedProcess(command,0,json.dumps(dict(output=str(target))),'' )
   rejected=False
   with patch.object(module,'R',root),patch.object(module.subprocess,'run',recording_run):
    try:proof=module.whole_cli(D,m,a.publication)
    except AssertionError:rejected=True
   assert len(seen)==1 and rejected==foreign
   if not foreign:assert proof['actual_whole_CLI']==dict(source_and_schema_valid=True,Model_READY=False,Native_dlopen=False,GPU=False)and proof['metadata_PID']==original['HOST_PID']and proof['stdout']['bytes']==94
   cases.append(dict(name='foreign_result_path_rejected'if foreign else'actual_whole_cli_complete_file_handoff',passed=True,subprocess_recording_calls=1))
 q=dict(passed=True,source_manifest_SHA256=mf,publication=a.publication,actual_preflight_source_SHA256=H(D/'preflight.py'),actual_consumer_source_SHA256=H(Path(__file__)),actual_production_whole_cli_called=True,actual_tempfile_handoff=True,cases=cases,original_probe_seal_SHA256=H(a.original_seal),original_probe_receipt_SHA256=sealed['files']['results/Deep451-HOST-instrumented-metadata-v3/result.json']['SHA256'],recorded_fields_substituted=['manifest_SHA256','publication','result/stdout/stderr temp paths'],subprocess_and_HOST_root_recording_leaves_only=True,Native_dlopen=False,GPU=False,Model_READY=False,old946_or_probe_or_actor_suites_repeated=False)
 with a.output.open('x',encoding='utf8')as f:json.dump(q,f,indent=2);f.write('\n')
 print(json.dumps(q))
if __name__=='__main__':main()
