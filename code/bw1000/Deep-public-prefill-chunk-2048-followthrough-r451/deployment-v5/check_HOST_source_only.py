"""Only new HOST source-check routing; never invokes CLI or a subprocess."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys
from unittest.mock import patch
D=Path(__file__).resolve().parent
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--evidence-root',type=Path,required=True);p.add_argument('--fixture-root',type=Path,required=True);p.add_argument('--provider-source',type=Path,required=True);p.add_argument('--publication',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 sys.path.insert(0,str(D));spec=importlib.util.spec_from_file_location('Deep451_actual_v5_HOST_preflight',D/'preflight.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());calls=dict(CLI=0,subprocess=0)
 def forbidden_cli(*args,**kwargs):calls['CLI']+=1;raise AssertionError('HOST source-only must not execute actualCLI')
 def forbidden_process(*args,**kwargs):calls['subprocess']+=1;raise AssertionError('HOST source-only must not spawn')
 with patch.object(module,'whole_cli',forbidden_cli),patch.object(module.subprocess,'run',forbidden_process):
  proof=module.check(D,a.provider_source,m,fixture_root=a.fixture_root,evidence_root=a.evidence_root,publication=a.publication,run_actual_cli=False)
 assert calls==dict(CLI=0,subprocess=0)and proof['actual_CLI_verified']is False and proof['actual_CLI_pending']is True and proof['Model_READY']is False and proof['Slurm']==0 and proof['requests']==6
 assert proof['actual_CLI']==dict(actual_CLI_verified=False,actual_CLI_pending=True,reason='HOST source-only; assigned CPUkeeper must run actual CLI before GPUepoch/fresh/Popen',Model_READY=False,Native_dlopen=False,GPU=False)
 assert not any(n in sys.modules for n in ['torch','_infinilm','flash_attn'])
 q=dict(passed=True,publication=a.publication,source_manifest_SHA256=H(D/'SOURCE_MANIFEST.json'),actual_preflight_source_SHA256=H(D/'preflight.py'),actual_consumer_source_SHA256=H(Path(__file__)),actual_HOST_check_function_called=True,actual_source_config_fixture_evidence_consumed=True,filesystem_paths_recording_leaf='Actual frozen R-equivalent L evidence/fixture bytes with pinned hashes; no fake evidence module',calls=calls,proof=proof,actual_CLI_verified=False,actual_CLI_pending=True,Native_dlopen=False,Runtime=False,GPU=False,Model_READY=False,Slurm_submitted=False,old_suites_or_probe_repeated=False)
 with a.output.open('x',encoding='utf8')as f:json.dump(q,f,indent=2);f.write('\n')
 print(json.dumps(q))
if __name__=='__main__':main()
