"""Fill only publication identity/source hash in the fixed approved template."""
from pathlib import Path
import argparse,json,re,subprocess,sys
from contract_r410 import publish_once,validate_plan,verify_bundle
from controller import R,D
def main():
 p=argparse.ArgumentParser();p.add_argument('--publication-commit',required=True);p.add_argument('--source-manifest-sha',required=True);a=p.parse_args()
 assert re.fullmatch('[0-9a-f]{40}',a.publication_commit)and a.publication_commit!='0'*40
 m=verify_bundle(D,a.source_manifest_sha);assert m['GPU_GO']is True and m['Model_GO']is True
 plan=json.loads((D/'PLAN.template.json').read_bytes());plan.update(publication_commit=a.publication_commit,controller_source_manifest_SHA256=a.source_manifest_sha,GPU_GO=True,Model_GO=True,execute_once=True);validate_plan(plan,True)
 file=R/'results'/('Kimi-ordinary-r410-'+plan['run']+'-execution-plan.json');publish_once(file,plan)
 # Existing intent/watch/label checks in invoke_once preclude replay if output is lost.
 p=subprocess.run([sys.executable,'-B',str(D/'invoke_once.py'),'--plan',str(file)],capture_output=True,text=True,timeout=120)
 if p.returncode:raise RuntimeError(dict(rc=p.returncode,stdout=p.stdout[-2000:],stderr=p.stderr[-4000:]))
 q=json.loads(p.stdout);q['execution_plan']=str(file);q['controller_source_manifest_SHA256']=a.source_manifest_sha;print(json.dumps(q))
if __name__=='__main__':main()
