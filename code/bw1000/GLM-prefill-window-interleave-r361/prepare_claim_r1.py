from pathlib import Path
import json,hashlib
D=Path(__file__).resolve().parent;OUT=D/'runtime-r1';repo=D.parents[2];results=repo/'results/bw1000/2026-11-02';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
observed=json.loads((results/'GLM361-CPU-r1-observed.json').read_bytes());entry=next(x for x in observed['receipts']if x['data']['actual_CPU']=='910407.707');q=entry['data']
assert q['completed']and q['passed']and q['GPU']==0 and not q['Native_dlopen']and not q['weights_opened']and not q['Native_constructor_called']
assert len(q['cases'])==8 and all(x['passed']for x in q['cases'])
assert q['owner_manifest_SHA256']==sha(OUT/'owner/SOURCE_MANIFEST.json') and q['controller_manifest_SHA256']==sha(OUT/'controller/SOURCE_MANIFEST.json') and q['CPU_source_SHA256']==sha(OUT/'CPU_whole.py')
raw=json.dumps(q,indent=2).encode()+bytes([10]);assert hashlib.sha256(raw).hexdigest()==entry['SHA256'];(results/'GLM361-CPU707-PASS.json').write_bytes(raw)
g=json.loads((OUT/'execution-GO.template.json').read_bytes());g.update(execute_once=True,publication_commit='a'*40,CPU_receipt_SHA256=entry['SHA256'],CPU_package_manifest_SHA256=sha(OUT/'CPU_PACKAGE_MANIFEST.json'),base_355_lifecycle_consumed=True,interleave_snapshot_manifest_SHA256='11cf0416e344e0fe8795c809c2cf54a87588f9dc6c8093a8d203749068deed8e')
g['quality']='Same complete479 read_file/8900 calculate fixtures and673 encoded IDs; actual original gold/EOS perrow/pair; first qualification/warm policy path failure stops whole residency, no edited-source retry/reload.'
g['measurement']='Actual full admission/first/last, short max/P95/P99 ITL and first-last rate; retain longTTFT/fullpair/max32complete schedule and all32 goodput. Both decode-defer ON; never compare trimmed remaining-decode window as whole gain.'
g['scope']='361 W4-boundary prefill interleave only, fixeddeferON both arms, original038d/C2M1/32768/ordinary4096, two unmodified inputs x6pairs12HTTP, no benchmark/onlineSLO/15 claim.'
g['authorization']='Root explicit361 continuation and Astra158 selection; source+CPU+concrete claim ordinarypush then fresh exactnode/parent idle/dedup permits one residency, no additional GO. First warm/path failure STOP; no source edit/retry/automaticreload.'
(D/'private-GO-r1-prepublish.json').write_bytes((json.dumps(g,indent=2)+'\n').encode())
claim={k:v for k,v in g.items()if k!='publication_commit'};claim.update(state='CPU707_PASS_FINAL_DEPENDENCY_PROBE_PENDING_NO_GPU',source='code/bw1000/GLM-prefill-window-interleave-r361/runtime-r1',actual_CPU='910407.707',CPU_receipt_path='results/bw1000/2026-11-02/GLM361-CPU707-PASS.json',GPU_submitted=False,source_snapshot_unchanged=True,algorithm_helper_SHA256='54861403adad4a6143eb260a5735fb0fdfbaf9e5202457202dc4a4a7054bb932',actual_Native_or_Model_numeric_qualified_by_CPU=False)
(results/'GLM361-r1-source-CPU-resource-claim.json').write_bytes((json.dumps(claim,indent=2)+'\n').encode())
remote='''from pathlib import Path
import json,hashlib,sys
R=Path.home()/'hygon-inference-refactor-20261014';D=R/'incoming/GLM-prefill-window-interleave-r361/runtime-r1';g=GO
sys.path.insert(0,str(D/'controller'));import controller
controller.verify(R,D/'controller',g)
m=json.loads((D/'CPU_PACKAGE_MANIFEST.json').read_bytes());assert hashlib.sha256((D/'CPU_PACKAGE_MANIFEST.json').read_bytes()).hexdigest()==g['CPU_package_manifest_SHA256']
assert all(hashlib.sha256((D/n).read_bytes()).hexdigest()==h for n,h in m['files_SHA256'].items())
q=json.loads((R/'candidates/GLM-prefill-window-interleave-r361/runtime-r1/CPU-result.json').read_bytes());assert q['CPU_source_SHA256']==hashlib.sha256((D/'CPU_whole.py').read_bytes()).hexdigest()
print(json.dumps({'source_and_355_ancestry_verified':True,'CPU707_receipt_SHA256':g['CPU_receipt_SHA256'],'owner_manifest_SHA256':g['owner_manifest_SHA256'],'controller_manifest_SHA256':g['controller_manifest_SHA256'],'CPU_package_manifest_SHA256':g['CPU_package_manifest_SHA256'],'namespace_verified':True,'full_package_members':len(m['files_SHA256']),'GPU_submitted':False}))
'''.replace('g=GO','g='+repr(g));compile(remote,'remote-dependency-probe-r1','exec');(D/'remote_dependency_probe_r1.py').write_bytes(remote.encode())
print(json.dumps({'ready_for_readonly_final_probe':True,'GPU_submitted':False,'CPU_receipt_SHA256':entry['SHA256']}))
