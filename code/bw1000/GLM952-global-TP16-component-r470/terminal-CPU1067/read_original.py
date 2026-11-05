"""Independent oneHOST diagnostic build metadata reader; no producer import."""
from pathlib import Path
import argparse,base64,gzip,hashlib,json,re,shlex
H=lambda b:hashlib.sha256(b).hexdigest()
def audit(p):
 raw=Path(p).read_bytes();assert H(raw)=='9b19af735c08d8f09f5de9b51c73acd6bc5e127b997a287e883c38a1dd8bc000';a=json.loads(gzip.decompress(raw));f={}
 for x in a['files']:
  b=gzip.decompress(base64.b64decode(x['gzip_base64'],validate=True));assert len(b)==x['bytes']and H(b)==x['SHA256']and x['path']not in f;f[x['path']]=b
 assert len(f)==36 and sum(map(len,f.values()))==451739;R='/public/home/aclsylqidf/hygon-inference-refactor-20261014';D=R+'/incoming/GLM952-global-TP16-component-r470/phase-CPU-v3/';O=R+'/candidates/GLM-global-TP16-phase-r470-CPU-v3/';P=R+'/incoming/GLM952-global-TP16-component-r470/';j=lambda p:json.loads(f[p]);assert H(f[D+'SOURCE_MANIFEST.json'])=='dbabdf2aa811eb5945790657e973c97911cae58730cd4bb07953a4b25bfa0415'
 for n,h in j(D+'SOURCE_MANIFEST.json')['files_SHA256'].items():assert H(f[D+n])==h
 assert H(f[D+'CPU_RESOURCE_CLAIM.json'])=='e275e1cf25e723ab9f56caefbee084c86f1b12af2e6caac734608d3470383d92';reuse=j(D+'REUSE.json');assert H(f[P+'BUILD_PIN.json'])==reuse['parent_build_pin_SHA256']and H(f[P+'TOOLCHAIN.json'])==reuse['parent_toolchain_SHA256'];pin=j(P+'BUILD_PIN.json');q=j(O+'CPU-result.json');assert H(f[O+'CPU-result.json'])=='ef43d8951fdd307233b12c21c547c6fe2d9bec7adb97f203f313ddfcaf80f3e7'and q['completed']and q['passed']and q['actual_CPU']=='910407.1067'and q['new_HOST']==q['new_CABI']==1 and q['new_GPU']==0 and q['seconds']<270
 assert all(not q[k]for k in ['GPU','Runtime','Native_dlopen','Common_dlopen','Model','weights','component_executed','old_geometry_suite_repeated','old_typed_suite_repeated'])and q['fresh_M_equals_MD'];assert [x['name']for x in q['commands']]==['phase-M','phase-compile','phase-link','phase-ELF','parent-ELF','phase-CABI','phase-loader-metadata']
 deps=lambda b:set(shlex.split(b.decode().replace(chr(92)+chr(10),' ').partition(':')[2]));assert deps(f[O+'phase-M.stdout'])==deps(f[O+'component.d'])==set(q['fresh_dependency_SHA256'])
 for path,h in q['fresh_dependency_SHA256'].items():
  if path.startswith('/refactor/incoming/GLM952-global-TP16-component-r470/phase-CPU-v3/'):
   assert H(f[D+Path(path).name])==h
  else:assert pin['dependency_SHA256'][path]==h
 nm=f[O+'phase-CABI.stdout'].decode();symbols={x.split()[-1]for x in nm.splitlines()};assert {'glm_global_tp16_'+n+'_r470'for n in ['error','id_size','id','create','geometry','probe','close','trace_path']}<=symbols
 need=lambda b:re.findall(r'\(NEEDED\).*?\[([^\]]+)\]',b.decode());assert need(f[O+'phase-ELF.stdout'])==need(f[O+'parent-ELF.stdout'])==q['NEEDED'];ldd=f[O+'phase-loader-metadata.stdout']+b'\n'+f[O+'phase-loader-metadata.stderr'];assert not re.search(b'not found|undefined symbol',ldd,re.I);resolved={}
 for line in ldd.decode().splitlines():
  m=re.match(r'\s*(\S+)\s+=>\s+(/\S+)\s+\(',line)
  if m:assert m[1]not in resolved;resolved[m[1]]=m[2]
 assert len(q['verified_libraries'])==5 and all(Path(path).name in resolved for path in q['verified_libraries']);assert q['verified_libraries'][reuse['Common']]==reuse['Common_SHA256'];assert H(f[O+'phase-loader-metadata.stdout'])==q['ldd_stdout_SHA256']and H(f[O+'phase-loader-metadata.stderr'])==q['ldd_stderr_SHA256'];typed=q['typed_trace'];assert typed['passed']and typed['high_ctx']==0x1234567800000042 and typed['recording_leaf']and not typed['Native']and not typed['Runtime']and not typed['GPU'];assert '910407.1067|GLM-global-TP16-phase-r470-CPU-v3|COMPLETED|0:0|8'in a['accounting']['stdout'];lib='/refactor/candidates/GLM-global-TP16-phase-r470-CPU-v3/libGLM_global_TP16_phase_r470.so';assert q['library']==lib and q['library_SHA256']=='73006f481891e82b6ac46c67bb53ca513c8df8f0f9378eec26dbcb3ecaea5e39'and a['artifact_metadata'][R+lib[len('/refactor'):]]['SHA256']==q['library_SHA256']
 return dict(audit_passed=True,actual_CPU='910407.1067',state='COMPLETED0',step_seconds=8,inside_seconds=q['seconds'],archive=dict(path=Path(p).name,SHA256=H(raw),files=36,decoded_bytes=451739),receipt_SHA256=H(f[O+'CPU-result.json']),library=lib,library_SHA256=q['library_SHA256'],source_manifest_SHA256=H(f[D+'SOURCE_MANIFEST.json']),new_HOST=1,new_GPU=0,new_CABI=1,fresh_M_equals_MD=True,dependency_count=len(q['fresh_dependency_SHA256']),retained1061_object_metadata=q['retained1061_object_metadata'],verified_libraries=q['verified_libraries'],ldd_both_outputs_clean=True,new_ffi_high64_recording=True,Runtime=False,SDK_execution=False,Common_dlopen=False,Model=False,GPU=False,boundary='Engineering compile/link+static metadata only. No trace fopen/Native probe/device collective executed; submit-return markers are not GPUcompletion. No new sync or math; original870 failure remains.')
def main():
 p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('--output',required=True);a=p.parse_args();q=audit(a.archive);b=(json.dumps(q,indent=2)+'\n').encode();Path(a.output).write_bytes(b);print(json.dumps(dict(passed=True,actual_CPU=q['actual_CPU'],output_SHA256=H(b))))
if __name__=='__main__':main()
