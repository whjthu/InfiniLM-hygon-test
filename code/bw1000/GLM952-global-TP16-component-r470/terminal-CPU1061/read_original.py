"""Offline74 original-file audit; never load component/SDK or execute build source."""
from pathlib import Path,PurePosixPath
import argparse,base64,gzip,hashlib,json,re,shlex
SEAL='95eea83a951f9a9df3650dcd7af382ed451d001c4c90963adb2ffd03d70d3cd7';ROOT='/public/home/aclsylqidf/hygon-inference-refactor-20261014';H=lambda b:hashlib.sha256(b).hexdigest()
def audit(path):
 raw=Path(path).read_bytes();assert H(raw)==SEAL;s=json.loads(gzip.decompress(raw));fs={}
 for f in s['files']:
  b=gzip.decompress(base64.b64decode(f['gzip_base64'],validate=True));assert len(b)==f['bytes']and H(b)==f['SHA256']and f['path']not in fs;fs[f['path']]=b
 assert len(fs)==74
 def one(tail):
  a=[(p,b)for p,b in fs.items()if p.endswith(tail)];assert len(a)==1,(tail,len(a));return a[0]
 def read(tail):return json.loads(one(tail)[1])
 pins={}
 for name,pin in [('GLM952-global-TP16-component-r470','9a16e02586232e1f3f9359d5d6994d679eb34459daf289ecad9e59bbb24c71d6'),('HOSTlaunch-v2','be24e20b43d9c71761d8983739668a15c3971d1029ed13905401af6ca90fba0e'),('CPU-link-v2','acbf2824330b5b4cd6f43597ca333840a14fe8b564ca88c99a8dace386a15246')]:
  p,b=one('/'+name+'/SOURCE_MANIFEST.json');assert H(b)==pin;pins[p]=pin
  for n,h in json.loads(b)['files_SHA256'].items():assert H(fs[str(PurePosixPath(p).parent/n)])==h
 old=read('/GLM-global-TP16-component-r470-CPU-v1/CPU-result.json');new=read('/GLM-global-TP16-component-r470-CPU-link-v2/CPU-result.json');assert H(one('/GLM-global-TP16-component-r470-CPU-v1/CPU-result.json')[1])=='e0ee9a89d2214b9501400ddd7fcec163bc68ff518ecd9623761214504aaa8064';assert H(one('/GLM-global-TP16-component-r470-CPU-link-v2/CPU-result.json')[1])=='1a14da10bee15f08575ee8db1bb0ab9772092fca323f93d757cf640c1daad3b5';assert old['completed']and not old['passed']and old['actual_CPU']=='910407.1060'and 'component-link'in old['error'];assert new['completed']and new['passed']and new['actual_CPU']=='910407.1061'and new['new_HOST']==new['new_GPU']==0 and new['reused_completed_HOST']==2
 for q in [old,new]:assert all(q[x]is False for x in ['GPU','Runtime','Native_dlopen','Model','weights'])
 assert not new['Common_dlopen']and not new['component_executed']and not new['geometry_SDK_linked']and new['fresh_M_MD_revalidated']and new['original1060_FAIL_preserved']
 assert [x['name']for x in old['commands']]==['component-M','component-compile','check_geometry-M','check_geometry-compile','component-link'];assert not any('-c'in x['argv']or'-M'in x['argv']for x in new['commands']);assert len([x for x in new['commands']if x['name'].endswith('-link')])==2
 build=read('/GLM952-global-TP16-component-r470/BUILD_PIN.json');reuse=read('/CPU-link-v2/REUSE.json');art={p.replace(ROOT,'/refactor',1):v for p,v in s['artifact_metadata'].items()}
 for p,h in reuse['objects'].items():
  if p.endswith('.o'):assert art[p]['SHA256']==h and art[p]['bytes']>0
  else:assert H(one('/GLM-global-TP16-component-r470-CPU-v1/'+PurePosixPath(p).name)[1])==h
  assert new['reused_objects_SHA256'][p]==h
 package='/refactor/incoming/GLM952-global-TP16-component-r470';parentmf=read('/GLM952-global-TP16-component-r470/SOURCE_MANIFEST.json')['files_SHA256']
 for stem,deps in old['fresh_dependency_SHA256'].items():
  md=one('/GLM-global-TP16-component-r470-CPU-v1/'+stem+'.d')[1].decode();assert set(shlex.split(md.replace(chr(92)+chr(10),' ').partition(':')[2]))==set(deps)
  for p,h in deps.items():
   if str(PurePosixPath(p).parent)==package:assert parentmf[PurePosixPath(p).name]==h
   else:assert build['dependency_SHA256'].get(p)==h,(p,h)
 expected=dict(build['libraries']);expected[reuse['Common']]=reuse['Common_SHA256'];assert expected==new['verified_libraries']and reuse['Common_SHA256']=='16910b4d649f81ba3b63eea9e5bc7c71be5e698c06e00cc669b4ad4d0ec88b1a';prefix='/GLM-global-TP16-component-r470-CPU-link-v2/'
 out=one(prefix+'component-loader-metadata.stdout')[1];err=one(prefix+'component-loader-metadata.stderr')[1];assert H(out)==new['loader_stdout_SHA256']and H(err)==new['loader_stderr_SHA256'];text=(out+err).decode();assert not re.search('not found|undefined symbol',text,re.I);resolved=[]
 for line in text.splitlines():
  m=re.match(r'\s*(\S+)\s+=>\s+(/\S+)\s+\(',line)
  if m:resolved.append((m[1],m[2]))
 for p in expected:
  a=[v for n,v in resolved if n==PurePosixPath(p).name];assert len(a)==1 and a[0]==new['actual_loader_resolution'][PurePosixPath(p).name]
 common=one(prefix+'Common-ELF.stdout')[1].decode();assert not re.search(r'\(SONAME\)',common);assert re.findall(r'\(NEEDED\).*?\[([^\]]+)\]',common)==reuse['Common_ELF']['NEEDED']and re.findall(r'\(RUNPATH\).*?\[([^\]]+)\]',common)==reuse['Common_ELF']['RUNPATH']
 nm=one(prefix+'component-CABI.stdout')[1].decode();exports={'glm_global_tp16_'+n+'_r470'for n in ['error','id_size','id','create','geometry','probe','close']};assert exports<={x.split()[-1]for x in nm.splitlines()}
 dyn=one(prefix+'geometry-ELF.stdout')[1].decode();assert not any(n in dyn for n in ['infinilm','infiniops','infinirt','infiniccl','cuda','hip','torch','python']);geom=json.loads(one(prefix+'geometry-contract.stdout')[1]);assert geom==new['geometry']and geom['passed']and geom['ranks']==16 and geom['full_indexer_layers']==21 and not geom['GPU'];typed=new['typed'];assert typed['passed']and typed['actual_high64_ctx']==0x1234567800000042 and typed['negative_cases']==4 and typed['Native_leaf']=='recording'and not any(typed[x]for x in ['Native_dlopen','Runtime','GPU'])
 actual_library='/refactor/candidates/GLM-global-TP16-component-r470-CPU-link-v2/libGLM_global_TP16_component_r470.so';assert art[actual_library]['SHA256']==new['library_SHA256']=='a2ecf2ccfe787cdb814368d1e556eae2658829277a993cca7d5d2c8e89cc8983';proof=read('/HOSTlaunch-v2/HOST_compile_only_original.json');assert proof['actual_compile_only']and not proof['validator_or_Native_executed']and proof['source_manifest_SHA256']=='be24e20b43d9c71761d8983739668a15c3971d1029ed13905401af6ca90fba0e' and proof['source_SHA256']==read('/HOSTlaunch-v2/SOURCE_MANIFEST.json')['files_SHA256']['launch.py']
 ac=s['accounting'];assert ac['rc']==0 and len(ac['rows'])==2
 for step,state,rc in [('910407.1060','FAILED','1:0'),('910407.1061','COMPLETED','0:0')]:assert any(v.split('|')[0]==step and v.split('|')[2:4]==[state,rc]for v in ac['rows'])
 return dict(schema='GLM470-CPU-original-audit-v1',passed=True,scope='Engineering build/link/config/typed recording only. No GPU/Model/constructor/physicalfit qualification.',archive=dict(path=str(path),SHA256=SEAL,files=len(fs),bytes=len(raw),decoded_bytes=sum(map(len,fs.values()))),source_manifests=pins,original_CPU=dict(step=old['actual_CPU'],passed=False,compile_commands=2,reported_final_new_HOST=old['new_HOST'],error=old['error'],receipt_SHA256=H(one('/GLM-global-TP16-component-r470-CPU-v1/CPU-result.json')[1])),actual_CPU=new['actual_CPU'],CPU_receipt_SHA256=H(one('/GLM-global-TP16-component-r470-CPU-link-v2/CPU-result.json')[1]),seconds=new['seconds'],new_compile=0,reused_completed_HOST=2,reused_objects_SHA256=reuse['objects'],library=actual_library,library_SHA256=new['library_SHA256'],actual_loader_resolution=new['actual_loader_resolution'],verified_library_SHA256=expected,Common_ELF=new['Common_ELF'],loader_stdout_SHA256=H(out),loader_stderr_SHA256=H(err),new_CABI=sorted(exports),geometry=geom,typed=typed,HOST_compile_only_actual=proof,accounting=ac['rows'],GPU=False,Runtime=False,Native_dlopen=False,Common_dlopen=False,Model_READY=False,physical_fit='UNKNOWN',all16_device_collectives='NOT_RUN',limitation='Archive holds actualproducer binaryhash/ELF/MD/stdio andsource, not binarypayload/headerpayload. No offline binarydisassembly, Nativeconstructor/TP16device/memory claim. Future actualdevice must independentlySHA/maps-check exactlibrary/provider group.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('--output',required=True);a=p.parse_args();q=audit(a.archive);Path(a.output).write_text(json.dumps(q,indent=2)+'\n',encoding='utf8');print(json.dumps(dict(passed=q['passed'],files=q['archive']['files'],CPU=q['actual_CPU'],original_CPU_passed=False,library_SHA256=q['library_SHA256'],output_SHA256=H(Path(a.output).read_bytes()))))
