"""CPU-only one actual MoE TU replacement; no dlopen or GPU initialization."""
import pathlib,json,hashlib,subprocess,shlex,os,datetime,shutil
r=pathlib.Path('/refactor');b=r/'candidates/Kimi-MLP-both-r8657';b.mkdir(exist_ok=False);(b/'src').mkdir()
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
out=dict(completed=False,CPU_only=True,GPU=False,dlopen=False,commands=[],compiled_units=[])
def run(cmd,label):
 out['commands'].append(cmd)
 with(b/(label+'.log')).open('x')as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=180)
 assert p.returncode==0,(label,(b/(label+'.log')).read_text()[-6000:])
try:
 oldpath=r/'results/Kimi-CCL512-CPU-r8606.json';assert sha(oldpath)=='45e63209189cadc05ed44dd85102f4c1ef505c46f9a5f23ecf099c39515bef52';old=json.loads(oldpath.read_text())
 original=json.loads((r/'results/Kimi-N4-CPU-r8174.json').read_text());assert old['completed']and original['completed'];assert sha(old['library'])==old['library_SHA256']=='f6f10c2b243e0336074b872859e6ca6acb08effdcd56d86b9ac4df1bf2ddb5e9'
 target='/refactor/candidates/kimi-N4-tile-CPU-r8174/moe.o';assert old['all_linked_object_SHA256'][target]=='33ad6fc01b70a4e66b8061deea330bc52266911929b01b986fe94c19d20aa9ae'
 for p,h in old['all_linked_object_SHA256'].items():assert sha(p)==h,p
 for p,h in old['actual_dynamic_dependency_SHA256'].items():assert sha(p)==h,p
 source=r/'incoming/kimi_k3_moe_MLP_both_r8657.cpp';proof=json.loads((r/'incoming/Kimi-MLP-CPU-source-r8657.json').read_text());assert sha(source)==proof['source_SHA256'][source.name]
 local=b/'src'/source.name;local.write_bytes(source.read_bytes())
 flags=original['commands'][0][:original['commands'][0].index('-MD')]
 assert flags[0]=='g++';flags.insert(6,'-I'+str(b/'src'))
 obj=b/'moe.o';md=b/'moe.d';run([*flags,'-MD','-MF',str(md),'-c',str(local),'-o',str(obj)],'moe')
 deps=sorted(set(shlex.split(md.read_text().replace('\\\n',' ').split(':',1)[1])))
 critical={p:sha(p)for p in deps if pathlib.Path(p).name in ['kimi_k3_moe.hpp','runtime.hpp','tensor.hpp','kimi_n4_tile_r8171.hpp','kimi_spec_math_r3676.hpp','kimi_serving_workspace_r4665.hpp','fused_linear.hpp','base_linear.hpp','linear.hpp']}
 assert critical and all('/incoming/'not in p for p in critical),critical
 out['compiled_units']=[dict(source=str(local),source_SHA256=sha(local),object=str(obj),object_SHA256=sha(obj),deps={p:sha(p)for p in deps},critical_header_SHA256=critical)]
 link=list(old['commands'][-1]);assert link.count(target)==1;link[link.index(target)]=str(obj)
 library=b/'libKimi_MLP_both_r8657.so';link[link.index('-o')+1]=str(library);link=[('-Wl,-Map,'+str(b/'link.map'))if x.startswith('-Wl,-Map,')else x for x in link];run(link,'link')
 defined=subprocess.check_output(['nm','-D','-C','--defined-only',str(library)],text=True);assert 'KimiK3MLP::forward('in defined
 needed=subprocess.check_output(['readelf','-d',str(library)],text=True);assert 'SYMBOLIC'in needed
 # Bind the unchanged thin wrapper to the new Model; never rely on preload.
 oldwrapper=r/'candidates/Kimi-embedding-layer0-CPU-r8635/libKimi_embedding_layer0_r8635.so';assert sha(oldwrapper)=='d0f3dfda93481f24d1c9eada7d596561bdd30246bf5b8fef2ef1fec330ab3af7'
 wrapper=b/'libKimi_embedding_layer0_MLP_r8657.so';oldneeded=subprocess.check_output(['readelf','-d',str(oldwrapper)],text=True)
 if shutil.which('patchelf'):
  shutil.copyfile(oldwrapper,wrapper);run(['patchelf','--replace-needed',old['library'],str(library),str(wrapper)],'wrapper_bind');method='patchelf replace-needed on exact original187 binary'
 else:
  common=r/'candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM';model=r/'candidates/kimi-serving-MoE-5f91ad7-r4665/InfiniLM';sdk=r/'install/hygon-kimi-5f91ad7-r3672';ws=r/'candidates/Kimi-embedding-layer0-CPU-r8635/kimi_embedding_layer0_r8634.cpp';assert sha(ws)=='d1417c44d96b0be273da23e8c911036400cf4079a17e52d8276e05b23a440c13'
  abi=[x for x in flags if x.startswith('-D_GLIBCXX_USE_CXX11_ABI=')];assert len(abi)==1
  cmd=['g++','-std=c++17','-O2','-fPIC','-shared',*abi,'-I/opt/dtk/cuda/cuda/include','-I'+str(model/'csrc'),'-I'+str(common/'csrc'),'-I'+str(sdk/'include'),'-I'+str(common/'third_party/json/include'),'-I'+str(common/'third_party/spdlog/include'),'-MMD','-MF',str(b/'wrapper.d'),str(ws),'-Wl,--no-as-needed',str(library),str(common/'python/_infinilm.cpython-310-x86_64-linux-gnu.so'),'-Wl,--as-needed','-L'+str(common/'python/infinilm/_lib'),'-linfinilm_runtime','-L'+str(sdk/'lib'),'-linfiniops','-linfinirt','-linfiniccl','-Wl,-rpath,'+str(common/'python/infinilm/_lib')+':'+str(sdk/'lib'),'-o',str(wrapper)];run(cmd,'wrapper_unchanged_source');method='original187 identical source/flags only dependency path changed'
  wdeps=sorted(set(shlex.split((b/'wrapper.d').read_text().replace('\\\n',' ').split(':',1)[1])));out['wrapper_fresh_MD']={p:sha(p)for p in wdeps};assert all('/incoming/'not in p for p in wdeps)
 newneeded=subprocess.check_output(['readelf','-d',str(wrapper)],text=True);assert str(library)in newneeded and old['library']not in newneeded
 import re
 pattern=r'\(NEEDED\).*?\[(.*?)\]';before=re.findall(pattern,oldneeded);after=re.findall(pattern,newneeded);assert [str(library)if x==old['library']else x for x in before]==after
 out['wrapper_binding']={'method':method,'old_wrapper':str(oldwrapper),'old_SHA256':sha(oldwrapper),'library':str(wrapper),'library_SHA256':sha(wrapper),'old_needed':oldneeded,'new_needed':newneeded,'source_unchanged':True,'GPU_dlopen':False}
 out.update(completed=True,step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],library=str(library),library_SHA256=sha(library),parent_DSO_SHA256=old['library_SHA256'],parent_raw_receipt_SHA256=sha(oldpath),replaced_object=target,unchanged_object_count=len(old['all_linked_object_SHA256'])-1,all_linked_object_SHA256={p:sha(p)for p in link if p.endswith('.o')},actual_dynamic_dependency_SHA256=old['actual_dynamic_dependency_SHA256'],nm_defined=defined,undefined=subprocess.check_output(['nm','-D','-C','--undefined-only',str(library)],text=True),needed=needed,patchelf_path=shutil.which('patchelf'))
except BaseException as error:out['error']=repr(error);raise
finally:
 p=r/'results/Kimi-MLP-boundary-CPU-r8657.json';assert not p.exists();p.write_text(json.dumps(out,indent=2));print(json.dumps({k:out.get(k)for k in ['completed','step','error','library','library_SHA256','patchelf_path']}))
