"""OneHCunit replacement plus reused741HIP object; compile/link only, noGPU."""
import pathlib,json,hashlib,subprocess,os,time,shlex,re,traceback
R=pathlib.Path('/refactor');S=pathlib.Path(__file__).resolve().parent;B=R/'candidates/Deep-HC-fullpre-Native-r377-cpu-v1';B.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
q=dict(completed=False,passed=False,CPU_only=True,GPU_initialized=False,Native_dlopen=False,Runtime_created=False,weights_payload_opened=False,Model_loaded=False,actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],commands=[],publication_commit=os.environ.get('DEEP377_PUBLICATION_COMMIT','MISSING'))
started=time.perf_counter()
def run(args,label,timeout=180):
 q['commands'].append(args);z=subprocess.run(args,capture_output=True,text=True,timeout=timeout);(B/(label+'.log')).write_text(z.stdout+z.stderr);assert z.returncode==0,(label,z.returncode,z.stderr[-6000:]);return z.stdout
def symbols(path):return {l.split()[-1]for l in run(['nm','-D','--defined-only',str(path)],'exports-'+path.stem).splitlines()if l.split()}
try:
 assert re.fullmatch('[0-9a-f]{40}',q['publication_commit'])
 mp=S/'SOURCE_MANIFEST.json';assert sha(mp)==os.environ['DEEP377_CPU_MANIFEST_SHA'];m=json.loads(mp.read_bytes());assert all(sha(S/n)==h for n,h in m['files_SHA256'].items())
 cp=R/'candidates/Deep-HC-T1-Model-r366-cpu-v1/CPU-build.json';assert sha(cp)=='e3012f3c27c58f3c2c18f237906cd0120a4232620873cbea9b90e5012fd61152';parent=json.loads(cp.read_bytes());oldlib=pathlib.Path(parent['library']);assert parent['passed']and sha(oldlib)=='976f97b51e9f4b11dfb2439d2c97f2a1e7788e4de05ee8a8a0138f33e38b6ed1'
 oldS=R/'incoming/Deep-HC-T1-Model-r366/cpu-v1';original=oldS/'deepseek_v4_hyper_connection_model_r366.cpp';src=S/'deepseek_v4_hyper_connection_native_r377.cpp';proof=json.loads((S/'SOURCE_PROOF.json').read_bytes());assert sha(original)==proof['parent_source_SHA256']and sha(src)==proof['new_source_SHA256'];assert re.sub(r'/\*HC377_BEGIN_([A-Z0-9_]+)\*/.*?/\*HC377_END_\1\*/','',src.read_text(),flags=re.S).encode()==original.read_bytes()
 assert sha(S/'hc_model_contract_r366.hpp')==sha(oldS/'hc_model_contract_r366.hpp')=='5b10cac06200383ea5401a4f7e7c04a284dd24ef07cec0412291d78fcf25e402'
 assert 'residual->is_contiguous()'in src.read_text()
 for p,h in parent['all_objects_SHA256'].items():assert sha(p)==h
 kp=R/'candidates/Deep-HC-full-pre-fusion-r372-cpu-v1/CPU-build.json';assert sha(kp)=='85363c2b0ce044bf105cba85f20d5707d12cbf4022670bfc6f5d63170a9d0f85';kernel_report=json.loads(kp.read_bytes());kernel=kp.with_name('staged.o');assert sha(kernel)==kernel_report['object_SHA256'][str(kernel)]=='d7f1951175b291d9bfd4b7fd8158f741762681dca018770c1f930c2d2fcbbb7c'
 orig_oracle=cp.with_name('CPU-all86-oracle.json');assert sha(orig_oracle)=='d3881d67fc0a4730b873a45c903ce8c4de86c3a75c8e3f07eaa08dc97299cdec';o=json.loads(orig_oracle.read_bytes());assert o['passed']and o['actual_HC_groups']==86 and o['oracle_cases']==172
 gpu_report=R/'results/Deep372-gpu-v1-910374.1298.json';assert sha(gpu_report)=='78f961b0ac59c71dff3f289a265b10e1794def58d267187a93a1453b4668da7d';g=json.loads(gpu_report.read_bytes());assert g['passed']and g['completed']and not any(g['protocol']['errors'])and len(g['protocol']['closes'])==8
 oldobj=pathlib.Path(parent['new_HC_object']);obj=B/'HC-unit3.o';md=B/'HC-unit3.d';base=next(a for a in parent['commands']if '-c'in a and str(original)in a)
 cmd=[('-I'+str(S))if a=='-I'+str(oldS)else str(src)if a==str(original)else str(obj)if a==str(oldobj)else str(md)if a==str(oldobj.with_suffix('.d'))else a for a in base]
 flags=[a for i,a in enumerate(cmd)if i not in [0,cmd.index('-c'),cmd.index('-c')+1,cmd.index('-o'),cmd.index('-o')+1,cmd.index('-MF'),cmd.index('-MF')+1]and a!='-MD']
 parse=lambda t:set(shlex.split(t.replace('\\\n',' ').split(':',1)[1]));fresh=parse(run(['g++',*flags,'-M',str(src)],'fresh-M'));before={p:sha(p)for p in fresh}
 for p,h in before.items():
  if pathlib.Path(p).is_relative_to(S):continue
  assert parent['dependencies_SHA256'].get(p)==h,('Originaldependencychanged',p)
 run(cmd,'HC-unit3-compile');assert parse(md.read_text())==fresh and all(sha(p)==h for p,h in before.items())
 link=[str(obj)if a==str(oldobj)else a for a in parent['link_argv']];lib=B/'libDeep_HC_fullpre_optin_r377.so';link[link.index('-o')+1]=str(lib);link[1:1]=[str(kernel)];assert len([a for a in link if a.endswith('.o')])==50;run(link,'link',120)
 oldsymbols=symbols(oldlib);newsyms=symbols(lib);assert oldsymbols<=newsyms
 required=['deep_hc_fullpre_'+n+'_r377'for n in ['enable','enabled','clear','stats']];assert set(required)<=newsyms
 demangled=run(['nm','-D','-C','--defined-only',str(lib)],'demangled');assert sum(' T 'in l and'HyperConnection::pre('in l for l in demangled.splitlines())==1
 needed=lambda p:[l.split('[',1)[1].split(']',1)[0]for l in run(['readelf','-d',str(p)],'dynamic-'+p.stem).splitlines()if'NEEDED'in l]
 oldneeded=needed(oldlib);newneeded=needed(lib);assert set(oldneeded)==set(newneeded)
 ld=run(['ldd',str(lib)],'ldd');assert 'not found'not in ld
 for p,v in parent['actual_DTK_bindings'].items():assert sha(p)==v['SHA256']
 hip=parent['existing_HIP_provider'];assert sha(hip['path'])==hip['SHA256']
 control=B/'CPU-control-contract';run(['g++','-std=c++17','-O2','-pthread','-I'+str(S),str(S/'CPU_control_contract.cpp'),'-o',str(control)],'CPU-control-compile');run([str(control)],'CPU-control-contract',30)
 unchanged={p:h for p,h in parent['all_objects_SHA256'].items()if p!=str(oldobj)};assert len(unchanged)==48 and all(sha(p)==h for p,h in unchanged.items());objects=dict(unchanged);objects[str(obj)]=sha(obj);objects[str(kernel)]=sha(kernel);assert len(objects)==50
 q.update(completed=True,passed=True,source_manifest_SHA256=sha(mp),source_SHA256=m['files_SHA256'],parent_library_SHA256=sha(oldlib),parent_CPU_receipt_SHA256=sha(cp),library=str(lib),library_SHA256=sha(lib),new_HC_object=str(obj),new_HC_object_SHA256=sha(obj),reused_current48_objects_SHA256=unchanged,all50_objects_SHA256=objects,original47_objects_SHA256=parent['unchanged_objects_SHA256'],new_HIP_reused=True,Kernel_CPU741_SHA256=sha(kp),Kernel_GPU1298_report_SHA256=sha(gpu_report),CPU724_172_oracle_reused_SHA256=sha(orig_oracle),MD_equals_fresh_M=True,dependencies_SHA256=before,old_header_identical=True,old_CABI_and_exports_retained=True,new_CABI=required,original_NEEDED=oldneeded,new_NEEDED=newneeded,added_NEEDED=sorted(set(newneeded)-set(oldneeded)),removed_NEEDED=sorted(set(oldneeded)-set(newneeded)),actual_DTK_bindings=parent['actual_DTK_bindings'],existing_HIP_provider=hip,link_argv=link,control_contract_passed=True,Native_runtime_and_forward_qualification='NOT_RUN',Model_gain='UNMEASURED',seconds=time.perf_counter()-started)
except BaseException as exc:q.update(error=repr(exc),traceback=traceback.format_exc(),seconds=time.perf_counter()-started)
(B/'CPU-build.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['completed','passed','actual_step','library_SHA256','error','seconds']}));assert q['passed']
