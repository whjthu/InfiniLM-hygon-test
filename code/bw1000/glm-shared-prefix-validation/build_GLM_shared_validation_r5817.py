"""One bridge TU only, no CPU dlopen/GPU. Reuse exact actual717 objects."""
import pathlib,json,hashlib,subprocess,shlex,shutil,os,time
r=pathlib.Path('/refactor');i=r/'incoming';base=r/'candidates/glm-shared-validation-model-r5817';assert not base.exists();(base/'src').mkdir(parents=True)
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();m=json.loads((i/'GLM-shared-validation-CPU-manifest-r5817.json').read_text())
for n,h in m['sources_SHA256'].items():assert sha(i/n)==h,n
pp=r/'candidates/glm-window-model-r5797/CPU-build.json';assert sha(pp)=='611d400db8802e6ac38931db0c222527d45adaa1f56d17f0c34a45a81fc31624';p=json.loads(pp.read_text());assert sha(p['library'])==p['library_SHA256']=='7de2990fe7028fe8514ce813a4a57f9d25315c4b2df4b84db0e8770af24dd910'
layer=json.loads((r/'candidates/glm-transient-layer-r5681/CPU-build.json').read_text());rec=next(v for v in layer['compiled']if pathlib.Path(v['source']).name=='glm_moe_dsa_projected_cache_view_r927.cpp');cmd=next(a for a in layer['commands']if '-c'in a and rec['source']in a);flags=[];j=1
while j<len(cmd):
 if cmd[j]in['-c','-o','-MF','-MT','-MQ']:j+=2;continue
 if cmd[j]in['-MD','-MMD']:j+=1;continue
 flags.append(cmd[j]);j+=1
assert sha('/refactor/candidates/glm-deferred-attention-model-5f91ad7-r5096/src/glm_prefix_proof_r1645.hpp')=='4df185ee83bc7706e524d99ffc24ac151142f003334a700d6db1197b82b013da'
commands=[];report={'completed':False,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'CPU_only':True};start=time.perf_counter()
def run(a,label):
 commands.append(a);q=subprocess.run(a,text=True,capture_output=True);(base/(label+'.log')).write_text(q.stdout+q.stderr)
 if q.returncode:raise RuntimeError((label,q.returncode,q.stderr[-6000:]))
 return q.stdout
parse=lambda s:set(shlex.split(s.replace('\\\n',' ').split(':',1)[1]))
try:
 src=base/'src/glm_shared_validation_cache_r5814.cpp';shutil.copyfile(i/src.name,src);obj=base/'bridge.o';dep=base/'bridge.d'
 run([cmd[0],*flags,'-MD','-MF',str(dep),'-c',str(src),'-o',str(obj)],'compile');ds=parse(dep.read_text());assert ds==parse(run([cmd[0],*flags,'-M',str(src)],'fresh-M'))
 assert '/refactor/candidates/glm-deferred-attention-model-5f91ad7-r5096/src/glm_prefix_proof_r1645.hpp' in ds and sha('/refactor/candidates/glm-deferred-attention-model-5f91ad7-r5096/src/glm_prefix_proof_r1645.hpp')=='4df185ee83bc7706e524d99ffc24ac151142f003334a700d6db1197b82b013da'
 args=next(list(a)for a in p['commands']if '-shared'in a);objects=[v for v in args if v.endswith('.o')];objectSHA={v:sha(v)for v in objects};assert len(objects)==28
 oldcache='/refactor/candidates/glm-layer77-model-r5775/bridge.o';assert oldcache in args;args=[str(obj)if v==oldcache else v for v in args];lib=base/'libglm_shared_validation_model_r5817.so';args[args.index('-o')+1]=str(lib);run(args,'link')
 exports=run(['nm','-D','--defined-only',str(lib)],'exports');required=['modern_glm_shared_validation_counts_r5814','modern_glm_cross_control_r5775','modern_glm_cross_journal_r5775','modern_glm_cross_error_r5775'];assert all(n in exports for n in required)
 er={'scope':'NOT_RUN no weights/fixture/extraction/GPU; onecacheTUcompileonly'}
 report.update(completed=True,elapsed_seconds=time.perf_counter()-start,commands=commands,ABI=1,library=str(lib),library_SHA256=sha(lib),parent_DSO_SHA256=p['library_SHA256'],parent_receipt_SHA256=sha(pp),source_SHA256=sha(src),object_SHA256=sha(obj),dependency_SHA256={v:sha(v)for v in ds},fresh_M_equal_MD=True,reused_objects_SHA256=objectSHA,extract=er,exports=required,scope='One changed cache TU; existing PrefixProof1645 header SHA bound. DefaultOFF shared validation readback bypass only, no math/class/header layout change. CPU compile/link/nm only, no dlopen/GPU/weights.')
except BaseException as e:report.update(failure=repr(e),commands=commands);(base/'CPU-failure.json').write_text(json.dumps(report,indent=2));raise
else:(base/'CPU-build.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:report[k]for k in ['completed','producer','elapsed_seconds','library_SHA256']}))
