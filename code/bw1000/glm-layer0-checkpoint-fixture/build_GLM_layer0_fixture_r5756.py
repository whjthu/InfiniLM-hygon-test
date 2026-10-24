"""One bridge TU only, no CPU dlopen/GPU. Reuse exact actual717 objects."""
import pathlib,json,hashlib,subprocess,shlex,shutil,os,time
r=pathlib.Path('/refactor');i=r/'incoming';base=r/'candidates/glm-layer0-fixture-r5756';assert not base.exists();(base/'src').mkdir(parents=True)
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();m=json.loads((i/'GLM-layer0-CPU-manifest-r5756.json').read_text())
for n,h in m['sources_SHA256'].items():assert sha(i/n)==h,n
pp=r/'candidates/glm-crossrequest-model-r7328/CPU-build.json';assert sha(pp)=='8b38d6f42f0f1821be3d61fdcf7dc6cb84e2a89c739031c9484dd6e956cb7f2a';p=json.loads(pp.read_text());assert sha(p['library'])==p['library_SHA256']=='3c780802bc9536cf0b8d407db7824ec0482b4fc6a738759c588a1d687fe3a7b0'
layer=json.loads((r/'candidates/glm-transient-layer-r5681/CPU-build.json').read_text());rec=next(v for v in layer['compiled']if pathlib.Path(v['source']).name=='glm_moe_dsa_attention_prepare_r809.cpp');cmd=next(a for a in layer['commands']if '-c'in a and rec['source']in a);flags=[];j=1
while j<len(cmd):
 if cmd[j]in['-c','-o','-MF','-MT','-MQ']:j+=2;continue
 if cmd[j]in['-MD','-MMD']:j+=1;continue
 flags.append(cmd[j]);j+=1
commands=[];report={'completed':False,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'CPU_only':True};start=time.perf_counter()
def run(a,label):
 commands.append(a);q=subprocess.run(a,text=True,capture_output=True);(base/(label+'.log')).write_text(q.stdout+q.stderr)
 if q.returncode:raise RuntimeError((label,q.returncode,q.stderr[-6000:]))
 return q.stdout
parse=lambda s:set(shlex.split(s.replace('\\\n',' ').split(':',1)[1]))
try:
 src=base/'src/glm_layer0_fixture_r5756.cpp';shutil.copyfile(i/src.name,src);obj=base/'bridge.o';dep=base/'bridge.d'
 run([cmd[0],*flags,'-MD','-MF',str(dep),'-c',str(src),'-o',str(obj)],'compile');ds=parse(dep.read_text());assert ds==parse(run([cmd[0],*flags,'-M',str(src)],'fresh-M'))
 args=next(list(a)for a in p['commands']if '-shared'in a);objects=[v for v in args if v.endswith('.o')];objectSHA={v:sha(v)for v in objects};assert len(objects)==27
 args.insert(1,str(obj));lib=base/'libglm_layer0_fixture_r5756.so';args[args.index('-o')+1]=str(lib);run(args,'link')
 exports=run(['nm','-D','--defined-only',str(lib)],'exports');required=['create','keys','load','finish','prepare','append_check','run','destroy','error'];assert all('glm_l0_'+n+'_r5756'in exports for n in required)
 extract=run(['python3',str(i/'GLM_layer0_extract_r5756.py')],'extract-CPU');er=json.loads(extract);assert er['completed']and not er['GPU_initialized']
 report.update(completed=True,elapsed_seconds=time.perf_counter()-start,commands=commands,ABI=1,library=str(lib),library_SHA256=sha(lib),parent_DSO_SHA256=p['library_SHA256'],parent_receipt_SHA256=sha(pp),source_SHA256=sha(src),object_SHA256=sha(obj),dependency_SHA256={v:sha(v)for v in ds},fresh_M_equal_MD=True,reused_objects_SHA256=objectSHA,extract=er,exports=required,scope='One TP8rank0-local layer0 fixture bridge. No CPU dlopen/GPU/attention/fullModel. Actualcheckpoint fixture, NOT historical479 QKV.')
except BaseException as e:report.update(failure=repr(e),commands=commands);(base/'CPU-failure.json').write_text(json.dumps(report,indent=2));raise
else:(base/'CPU-build.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:report[k]for k in ['completed','producer','elapsed_seconds','library_SHA256']}))
