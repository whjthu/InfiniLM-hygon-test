"""Future CPU-only build entry; five HOST replacements, no GPU compilation/load."""
import pathlib,json,hashlib,subprocess,shlex,os,time,traceback
R=pathlib.Path('/refactor');D=pathlib.Path(__file__).resolve().parent;B=R/'candidates/Kimi-TP32-Native-r120f';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def deps(p):return sorted(set(shlex.split(p.read_text().replace('\\\n',' ').split(':',1)[1])))
def fresh_command(actual,source,output):
 prefix=actual[:actual.index(str(source))];out=[];i=0
 while i<len(prefix):
  if prefix[i]in ['-MD','-MMD','-c']:i+=1;continue
  if prefix[i]=='-MF':i+=2;continue
  out.append(prefix[i]);i+=1
 return [*out,'-M','-MF',str(output),str(source)]
def main():
 assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
 m=json.loads((D/'manifest-r120f.json').read_bytes())
 for n,h in m['source_SHA256'].items():assert sha(D/n)==h,n
 for p,h in m['required_DSO_SHA256'].items():assert sha(p)==h,p
 B.mkdir(exist_ok=False);start=time.monotonic();q={'completed':False,'CPU_only':True,'GPU':False,'GPU_initialized':False,'dlopen':False,'Runtime_create':False,'HOST_TUs_compiled':0,'GPU_TUs_compiled':0,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'commands':[],'source_manifest_SHA256':sha(D/'manifest-r120f.json')}
 def run(cmd,label,seconds=200):
  q['commands'].append(cmd);z=subprocess.run(cmd,text=True,capture_output=True,timeout=seconds);(B/(label+'.log')).write_text(z.stdout+z.stderr);assert z.returncode==0,(label,z.stderr[-12000:]);return z.stdout
 try:
  p=json.loads((D/'parent-commands-r120f.json').read_bytes());base=p['Native305726'];assert base['completed']and base['native_SHA256']==m['base_Native_SHA256']and sha(base['native'])==m['base_Native_SHA256']
  for f,h in base['all_linked_objects'].items():assert sha(f)==h,f
  # Strict old concrete-type ODR scan before touching any object.
  suffixes=['/pair_probe.o','/worker.o','/delta.o','/head.o','/policy.o'];replace={}
  for old in base['all_linked_objects']:
   if any(old.endswith(x)for x in suffixes):replace[old]=B/pathlib.Path(old).name
  assert len(replace)==5
  for old in base['all_linked_objects']:
   names=subprocess.check_output(['nm','-C',old],text=True,timeout=30)
   if any(x in names for x in ['ShardOnlyVocabHeadR9183','TextCausalLMShardR9183','KimiK3ForConditionalGenerationShardR9183','kimi_shard_r9183::']):assert old in replace,('unreplaced head ABI reference',old)
  pure=['g++','-std=c++17','-O2','-I'+str(D),str(D/'CPU_contract_r120f.cpp'),'-o',str(B/'CPU_contract')];run(pure,'pure-compile',45);q['pure_CPU_contract']=json.loads(run([str(B/'CPU_contract')],'pure-run',10))
  source_names={'delta':'delta.cpp','worker':'worker.cpp','pair_probe':'pair_probe.cpp','head':'kimi_shard_model_r9183.cpp','policy':'kimi_shard_policy_r9183.cpp'};all_md={}
  for name,filename in source_names.items():
   record=p['batch']if name in ['delta','worker','pair_probe']else p['head'];cmd=next(c[:]for c in record['commands']if '-c'in c and pathlib.Path(c[c.index('-o')+1]).name==name+'.o');cmd[1:1]=['-I'+str(D),'-I'+str(D/'engine')]
   src=D/filename;cmd[cmd.index('-c')+1]=str(src);cmd[cmd.index('-o')+1]=str(B/(name+'.o'));cmd[cmd.index('-MF')+1]=str(B/(name+'.d'));run(cmd,name);q['HOST_TUs_compiled']+=1
   run(fresh_command(cmd,src,B/(name+'.fresh.d')),name+'-freshMD',75);actual=deps(B/(name+'.d'));assert actual==deps(B/(name+'.fresh.d')),(name,'dependency-set mismatch');all_md[name]={f:sha(f)for f in actual}
   assert str(D/'tp32_contract_r120f.hpp')in all_md[name]if name in ['pair_probe','head']else True
   if name in ['delta','worker']:assert str(D/'batch_state_r9122.hpp')in all_md[name]and str(D/'state_only_r9113.hpp')in all_md[name]
  native=B/'libKimi_TP32_Native_r120f.so';link=next(c[:]for c in base['commands']if '-o'in c and c[c.index('-o')+1]==base['native'])
  for old,new in replace.items():assert link.count(old)==1;link[link.index(old)]=str(new)
  link[link.index('-o')+1]=str(native)
  for i,x in enumerate(link):
   if x.startswith('-Wl,-Map'):link[i]='-Wl,-Map,'+str(B/'link.map')
  assert '-Wl,--no-undefined'in link;run(link,'link',180)
  unchanged={f:h for f,h in base['all_linked_objects'].items()if f not in replace};assert all(sha(f)==h for f,h in unchanged.items())
  env=dict(os.environ);env['LD_LIBRARY_PATH']='/usr/local/lib/python3.10/dist-packages/torch/lib:'+base['LD_LIBRARY_PATH'];z=subprocess.run(['ldd','-r',str(native)],env=env,text=True,capture_output=True,timeout=60);assert z.returncode==0 and 'not found'not in z.stdout+z.stderr and 'undefined symbol:'not in z.stdout+z.stderr;(B/'ldd.log').write_text(z.stdout+z.stderr)
  defs=run(['nm','-D','--defined-only',str(native)],'exports',40);assert all(x in defs for x in ['modern_kimi_engine_create_batch_r9122c','modern_kimi_batch_replay_r9122','modern_kimi_shard_head_expected_r9183','modern_kimi_W2_pair_counts_r9280'])
  for p,h in m['required_DSO_SHA256'].items():assert sha(p)==h,p
  q.update(completed=True,native=str(native),native_SHA256=sha(native),all_linked_object_SHA256={f:sha(f)for f in link if f.endswith('.o')},unchanged_objects=unchanged,replaced_objects={f:str(n)for f,n in replace.items()},actual_complete_freshMD=all_md,required_DSO_SHA256=m['required_DSO_SHA256'],ldd=z.stdout+z.stderr,defaultOFF=True,GPU_GO=False,normal_CPU_exit=True)
 except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
 q['elapsed_s']=time.monotonic()-start;(B/'CPU-build.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['completed','native_SHA256','HOST_TUs_compiled','error']}));assert q['completed']
if __name__=='__main__':main()
