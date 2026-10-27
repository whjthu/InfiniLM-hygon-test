"""Optional second CPU-only entry after five-HOST Native build; no invocation."""
import pathlib,json,hashlib,subprocess,os,traceback,time
from CPU_build_r120f import fresh_command,deps
R=pathlib.Path('/refactor');D=pathlib.Path(__file__).resolve().parent;B=R/'candidates/Kimi-TP32-construct-r120f';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
 assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']);m=json.loads((D/'manifest-r120f.json').read_bytes())
 for n,h in m['source_SHA256'].items():assert sha(D/n)==h,n
 p=R/'candidates/Kimi-TP32-Native-r120f/CPU-build.json';c=json.loads(p.read_bytes());assert c['completed']and c['CPU_only']and not c['GPU']and not c['dlopen'];assert sha(c['native'])==c['native_SHA256']
 for library_path,h in c['required_DSO_SHA256'].items():assert sha(library_path)==h,library_path
 B.mkdir(exist_ok=False);q={'completed':False,'CPU_only':True,'GPU':False,'GPU_initialized':False,'dlopen':False,'HOST_TUs_compiled':0,'GPU_TUs_compiled':0,'parent_CPU_receipt_SHA256':sha(p),'native_SHA256':c['native_SHA256'],'commands':[],'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']}
 def run(cmd,label):
  q['commands'].append(cmd);z=subprocess.run(cmd,text=True,capture_output=True,timeout=180);(B/(label+'.log')).write_text(z.stdout+z.stderr);assert z.returncode==0,z.stderr[-10000:];return z.stdout
 try:
  cmd=next(x[:]for x in c['commands']if '-c'in x and x[x.index('-c')+1].endswith('/kimi_shard_model_r9183.cpp'));src=D/'construct_TP32_r120f.cpp';cmd[cmd.index('-c')+1]=str(src);cmd[cmd.index('-o')+1]=str(B/'construct.o');cmd[cmd.index('-MF')+1]=str(B/'construct.d');run(cmd,'compile');q['HOST_TUs_compiled']=1;run(fresh_command(cmd,src,B/'construct.fresh.d'),'freshMD');actual=deps(B/'construct.d');assert actual==deps(B/'construct.fresh.d')
  native=c['native'];lib=B/'libKimi_TP32_construct_r120f.so';sdk=R/'install/hygon-kimi-5f91ad7-r3672/lib';runtime=R/'candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM/python/infinilm/_lib'
  link=['g++','-shared','-pthread',str(B/'construct.o'),'-Wl,--no-as-needed',native,'-Wl,--as-needed','-L'+str(sdk),'-linfiniops','-linfiniccl','-linfinirt','-L'+str(runtime),'-linfinilm_runtime','-L/opt/dtk/cuda/cuda/lib64','-lcudart','-Wl,--no-undefined','-Wl,-rpath,'+':'.join([str(sdk),str(runtime)]),'-o',str(lib)];run(link,'link')
  env=dict(os.environ);env['LD_LIBRARY_PATH']='/usr/local/lib/python3.10/dist-packages/torch/lib:'+env.get('LD_LIBRARY_PATH','');z=subprocess.run(['ldd','-r',str(lib)],env=env,text=True,capture_output=True,timeout=60);assert z.returncode==0 and 'not found'not in z.stdout+z.stderr and 'undefined symbol:'not in z.stdout+z.stderr
  for library_path,h in c['required_DSO_SHA256'].items():assert sha(library_path)==h,library_path
  q.update(completed=True,library=str(lib),library_SHA256=sha(lib),fresh_MD_SHA256={f:sha(f)for f in actual},required_DSO_SHA256=c['required_DSO_SHA256'],GPU_GO=False,qualification_only=True,no_weights_no_model_forward=True,normal_CPU_exit=True,ldd=z.stdout+z.stderr)
 except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
 (B/'CPU-build.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['completed','library_SHA256','error']}));assert q['completed']
if __name__=='__main__':main()
