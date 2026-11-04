"""One changed entry integration + real source/ELF metadata; no Runtime execution."""
from pathlib import Path
import hashlib,json,os,subprocess,tarfile,time,traceback
D=Path(__file__).resolve().parent;R=Path('/refactor');B=R/'candidates/Deep-r467-current-profile-CPU-v1';H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
 assert not B.exists(),'Existing CPU output: reconcile original, no replay';B.mkdir(parents=True)
 q=dict(completed=False,passed=False,Native_dlopen=False,SDK_Runtime=False,weights_opened=False,GPU=False,Model_READY=False,new_compile=0,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'])
 def save():(B/'result.json').write_text(json.dumps(q,indent=2,allow_nan=False)+'\n')
 save();start=time.monotonic()
 try:
  m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes())
  for n,h in m['files_SHA256'].items():assert H(D/n)==h,n
  q['source_manifest_SHA256']=H(D/'SOURCE_MANIFEST.json');pin=json.loads((D/'ABI_EVIDENCE.json').read_bytes());p=R/'candidates/Deep-HC-fullpre-Native-r377-cpu-v1/CPU-build.json';assert H(p)==pin['CPU750_SHA256'];receipt=json.loads(p.read_bytes());assert receipt['passed']and receipt['library_SHA256']==pin['Native_SHA256']and H(receipt['library'])==pin['Native_SHA256'];assert len(receipt['all50_objects_SHA256'])==50
  for path,h in receipt['all50_objects_SHA256'].items():assert H(path)==h,path
  for path,v in receipt['actual_DTK_bindings'].items():assert H(path)==v['SHA256'],path
  for path,h in pin['actual_profile_sources_SHA256'].items():assert H(path)==h,path
  nm=subprocess.run(['nm','-D','--defined-only',receipt['library']],capture_output=True,text=True,timeout=30);assert nm.returncode==0;names={s.split()[-1]for s in nm.stdout.splitlines()if s.split()};assert set(pin['CABI'])<=names
  q['actual_ELF_CPU750_50objects_SDK_profile_sources_verified']=True;q['CABI']=pin['CABI'];q['Native_SHA256']=pin['Native_SHA256']
  archive=R/'results/Deep438-1378-analysis-terminal-v1.tar.gz';assert H(archive)=='c51c493632f2a1b64f7a396db59a85a12e06f76a507e94cd82aab511b91daed2'
  with tarfile.open(archive,'r:gz')as t:b=t.extractfile('run/private-input-IDs.json').read()
  assert hashlib.sha256(b).hexdigest()=='1717324f35a2f7e987928e4a87b7c878a8115f0910f3af56fe4e611cd62c769f';inputfile=B/'private-input-IDs.json';inputfile.write_bytes(b)
  env=dict(os.environ,DEEP467_PARENT_ROOT=str(R/'incoming/Deep-HC-fullpre-public-pair-r449/source-v1'),DEEP467_BASE_PROVIDER_ROOT=str(R/'incoming/public-Deep-native-provider-r427/source-v3'),DEEP467_PUBLIC_ROOT=str(R/'incoming/public-GLM-native-provider-r425'),DEEP467_NATIVE_SOURCE_ROOT=str(R/'incoming/Deep-HC-fullpre-Native-r377/cpu-v1'),DEEP467_MODEL_CONFIG='/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8/config.json')
  result=B/'entry-recording.json';argv=['python3','-B',str(D/'check_entry_CPU.py'),'--input',str(inputfile),'--output',str(result)];z=subprocess.run(argv,capture_output=True,text=True,env=env,timeout=180);(B/'stdout.log').write_text(z.stdout);(B/'stderr.log').write_text(z.stderr);q['actual_entry_command']=dict(argv=argv,returncode=z.returncode,stdout_SHA256=H(B/'stdout.log'),stderr_SHA256=H(B/'stderr.log'));save();assert z.returncode==0
  entry=json.loads(result.read_bytes());assert entry['passed']and entry['actual_entry_main_execute']and not entry['Native_dlopen']and not entry['SDK_Runtime'];q['actual_changed_entry_consumer']=entry;q.update(completed=True,passed=True,seconds=time.monotonic()-start);save()
 except BaseException as e:q.update(completed=True,passed=False,error=repr(e),traceback=traceback.format_exc(),seconds=time.monotonic()-start);save();raise
 print(json.dumps(dict(passed=q['passed'],actual_CPU=q['actual_CPU'],receipt=str(B/'result.json'),receipt_SHA256=H(B/'result.json'),GPU=False)))
if __name__=='__main__':main()
