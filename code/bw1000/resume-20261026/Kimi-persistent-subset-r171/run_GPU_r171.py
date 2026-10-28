from pathlib import Path
import os,json,sys,hashlib,socket,ctypes
R=Path('/refactor');D=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=json.loads(Path(sys.argv[1]).read_bytes());assert m['GPU_authorized']is True and m['source_published']is True,'GPU_GO=false until root release'
assert os.environ['SLURM_JOB_ID']=='910374'and socket.gethostname()=='a02r4n01'and int(os.environ['SLURM_NTASKS'])==1
for n,h in json.loads((D/'source-manifest-r171.json').read_bytes())['files_SHA256'].items():assert sha(D/n)==h,n
bp=R/'candidates/Kimi-persistent-subset-r171/CPU-build.json';assert sha(bp)==m['CPU_build_SHA256'];b=json.loads(bp.read_bytes());assert b['completed']and b['CPU_only']and not b['GPU']and not b['Native_loaded']and b['source_manifest_SHA256']==sha(D/'source-manifest-r171.json')
assert sha(b['native'])==b['native_SHA256']==m['candidate_Native_SHA256']
config=Path('/shared/Kimi-K3/config.json');assert sha(config)=='9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213'
lib=ctypes.CDLL(b['native']);api=lib.modern_kimi_persistent_subset_r171;api.argtypes=[ctypes.c_char_p];api.restype=ctypes.c_int;lib.modern_kimi_persistent_subset_result_r171.restype=ctypes.c_char_p
class DI(ctypes.Structure):_fields_=[('path',ctypes.c_char_p),('base',ctypes.c_void_p),('symbol',ctypes.c_char_p),('addr',ctypes.c_void_p)]
dl=ctypes.CDLL('libdl.so.2');dl.dladdr.argtypes=[ctypes.c_void_p,ctypes.POINTER(DI)];proof=[]
for name in ['modern_kimi_persistent_subset_r171',b['cache_factory_symbol']]:
 z=DI();assert dl.dladdr(ctypes.cast(getattr(lib,name),ctypes.c_void_p),ctypes.byref(z));assert sha(z.path.decode())==b['native_SHA256'];proof.append({'symbol':name,'actual_path':z.path.decode(),'actual_DSO_SHA256':sha(z.path.decode())})
step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];out=R/'results'/('Kimi-persistent-subset-r171-'+step+'.json');assert not out.exists()
rc=api(config.read_bytes());data=json.loads(lib.modern_kimi_persistent_subset_result_r171());report={'actual_step':step,'publication_commit':m['publication_commit'],'CPU_build_SHA256':m['CPU_build_SHA256'],'source_manifest_SHA256':sha(D/'source-manifest-r171.json'),'candidate_Native_SHA256':b['native_SHA256'],'probe_source_SHA256':b['probe_source_SHA256'],'production_prefix_SHA256':b['production_prefix_SHA256'],'actual_bindings':proof,'returncode':rc,'data':data,'fullModel_GO':False};out.write_text(json.dumps(report,indent=2)+'\n')
assert rc==0 and data['completed']and data['normal_Runtime_close'];print('KIMI_PERSISTENT_SUBSET_R171_NORMAL',step,flush=True)
