"""One arm in one fresh process; root owns staging/singleGPU steps/cleanup."""
from pathlib import Path
import os,json,sys,hashlib,socket,ctypes,re
R=Path('/refactor');D=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
arm,receipt_SHA,publication=sys.argv[1:4];assert arm in ['regular','exact']and re.fullmatch('[0-9a-f]{64}',receipt_SHA)and re.fullmatch('[0-9a-f]{40}',publication)
assert os.environ['SLURM_JOB_ID']=='910374'and socket.gethostname()=='a02r4n01'and int(os.environ['SLURM_NTASKS'])==1
m=json.loads((D/'source-manifest-r168.json').read_bytes())
for n,h in m['files_SHA256'].items():assert sha(D/n)==h,n
bp=R/'candidates/Kimi-I96-shared-storage-probe-r168/CPU-build.json';assert sha(bp)==receipt_SHA;b=json.loads(bp.read_bytes());assert b['completed']and b['CPU_only']and not b['GPU']and not b['Native_loaded']and b['source_manifest_SHA256']==sha(D/'source-manifest-r168.json')
assert b['shared_helper_source_equivalence']and b['no_comm_no_fake_world'];a=b['arms'][arm];assert sha(a['native'])==a['native_SHA256']and sha(D/('kimi_moe_storage_'+arm+'_r168.cpp'))==a['source_SHA256']
step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];out=R/'results'/('Kimi-I96-storage-r168-'+arm+'-'+step+'.json');assert not out.exists()
# Exactly one candidate Native load. No old adapter, Torch, second candidate, or factory substitution.
lib=ctypes.CDLL(a['native']);api=lib.modern_kimi_I96_storage_probe_r168;api.argtypes=[];api.restype=ctypes.c_int;lib.modern_kimi_I96_storage_probe_result_r168.restype=ctypes.c_char_p
class DI(ctypes.Structure):_fields_=[('path',ctypes.c_char_p),('base',ctypes.c_void_p),('symbol',ctypes.c_char_p),('addr',ctypes.c_void_p)]
dl=ctypes.CDLL('libdl.so.2');dl.dladdr.argtypes=[ctypes.c_void_p,ctypes.POINTER(DI)];z=DI();assert dl.dladdr(ctypes.cast(api,ctypes.c_void_p),ctypes.byref(z));assert sha(z.path.decode())==a['native_SHA256']
rc=api();data=json.loads(lib.modern_kimi_I96_storage_probe_result_r168());report={'arm':arm,'actual_step':step,'publication_commit':publication,'CPU_receipt_SHA256':receipt_SHA,'candidate_Native_SHA256':a['native_SHA256'],'candidate_source_SHA256':a['source_SHA256'],'actual_dladdr_path':z.path.decode(),'actual_dladdr_SHA256':sha(z.path.decode()),'source_manifest_SHA256':sha(D/'source-manifest-r168.json'),'returncode':rc,'data':data,'fullModel_fit':'UNKNOWN_STOP','scope':'same production helper actual allocation; not TP32 execution or fullMoE constructor'}
out.write_text(json.dumps(report,indent=2)+'\n');assert rc==0 and data['completed']and data['normal_Runtime_close']and data['actual_alias_geometry']['leaf_aliases_checked']==5376
print('KIMI_I96_STORAGE_R168_NORMAL',arm,step,flush=True)
