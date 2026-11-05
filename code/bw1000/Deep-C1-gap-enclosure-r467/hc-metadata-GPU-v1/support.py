from pathlib import Path
import hashlib,json,os,socket,subprocess,time
D=Path(__file__).resolve().parent;HOST_R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');R=Path('/refactor')if D.is_relative_to('/refactor')else HOST_R;LABEL='Deep-HC-metadata-r467-device-v1';NODE='a02r4n04';OUT=R/'results'/LABEL
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def actual(p):return R/Path(p).relative_to('/refactor')if str(p).startswith('/refactor/')else Path(p)
def save(p,q):
 p=Path(p);tmp=p.with_name(p.name+'.tmp-'+str(os.getpid()));tmp.write_text(json.dumps(q,indent=2)+'\n');os.replace(tmp,p)
def verify():
 m=json.loads((D/'GPU_MANIFEST.json').read_bytes());assert all(H(D/n)==h for n,h in m['files_SHA256'].items());b=m['CPU'];p=actual(b['path']);assert H(p)==b['SHA256'];q=json.loads(p.read_bytes());assert q['passed']and q['actual_CPU']=='910407.1048'and q['new_HOST']==1 and q['new_GPU']==0 and not q['probe_executed']and q['source_manifest_SHA256']=='deb468c46e3c544d192491e80dd2d3bd34f695b73a6c9911860765e4279f7b47';assert H(actual(q['executable']))==q['executable_SHA256']==m['executable_SHA256']
 for p,h in q['parent_libraries'].items():assert H(actual(p))==h
 return m,q
