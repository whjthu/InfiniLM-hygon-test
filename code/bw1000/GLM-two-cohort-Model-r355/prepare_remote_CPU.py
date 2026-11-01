"""Build one exclusive upload/CPU submit helper in L; no execution on import."""
from pathlib import Path
import base64,hashlib,io,json,tarfile
D=Path(__file__).resolve().parent;L=D.parents[3]
m=json.loads((D/'CPU_PACKAGE_MANIFEST.json').read_bytes());data=io.BytesIO()
with tarfile.open(fileobj=data,mode='w:gz')as tar:
 for n,h in m['files_SHA256'].items():
  p=D/n;raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==h,n
  info=tarfile.TarInfo(n);info.size=len(raw);info.mode=0o644;tar.addfile(info,io.BytesIO(raw))
 raw=(D/'CPU_PACKAGE_MANIFEST.json').read_bytes();info=tarfile.TarInfo('CPU_PACKAGE_MANIFEST.json');info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
payload=base64.b64encode(data.getvalue()).decode()
script="""from pathlib import Path
import base64,hashlib,io,json,tarfile,subprocess,os
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'incoming/GLM-two-cohort-Model-r355'
assert not D.exists(),'Existing source must never be overwritten/replayed'
raw=base64.b64decode(PAYLOAD);D.mkdir()
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz')as tar:
 for member in tar.getmembers():
  p=D/member.name;assert member.isfile()and not Path(member.name).is_absolute()and'..'not in Path(member.name).parts and p.resolve().is_relative_to(D.resolve())
  p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as out:out.write(tar.extractfile(member).read())
m=json.loads((D/'CPU_PACKAGE_MANIFEST.json').read_bytes())
assert all(hashlib.sha256((D/n).read_bytes()).hexdigest()==h for n,h in m['files_SHA256'].items())
rawparent=subprocess.check_output(['scontrol','show','job','910407','-o'],text=True,timeout=20);assert'UserId=aclsylqidf('in rawparent and'JobState=RUNNING'in rawparent
args=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','--ntasks-per-node=1','-c4','--gres=none','--mem=8192M','-w','a14r2n20','-t','00:10:00','--job-name=GLM-r355-whole-CPU','--output='+str(R/'logs/GLM-r355-whole-CPU-%J.log'),'--export=ALL','bash',str(D/'controller/resource_wave_host_alias_outer.sh'),'probe']
env=dict(os.environ,WAVE_INSIDE_R2323='GLM-two-cohort-Model-r355/CPU_inside.sh')
log=R/'logs/GLM-r355-whole-CPU-client.log'
with log.open('x')as out:p=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'CPU_submitted':True,'GPU':0,'PID':p.pid,'label':'GLM-r355-whole-CPU','source':str(D),'package_SHA256':hashlib.sha256((D/'CPU_PACKAGE_MANIFEST.json').read_bytes()).hexdigest()}))
""".replace('PAYLOAD',repr(payload))
p=L/'remote355-upload-CPU.py';p.write_text(script,encoding='utf8');print(json.dumps({'helper':str(p),'archive_bytes':len(data.getvalue())}))
