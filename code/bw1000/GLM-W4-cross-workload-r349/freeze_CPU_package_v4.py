import pathlib,json,hashlib,base64
D=pathlib.Path(__file__).parent;O=D/'owner';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
preparation={'scope':'Three complete recorded workloads, each two warmups and one ABBA W2/W4 block; buffered HTTP; no online SLO or full benchmark qualification','GPU_GO':False,'source_ancestry':'GLM-prefill-W2-W4-owner-r307','parent_manifest_SHA256':'3508075aa744f3bfb8f3631af73b1112f8905b0c810800b7ec0819885b718545','TP':8,'PP':4,'cards':32,'context':32768,'ordinary_budget':4096,'HTTP':18,'Native':18,'Native_SHA256':'038d4ff2e7aee2e6db80f691dd885222f3e3160e25ae9d0e269296daf9596c00','no_CXX_recompile':True,'no_decode_math_or_SDK_change':True,'fixture_card_SHA256':sha(O/'FIXTURE_CARD.json')}
(O/'PREPARATION.json').write_bytes((json.dumps(preparation,indent=2)+'\n').encode())
files={p.relative_to(O).as_posix():sha(p) for p in sorted(O.rglob('*')) if p.is_file() and p.name!='SOURCE_MANIFEST.json'}
mf={'files_SHA256':files,'parent_manifest_SHA256':preparation['parent_manifest_SHA256'],'scope':preparation['scope'],'external_fixture_SHA256':[x['request_SHA256'] for x in json.loads((O/'FIXTURE_CARD.json').read_bytes())['fixtures']]}
(O/'SOURCE_MANIFEST.json').write_bytes((json.dumps(mf,indent=2)+'\n').encode())
for p in O.glob('*.py'):compile(p.read_bytes(),str(p),'exec')
for p in O.glob('*.sh'):assert b'\r' not in p.read_bytes(),p
compile((D/'CPU_whole_glue.py').read_bytes(),'CPU_whole_glue.py','exec')
shell=(D/'CPU_encode_inside_v3.sh').read_bytes().replace(b'/refactor/incoming/GLM-prefill-W2-W4-owner-r307',b'/refactor/incoming/GLM-W4-cross-workload-r349/runtime_v4/owner').replace(b'encoder_cpu_v3/CPU_encode_fixtures_v3.py',b'runtime_v4/CPU_whole_glue.py')
(D/'CPU_whole_inside.sh').write_bytes(shell)
allfiles={'owner/'+p.relative_to(O).as_posix():base64.b64encode(p.read_bytes()).decode() for p in O.rglob('*') if p.is_file()}
for n in ['CPU_whole_glue.py','CPU_whole_inside.sh']:allfiles[n]=base64.b64encode((D/n).read_bytes()).decode()
source={n:hashlib.sha256(base64.b64decode(b)).hexdigest() for n,b in allfiles.items()};sourceMF=json.dumps({'files_SHA256':source,'owner_manifest_SHA256':sha(O/'SOURCE_MANIFEST.json'),'CPU_scope':'recording CAPI/HTTP/WORLD/process with actual owner/driver/client entry; no GPU or Native Context'},indent=2)+'\n'
(D/'CPU_SOURCE_MANIFEST.json').write_bytes(sourceMF.encode());allfiles['CPU_SOURCE_MANIFEST.json']=base64.b64encode(sourceMF.encode()).decode()
upload="""import pathlib,json,base64,subprocess,os,hashlib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';d=r/'incoming/GLM-W4-cross-workload-r349/runtime_v4';assert not d.exists();d.mkdir()
files=FILES
for n,b in files.items():
 p=d/n;p.parent.mkdir(exist_ok=True,parents=True);p.write_bytes(base64.b64decode(b))
mf=json.loads((d/'CPU_SOURCE_MANIFEST.json').read_bytes());assert all(hashlib.sha256((d/n).read_bytes()).hexdigest()==h for n,h in mf['files_SHA256'].items())
env={k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_'))}
a=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c4','-w','a14r2n20','--gres=none','--mem=8192M','-t','00:06:00','--job-name=GLM349-whole-glue-v4','--output='+str(r/'logs/GLM349-whole-glue-v4-%J.log'),'env','WAVE_INSIDE_R2323=GLM-W4-cross-workload-r349/runtime_v4/CPU_whole_inside.sh','bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
with (r/'logs/GLM349-whole-glue-v4-client.log').open('x') as f:p=subprocess.Popen(a,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'CPU_submit_PID':p.pid,'argv':a,'GPU_submitted':False,'CPU_source_manifest_SHA256':hashlib.sha256((d/'CPU_SOURCE_MANIFEST.json').read_bytes()).hexdigest()}))
""".replace('FILES',repr(allfiles))
(D/'remote_upload_whole_v4.py').write_bytes(upload.encode())
print(json.dumps({'CPU_source_manifest_SHA256':sha(D/'CPU_SOURCE_MANIFEST.json'),'owner_manifest_SHA256':sha(O/'SOURCE_MANIFEST.json'),'GPU_GO':False}))
