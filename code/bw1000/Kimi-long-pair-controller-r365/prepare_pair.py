"""Freeze only the run-d worker binding delta and prepare two bounded CPU jobs."""
from pathlib import Path
import ast,base64,hashlib,json

C=Path(__file__).resolve().parent;T=C.parent/'Kimi-long-target-bootstrap-r364';G=C.parents[2];L=G.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
worker='61ac8b04af464d76b4ce46f0793e6d8a2a8e57f41224d6228c5b826bda6ffd31'
oldT=C.parent/'Kimi-long-target-bootstrap-r356';oldC=C.parent/'Kimi-long-pair-controller-r357'
for D in [T,C]:
    for p in D.iterdir():
        if p.suffix=='.py':ast.parse(p.read_bytes())
        if p.suffix=='.sh':assert b'\r'not in p.read_bytes()
tm=dict(state='RUN_D_WORKER_BINDING_CPU_PENDING',GPU_GO=False,GPU=0,default_OFF=True,
    files_SHA256={p.name:sha(p)for p in sorted(T.iterdir())if p.is_file()and p.name!='SOURCE_MANIFEST.json'},
    runtime_directory=T.name,parent_source_manifest_SHA256=sha(oldT/'SOURCE_MANIFEST.json'),
    Native_SHA256='4dfdc9a7809b367ef91b476280f12bca6da0c12511eb14dae679147382246505',
    immutable_worker_manifest_SHA256=worker,CPU_receipt='/refactor/candidates/'+T.name+'/CPU-result.json',
    qualification_scope='Same327/qualified8df/252508 diagnostic; no latest99/ec inheritance. CPU providers do not qualify actual Context/weights/Model or acceptance.',
    delta='Worker binding only: actual362b61ac unchanged worker+memoryfloor. Source/CLI/marker/passivePID bound;333/335/337 and327DSO math immutable; original captureFDlease/deadline/cancel/fourarms unchanged.')
(T/'SOURCE_MANIFEST.json').write_bytes((json.dumps(tm,indent=2)+'\n').encode());target_sha=sha(T/'SOURCE_MANIFEST.json')
p=C/'execution-plan.template.json';plan=json.loads(p.read_bytes());plan.update(run='longpair-20261102-d',target_manifest_SHA256=target_sha,
    scope='Four engine-level original arms, changed actual362b worker; source+twoCPUreceipts+newpublishedclaim before once-only GPU. No HTTP SLO or acceptance/performance preclaim.')
p.write_bytes((json.dumps(plan,indent=2)+'\n').encode())
canonical=json.loads((oldC/'SOURCE_MANIFEST.json').read_bytes())['files_SHA256']
cm=dict(state='RUN_D_WORKER_BINDING_CPU_PENDING',GPU_GO=False,default_enabled=False,
    files_SHA256={n:sha(C/n)for n in sorted(canonical)},outer_wrapper_SHA256=sha(C/'outer.sh'),
    target_bootstrap=T.name,target_manifest_SHA256=target_sha,immutable_worker_SHA256=worker,
    immutable_driver_SHA256='445f96b37c1ceab9e62c3599879834b8be21da9b9585384c942c12ffef86644f',
    runtime_directory=C.name,parent_source_manifest_SHA256=sha(oldC/'SOURCE_MANIFEST.json'),
    delta='Source/runtime/target paths and realworker61ac identity updated; same357 state machine,original29400/28800/32400epoch,90saccount/NFSbounds,passiveproc,no retry/reload,owned cleanup.',
    CPU=dict(parent=910407,node='a14r2n20',CPUs=4,memory_MiB=8192,hard_s=360,inner_s=300,GPU=0,Native_dlopen=False))
(C/'SOURCE_MANIFEST.json').write_bytes((json.dumps(cm,indent=2)+'\n').encode());controller_sha=sha(C/'SOURCE_MANIFEST.json')
task=dict(owner='/root/kimi_operator_tuning',state='SOURCE_AND_TWO_CPU_CLAIMS_READY',GPU=0,GPU_GO=False,
    run='longpair-20261102-d',target_source=str(T.relative_to(G)).replace('\\','/'),controller_source=str(C.relative_to(G)).replace('\\','/'),
    target_manifest_SHA256=target_sha,controller_manifest_SHA256=controller_sha,worker_manifest_SHA256=worker,
    CPU_claims=[dict(parent=910407,node='a14r2n20',tasks=1,CPUs=4,memory_MiB=8192,hard_s=360,GPU=0,source=T.name),
                dict(parent=910407,node='a14r2n20',tasks=1,CPUs=4,memory_MiB=8192,hard_s=360,GPU=0,source=C.name)],
    CPU_scope='Actualbootstrap/qualifiedloader at recordedNative/tensor boundaries,4realleadercaptureTCP/cancel/deadline/errorcleanup,actual362bserve/335DraftClient realTCP with recordingloader,actualcontroller/backend/proc/marker/accounting contracts; noNativeContext/weights/device.',
    intended_once_only_GPU_scope=dict(target_parent=910380,target_nodes=['a14r2n08','a14r2n09','a14r2n10','a14r2n12'],target_cards=32,TP=16,PP=2,
        draft_parent=910371,draft_node='a02r3n15',draft_cards=1,CPU_parent=910407,CPU_node='a14r2n20',GPU_max_s=29400,Native_max_s=28800,CPU_max_s=32400,
        requests=[['retained8635','AR'],['near32k','AR'],['near32k','spec'],['retained8635','spec']],prompt_tokens=[8635,32638],output_budget=128,context=32768,
        no_retry_reload=True,no_fifth_or_warmup_request=True,profile=False,first_capture_before_firstproposal='5tap512rows/finite/epoch/node2to0 route',
        final_decision='Same residency4arms only; acceptance/useful tokens/verify/replay/draft/transfer/emit/cardcost/quality/EOS/all32normal+draftclose+fiveCPUcleanup required; oldARs not pooled'),
    prior_failures='536listener/539captureport/544draftfloor and362hostparse/704CPU preserved;362b705CPU and662actualdraftgeometry+normalclose reused',
    authorization='Root explicit changed-worker run-d once after two newCPUreceipts+concrete resourceclaim ordinarypush and fresh5node idle/dedup; no additional rootGO needed; this sourceCPU stage has no GPU_GO')
(G/'coordination/tasks/Kimi-long-pair-run-d-r364365.json').write_bytes((json.dumps(task,indent=2)+'\n').encode())
payload={}
for D,m in [(T,tm),(C,cm)]:
    payload[D.name]={n:base64.b64encode((D/n).read_bytes()).decode()for n in list(m['files_SHA256'])+['SOURCE_MANIFEST.json']}
script="""from pathlib import Path
import json,hashlib,base64,subprocess,os
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');jobs=[]
for directory,files in __PAYLOAD__.items():
 D=R/'incoming'/directory;assert not D.exists();D.mkdir()
 for n,b in files.items():(D/n).write_bytes(base64.b64decode(b))
 m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert all(hashlib.sha256((D/n).read_bytes()).hexdigest()==h for n,h in m['files_SHA256'].items())
for directory,kind,digest in [('Kimi-long-target-bootstrap-r364','target',__TARGET_SHA__),('Kimi-long-pair-controller-r365','controller',__CONTROLLER_SHA__)]:
 label='Kimi-run-d-'+kind+'-CPU';env=dict(os.environ,WAVE_INSIDE_R2323=directory+'/CPU_inside.sh')
 env['KIMI_R346_CPU_MANIFEST_SHA'if kind=='target'else'KIMI_LONG_CONTROLLER_SOURCE_SHA_R347']=digest
 args=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c4','--gres=none','--mem=8192M','-w','a14r2n20','-t','00:06:00','--job-name='+label,'--output='+str(R/'logs'/(label+'-%J.log')),'--export=ALL','bash',str(R/'incoming/Kimi-long-pair-controller-r365/outer.sh'),'probe']
 with(R/'logs'/(label+'-srun-client.log')).open('x')as f:p=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 jobs.append(dict(label=label,PID=p.pid,source_manifest_SHA256=digest))
print(json.dumps({'GPU':0,'CPU_jobs_submitted':jobs}))
""".replace('__PAYLOAD__',repr(payload)).replace('__TARGET_SHA__',repr(target_sha)).replace('__CONTROLLER_SHA__',repr(controller_sha))
ast.parse(script);(L/'remote364365-upload-CPU.py').write_bytes(script.encode())
print(json.dumps(dict(target_manifest_SHA256=target_sha,controller_manifest_SHA256=controller_sha,helper_AST_PASS=True,GPU=0)))
