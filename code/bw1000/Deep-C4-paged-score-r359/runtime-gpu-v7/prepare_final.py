"""Append only the launch/manifest to the already frozen CPU725 package."""
from pathlib import Path
import ast,base64,hashlib,json

S=Path(__file__).resolve().parent;G=S.parents[3];L=G.parent;O=G/'results/bw1000/2026-11-02'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
original=json.loads((S/'CPU_SOURCE_MANIFEST.json').read_bytes())
assert sha(S/'CPU_SOURCE_MANIFEST.json')=='6080d83b99e982009795e089f5dda518461e7eacd493caea581286dbc9ba34ac'
assert all(sha(S/n)==h for n,h in original['files_SHA256'].items())
formal=json.loads((L/'Deep359-consumer-CPU7-formal.json').read_bytes());cpu=formal['receipt']
raw=(json.dumps(cpu,indent=2)+'\n').encode();assert hashlib.sha256(raw).hexdigest()==formal['receipt_SHA256']
(O/'Deep359-CPU725-consumer-sealed.json').write_bytes(raw)
(O/'Deep359-CPU725-consumer-formal-PASS.json').write_bytes((json.dumps(formal,indent=2)+'\n').encode())
files=dict(original['files_SHA256']);files.update({n:sha(S/n)for n in ['CPU_SOURCE_MANIFEST.json','launch_once.py','check_launch_CPU.py']})
for n in files:
    if n.endswith('.py'):ast.parse((S/n).read_bytes())
    if n.endswith('.sh'):assert b'\r'not in(S/n).read_bytes()
m=dict(schema=1,GPU_GO=True,source_SHA256=files,
    CPU_source_manifest_SHA256='6080d83b99e982009795e089f5dda518461e7eacd493caea581286dbc9ba34ac',
    CPU_receipt='/refactor/candidates/Deep-C4-paged-score-r359-runtime-link-v5/CPU-link.json',
    CPU_receipt_SHA256='4878209aa64f85b92a96bc8cb5cad606a1a814ab9502677d580e3953bee96f5d',
    DSO_SHA256='a4e6208eef89bd63045c01bcab5ccc358e6a36cd6ea034aa286b2934ba82d91b',
    CPU_consumer_receipt='/refactor/candidates/Deep-C4-paged-score-r359-runtime-gpu-v7-CPU/CPU-consumers.json',
    CPU_consumer_receipt_SHA256='7909748755b5579dc080b634ce0633f76c857849e570042bfe78307383f5feef',
    owner='/root/kimi_operator_tuning',owner_handoff='Root explicitly delegated359finalcomponent;root stops editing359 until receipt; original7files/driver/controller/C++/HIP/DSO unchanged',
    exact_scope=dict(parent=910371,node='a02r3n16',GPU_cards=8,TP=8,processes=1,owner_threads=8,CPU_cores=32,memory_MiB=64000,GPU_max_s=600,
        CPU_keeper_parent=910407,CPU_keeper_node='a14r2n20',CPU_keeper_cores=1,CPU_keeper_memory_MiB=4096,CPU_keeper_max_s=900,
        pasts=[8496,8575],trials_total=52,per_past=dict(QA=3,warm=2,measurement=20,restore=1),ABBA_blocks_per_past=5,prefix_reset_reseed_chunk=1024,
        T=1,heads=8,dim=128,TopK=512,hidden_BF16=[1,4096],query_lora_BF16=[1,1024],
        input='Syntheticseed3598496 hidden/q_lora; actual layer16indexercheckpoint; notModelactivations or traffic',
        no_retry=True,no_new_kernel=True,Model_loaded=False,Model_gain='UNKNOWN'),
    validation='Original baseline CPUscore/FP8/TopK first, candidatefinite/canary/committedstate/nextstep; actual8PCI/gfx936 and32CPU→8disjoint4cores;8normalRuntime/CCLclose then independentKFD/process checks',
    timing='CompleteC4prepare+commit CAPI wall; H2D/clone/canary/reset/reseed excluded there but separately reported and checked enclosure measured; no wholeModel/peakmemory claim')
(S/'GPU_MANIFEST.json').write_bytes((json.dumps(m,indent=2)+'\n').encode());digest=sha(S/'GPU_MANIFEST.json')
task=G/'coordination/tasks/Deep-C4-paged-score-r359.json';t=json.loads(task.read_bytes())
t.update(owner='/root/kimi_operator_tuning',state='CPU725_PASS_FINAL_ONCE_EIGHT_CARD_COMPONENT_CLAIM',GPU_GO=True,
    GPU_authorized_in_this_initial_card=False,new_GPU_submitted=False,
    handoff_from='bw1000-root explicitdelegate after725; unchanged7source/driver/controller/CPP/HIP/DSO reused',
    GPU_driver_source='code/bw1000/Deep-C4-paged-score-r359/runtime-gpu-v7',GPU_driver_manifest_SHA256=digest,
    CPU725_result='results/bw1000/2026-11-02/Deep359-CPU725-consumer-formal-PASS.json',CPU725_receipt_SHA256=formal['receipt_SHA256'],
    concrete_once_GPU_claim=m['exact_scope'],source_driver_SHA256=files['GPU_driver.py'],remaining='After this exactsource/CPU/claimordinarypush: controllerfreshmembership/dedup/PID/KFD thenone8GPU52trial; noModel')
task.write_bytes((json.dumps(t,indent=2)+'\n').encode())
payload={n:base64.b64encode((S/n).read_bytes()).decode()for n in ['launch_once.py','check_launch_CPU.py','GPU_MANIFEST.json']}
script="""from pathlib import Path
import base64,json,hashlib
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');S=R/'incoming/Deep-C4-paged-score-r359-runtime-gpu-v7'
old=json.loads((S/'CPU_SOURCE_MANIFEST.json').read_bytes());assert hashlib.sha256((S/'CPU_SOURCE_MANIFEST.json').read_bytes()).hexdigest()=='6080d83b99e982009795e089f5dda518461e7eacd493caea581286dbc9ba34ac'
assert all(hashlib.sha256((S/n).read_bytes()).hexdigest()==h for n,h in old['files_SHA256'].items())
for n,b in __PAYLOAD__.items():
 p=S/n;assert not p.exists()
 with p.open('xb')as f:f.write(base64.b64decode(b))
assert hashlib.sha256((S/'GPU_MANIFEST.json').read_bytes()).hexdigest()==__MANIFEST__
m=json.loads((S/'GPU_MANIFEST.json').read_bytes());assert all(hashlib.sha256((S/n).read_bytes()).hexdigest()==h for n,h in m['source_SHA256'].items())
print(json.dumps({'GPU':0,'uploaded_new_launch_manifest_only':True,'manifest_SHA256':__MANIFEST__,'original7unchanged':True}))
""".replace('__PAYLOAD__',repr(payload)).replace('__MANIFEST__',repr(digest))
ast.parse(script);(L/'remote359-upload-final.py').write_bytes(script.encode())
print(json.dumps(dict(GPU_manifest_SHA256=digest,helper_AST_PASS=True,driver_SHA256=files['GPU_driver.py'],original7unchanged=True)))
