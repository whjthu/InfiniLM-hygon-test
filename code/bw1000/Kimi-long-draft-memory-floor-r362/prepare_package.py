from pathlib import Path
import ast,base64,hashlib,json

D=Path(__file__).resolve().parent;G=D.parents[2];L=G.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files=['memory_floor.py','worker.py','pressure.py','GPU_inside.sh','check_CPU.py','CPU_inside.sh','CPU_main.py','idle.py','controller.py']
for n in files:
    p=D/n
    if p.suffix=='.py':ast.parse(p.read_text())
    if p.suffix=='.sh':assert b'\r'not in p.read_bytes()
m=json.loads((D.parent/'Kimi-long-draft-worker-r334/SOURCE_MANIFEST.json').read_bytes())
m.update(state='PRIVATE_ONCE_PRESSURE_TRIM_SOURCE_CPU_CONTRACT_PENDING',GPU_GO=False,new_GPU=0,
         files_SHA256={n:sha(D/n)for n in files},
         qualification='CPU recording providers do not qualify strict loader, real long query math, allocator reclamation or acceptance; one new1GPUpressure claim separately published',
         change='GuardedCodec only: when original4GiB floor is low, one empty_cache+synchronize and same floor recheck. No append/query retry or cat/math changes.',
         outer_wrapper_SHA256=json.loads((D.parent/'Kimi-C1-long-draft-cost-r324/SOURCE_MANIFEST.json').read_bytes())['outer_wrapper_SHA256'])
(D/'SOURCE_MANIFEST.json').write_bytes((json.dumps(m,indent=2)+'\n').encode())
manifest=sha(D/'SOURCE_MANIFEST.json')
raw=json.loads((L/'Kimi362-runc-raw-read.json').read_bytes());watch=json.loads((L/'Kimi357-runc-fourth-observed.json').read_bytes());post=json.loads((L/'Kimi362-delayed-post.json').read_bytes())
draft=raw['files'][0];targets=[]
for f in raw['files'][1:5]:
    r=f['report'];requests=[]
    for x in r['pair_report']['requests']:
        requests.append({k:v for k,v in x.items()if k not in ['generated_ids','emitter_events','Native_calls','cycles']})
    targets.append(dict(path=f['path'],SHA256=f['SHA256'],bytes=f['bytes'],node=r['node'],completed=r['completed'],
        first_capture_validation=r['first_capture_validation'],requests=requests,
        all8_owned_workers_shutdown_completed=r['all8_owned_workers_shutdown_completed'],cleanup_errors=r['cleanup_errors'],
        error=r['error'],traceback=r['traceback']))
assert all(t['first_capture_validation']['passed']and t['all8_owned_workers_shutdown_completed']and not t['cleanup_errors']for t in targets)
assert all([(x['ids_SHA256'],x['tokens'],x['finish'],x['normal_reset'],x['quality'])for x in t['requests'][:2]]==[(x['ids_SHA256'],x['tokens'],x['finish'],x['normal_reset'],x['quality'])for x in targets[0]['requests'][:2]]for t in targets)
diag=dict(schema=1,state='ACTUAL_APPEND_PROMPT_MEMORY_FLOOR_FAIL_FIRST_CAPTURE_TWO_AR_VALID_NO_SPEC_QUERY',run='longpair-20261102-c',GPU_new=0,
    draft=dict(path=draft['path'],SHA256=draft['SHA256'],report=draft['report']),targets=targets,
    logs=[dict(path=f['path'],SHA256=f['SHA256'],bytes=f['bytes'],traceback=f['tail'][f['tail'].find('Traceback (most recent call last):'):])for f in raw['files']if 'draft' in f['path']and f.get('tail')and 'Traceback (most recent call last):'in f['tail']],
    accounting=raw['accounting'],
    proven=['Authenticated and ready;87BF16 tensors strict-loaded; no listener/hello or Runtime-constructor failure',
            '53successful512row appends=27136rows;54th append fence fails before publication; no seal/query/acceptance',
            'Original4GiB guard raised, not an allocator OutOfMemory error',
            'Five tap firstcapture finite/epoch/512rows/route passed on all4leaders; both AR59EOS/calculate(16195,39747)/normalreset',
            'All32 Native workers shutdown; all4 target report cleanup_errors empty'],
    inference=['Repeated growing torch.cat allocations can retain inactive allocator cache; peak reserved62,616,764,416B versus peak allocated14,528,446,976B supports pressure hypothesis but peaks are not contemporaneous'],
    unknown=['Exact free/allocated/reserved at failed fence was not recorded by334b; last-successful minimum is5,390,729,216B',
             'Physical GRES device ordinal/BDF not recorded; only visible_count1/logical cuda0 and actual node/step guaranteed by successfulRuntime assertion',
             'How much cache can actually be released and whether real32760query fits originalfloor remain unknown before newdraft-onlycard'],
    immediate_post_not_clean=[r for d in watch['watch']['draft_startup_diagnostics']if d.get('suffix')=='failure-post'for r in d['rows']],
    independent_delayed_post=post,delayed_all5_clean=all(r['passed']for g in post['groups']for r in g['reports']),
    cleanup_scope='Immediate546/657 port failures remain historical facts; delayed547/660clean is a separate time observation, not evidence allnormal protocol close',
    smallest_fix='Private single pressure-triggered inactive-cache release then original4GiB recheck; no guard reduction, requestretry, cat/math change, acceptance/Model claim')
out=G/'results/bw1000/2026-11-02';out.mkdir(parents=True,exist_ok=True)
(out/'Kimi357-runc-653-memory-floor-diagnosis-r362.json').write_bytes((json.dumps(diag,indent=2)+'\n').encode())
task=dict(owner='/root/kimi_operator_tuning',state='SOURCE_CPU_CONTRACT_PENDING',source=str(D.relative_to(G)).replace('\\','/'),source_manifest_SHA256=manifest,GPU_GO=False,GPU=0,
    CPU_claim=dict(parent=910407,node='a14r2n20',tasks=1,CPUs=4,memory_MiB=8192,hard_s=300,inner_s=260,Native_dlopen=False,GPU=0),
    next_GPU_candidate=dict(parent=910371,node='a02r3n15',cards=1,target_cards=0,GPU_max_s=600,CPU_keeper_parent=910407,CPU_keeper_node='a14r2n20',CPU_max_s=900,CPUs=8,mem_MiB=65536,
       prompt_tokens=32760,raw_dim=35840,BF16=True,periodic_seed324_fixture=True,phase_counts=dict(begin=1,append_prompt=64,seal=1,query=1,retire=1,close=1),
       query_positions=8,proposals=7,no_retry_reload=True,profile=False,synthetic_cost_only=True,actual_runtime='333+private334worker362+unchangedstrictloader',
       authorization='Root explicitly authorized one draft-only card after this source, CPUreceipt and concreteclaim ordinarypush; no32target authorization'),
    diagnosis='results/bw1000/2026-11-02/Kimi357-runc-653-memory-floor-diagnosis-r362.json')
(G/'coordination/tasks/Kimi-long-draft-memory-floor-r362.json').write_bytes((json.dumps(task,indent=2)+'\n').encode())
payload={n:base64.b64encode((D/n).read_bytes()).decode()for n in files+['SOURCE_MANIFEST.json']}
script="""from pathlib import Path
import base64,json,hashlib,subprocess,os
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'incoming/Kimi-long-draft-memory-floor-r362';assert not D.exists();D.mkdir()
for n,b in FILES.items():(D/n).write_bytes(base64.b64decode(b))
assert hashlib.sha256((D/'SOURCE_MANIFEST.json').read_bytes()).hexdigest()==MANIFEST
args=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c4','--gres=none','--mem=8192M','-w','a14r2n20','-t','00:05:00','--job-name=Kimi-r362-whole-CPU','--output='+str(R/'logs/Kimi-r362-whole-CPU-%J.log'),'--export=ALL','bash',str(R/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
env=dict(os.environ,WAVE_INSIDE_R2323=D.name+'/CPU_inside.sh')
with(R/'logs/Kimi-r362-whole-CPU-client.log').open('x')as f:p=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'GPU':0,'CPU_submitted':True,'PID':p.pid,'label':'Kimi-r362-whole-CPU','source_manifest_SHA256':MANIFEST}))
""".replace('FILES',repr(payload)).replace('MANIFEST',repr(manifest))
(L/'remote362-upload-CPU.py').write_bytes(script.encode())
print(json.dumps(dict(manifest_SHA256=manifest,CPU_helper='remote362-upload-CPU.py',GPU=0)))
