"""Read exact I03 namespaces,source/Native pins and complete input identities."""
from pathlib import Path
import base64,hashlib,json,subprocess,time
D=Path(__file__).resolve().parent;G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
commit=json.loads((D/'source-publication-receipt.json').read_bytes())['commit'];git=['git','-c','safe.directory='+G.as_posix(),'-C',str(G)]
task=json.loads(subprocess.check_output(git+['show',commit+':coordination/tasks/deep-i03-offoff-repeatability-v1.json']))
def blob(n):return subprocess.check_output(git+['show',commit+':'+task['source']+'/'+n])
inputs=blob('pilot_inputs.py');spec=json.loads(blob('input_binding.json'))
remote=('INPUTS='+repr(base64.b64encode(inputs).decode())+'\nSPEC='+repr(spec)+'\nCLAIM='+repr(task['claim'])+'\n').encode()+b'''
from pathlib import Path
import base64,datetime,hashlib,json,subprocess,types
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');label=CLAIM['label'];node=CLAIM['node'];parent=CLAIM['parent']
def query(args):
 r=subprocess.run(args,capture_output=True,text=True,timeout=15);assert r.returncode==0 and len(r.stdout)<1024**2;return r.stdout
row=query(['squeue','-h','-j',parent,'-o','%T|%D|%N']).strip().split('|');assert row[:2]==['RUNNING','40'] and len(row)==3
nodes=query(['scontrol','show','hostnames',row[2]]).splitlines();assert len(set(nodes))==40 and node in nodes
steps=query(['squeue','--steps','-h','-j',parent,'-w',node,'-o','%i|%j|%N']).splitlines()
assert all(s.split('|')[0] in (parent+'.batch',parent+'.extern') for s in steps)
old=query(['sacct','-n','-P','-j',parent,'--format=JobIDRaw,JobName%100,State%50,ExitCode,ElapsedRaw,NodeList%500'])
assert not [r for r in old.splitlines() if len(r.split('|'))>=6 and r.split('|')[1].strip() in (label,label+'-pre',label+'-post')]
assert not (R/'results'/(label+'-private')).exists() and not (R/'results'/(label+'-controller')).exists()
library=R/'candidates/Deep-C4-history-bank-alias-CPU-v1/artifacts/libDeep_c4_history_bank_alias_v1.so';receipt=library.with_name('CPU-build.json')
assert hashlib.sha256(library.read_bytes()).hexdigest()==CLAIM['candidate_Native_SHA256'] and hashlib.sha256(receipt.read_bytes()).hexdigest()==CLAIM['CPU_receipt_SHA256']
module=types.ModuleType('pilot_inputs');exec(compile(base64.b64decode(INPUTS,validate=True),'pinned_pilot_inputs','exec'),module.__dict__)
selected=module.resolve(R,SPEC,host_shared=Path('/public/home/aclsylqidf/shared'))
assert len(selected)==2 and selected[0][0]['prompt_tokens']==1242 and selected[0][0]['output_budget']==4096
print(json.dumps(dict(passed=True,observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent_state=row[0],parent_nodes=40,target_node=node,
 target_active_steps_excluding_parent=0,new_namespaces_absent=True,new_scope_accounting_absent=True,Native_file_SHA_checked=True,CPU_receipt_SHA_checked=True,
 original_input_binding_checked=True,original_case_inventory=256,selected_prompt_lengths=[v[0]['prompt_tokens'] for v in selected],
 actual_planned_requests=2,actual_planned_modes=[0,0],actual_planned_unique_cases=1,GPU_allocated=False,Native_dlopen=False,model_weights_read=False)))
'''
source=(D.parent/'i03-deep-c4-history-staging-v1/read_source_once.py').read_text(encoding='utf8');ns={'__file__':str(D/'reuse.py')};exec(compile(source.split('REMOTE=',1)[0],'existing_ordinary_cluster_transport','exec'),ns)
started=time.monotonic();r=subprocess.run(ns['ssh'],input=remote,capture_output=True,timeout=90);ended=time.monotonic()
for n,v in [('preflight.stdout.private.json',r.stdout),('preflight.stderr.private.txt',r.stderr)]:
 with (D/n).open('xb') as f:f.write(v)
result=dict(commit=commit,transport_rc=r.returncode,payload_SHA256=hashlib.sha256(remote).hexdigest(),started_monotonic=started,ended_monotonic=ended,RTT_seconds=ended-started,stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())
if r.returncode==0:result['value']=json.loads(r.stdout)
with (D/'preflight-result.json').open('x') as f:json.dump(result,f,indent=2)
v=result.get('value',{});print(json.dumps(dict(transport_rc=r.returncode,passed=v.get('passed'),parent_nodes=v.get('parent_nodes'),new_namespaces_absent=v.get('new_namespaces_absent'),GPU_allocated=False,model_weights_read=False)))
