"""New offline union/regrade contract; reuse archived945 bytes, no oldHTTP replay."""
import argparse,base64,copy,gzip,hashlib,importlib.util,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[5]);p.add_argument('--work-name',required=True);args=p.parse_args()
L=args.workspace.resolve();G=L/'InfiniLM-hygon-test'
S=G/'code/bw1000/Deep-broad-fewshot-fullpre-quality-r450'
path=S/'analysis-eight-shards/global_original.py'
spec=importlib.util.spec_from_file_location('root_global450',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
H=mod.H;C=mod.C;modules=mod.consumers()
blob=(L/'Deep450-CPU945-original-sealed-v1.json.gz').read_bytes()
assert H(blob)=='87ec623677735978f2f3370f4377044492673bef34f4d1e2906000e4fc754155'
archive=json.loads(gzip.decompress(blob))
def original(suffix):
    keys=[k for k in archive['files']if k.endswith(suffix)];assert len(keys)==1
    row=archive['files'][keys[0]];raw=base64.b64decode(row['base64'])
    assert len(raw)==row['bytes']and H(raw)==row['SHA256'];return raw
assert Path(args.work_name).name==args.work_name
work=L/args.work_name;work.mkdir(exist_ok=False)
files={}
def add(name,raw):
    p=work/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
    files[name]=dict(path=name,bytes=len(raw),SHA256=H(raw))
add('inputs/encoding.json',original('/encoding.json'))
add('inputs/requests-private.json',original('/requests-private.json'))
cases_path=next(k for k,v in archive['files'].items()if v['SHA256']==mod.INPUTS['inputs/cases.json'])
add('inputs/cases.json',base64.b64decode(archive['files'][cases_path]['base64']))
assets=json.loads((work/'inputs/requests-private.json').read_bytes());cases=json.loads((work/'inputs/cases.json').read_bytes())['cases']
oldclient=json.loads(original('/changed-client-recording/result.json'));entries=assets['logical_plan'][:2]
recording_owners=[]
for entry,row in zip(entries,oldclient['rows']):
    p=entry['prompt_tokens'];pre=[dict(kind='prefill',past=s,tokens=min(1024,p-s),total=min(s+1024,p))for s in range(0,p,1024)]
    f=len(pre)+1;mode=entry['mode'];t1=86
    counts=[dict(owner=d,created=86,ready=86,
        legacy=dict(created=0,ready=0,pre=86*f,eligible_bf16_t1=t1,completed=0,other_t=86*(f-1),f32_t1=0),
        fullpre=dict(observed=mode*86*f,eligible=mode*t1,selected=mode*t1,completed=mode*t1,prefill=mode*86*(f-1),f32_t1=0,layout=0,selected_exceptions=0,fallback_exceptions=0,active=0))for d in range(8)]
    wire=original('/changed-client-recording/%04d.wire'%entry['logical_index'])
    message,finish,usage=modules['wire_client_r290'].sse_message(wire)
    request_id=json.loads(wire.splitlines()[0][6:])['id']
    owner=dict(request_id=request_id,r450_logical_index=entry['logical_index'],r450_case_id=entry['case_id'],r450_mode=mode,r450_r366_fixed=0,
       normal_reset_before=True,normal_reset_after=True,r450_quiescent=True,r450_poisoned=False,actual_forward_completed=f,actual_forward_attempts=f,
       generated_token_ids=[9,1],max_new_tokens=4096,prompt_tokens=p,Native_continuation=pre+[dict(kind='decode_summary',calls=1,first_past=p,last_total=p+1)],
       r450_actual_owner_counts=counts,r450_binding_proof=dict(actual_HC_groups_per_owner=86,HC0_fixed=True,providers={'enable':'/refactor/candidates/Deep-HC-fullpre-Native-r377-cpu-v1/libDeep_HC_fullpre_optin_r377.so'}),
       native_finish_reason='eos_token',backend_token_monotonic=[1.,2.],admission=dict(history_render_audit=dict(qualified=True,prequeue_bound=True,Native_prompt_token_ids_SHA256=entry['prompt_IDs_SHA256'],pending_thinking_prefix_open=False),text_delivery='P1'),final_metadata=dict(buffered_P0_text=False))
    body=original('/changed-client-recording/%04d.request.json'%entry['logical_index'])
    assert mod.response_row(entry,cases[0],assets['requests'][0]['body'],row,owner,body,wire,modules)['correct']==(mode==0)
    add('j0/%04d.request.json'%entry['logical_index'],body);add('j0/%04d.wire'%entry['logical_index'],wire)
    recording_owners.append(owner)
deployment=S/'deployment-v1';shards=json.loads((deployment/'SHARDS.json').read_bytes())
add('deployment/SHARDS.json',(deployment/'SHARDS.json').read_bytes())
m=dict(provider_source_SHA256=mod.PROVIDER,GPU_GO=True,Model_GO=True,files_SHA256={'SHARDS.json':H((deployment/'SHARDS.json').read_bytes())},configs_SHA256={},plans_SHA256={})
for s in shards:
    raw=(deployment/s['config']).read_bytes();q=json.loads(raw);q['execute_authorized']=True;raw=(json.dumps(q,ensure_ascii=False,indent=2)+'\n').encode()
    add('deployment/'+s['config'],raw);m['configs_SHA256'][str(s['index'])]=H(raw)
    raw=(deployment/s['plan']).read_bytes();add('deployment/'+s['plan'],raw);m['plans_SHA256'][str(s['index'])]=H(raw)
raw=(json.dumps(m,indent=2)+'\n').encode();add('deployment/SOURCE_MANIFEST.json',raw);deployment_sha=H(raw)
bundle=work/'bundle.json'
def seal():bundle.write_bytes((json.dumps(dict(schema_version=1,diagnostic_recording_only=True,provider_source_SHA256=mod.PROVIDER,files=files),indent=2)+'\n').encode())
seal();empty=mod.analyze(bundle,deployment_sha)
assert empty['verified_requests']==0 and empty['missing_or_unverified_requests']==512 and empty['summary']['pooled']['arms']['0']['planned']==256
identity=dict(actual_step='910374.123456',run=shards[0]['run'],source_manifest_SHA256=mod.PROVIDER,provider_config_SHA256=m['configs_SHA256']['0'])
for name,q in [
 ('j0/watch.json',dict(shard=0,provider_source_SHA256=mod.PROVIDER,deployment_source_SHA256=deployment_sha,actual_GPU='910374.123456',actual_client='910374.123457',CPU_keeper='910407.123456',completed=False,passed=False)),
 ('j0/client.json',dict(oldclient,assigned_HTTP=64)),
 ('j0/owner.json',dict(identity=identity,Native_SHA256=mod.NATIVE,HC_mode=0,TP=8,PP=1,cache_rows=1,cache_capacity=32768,loaded_key_count=67569,requests=recording_owners)),
 ('j0/ready.json',dict(identity=identity,Model_READY=True,Native_internal_owner_count=8))]:add(name,(json.dumps(q,indent=2)+'\n').encode())
seal();partial=mod.analyze(bundle,deployment_sha)
assert partial['verified_requests']==2 and partial['missing_or_unverified_requests']==510
assert partial['summary']['pooled']['arms']['0']['correct']==1 and partial['summary']['pooled']['arms']['1']['correct']==0
assert partial['summary']['per_suite']['mmlu']['arms']['0']['planned']==128
assert not partial['normal_lifecycle_qualified']and not partial['actual_Model_quality_data_complete']and not partial['default_promotion']
checks=['Original945twoSSEwire regrade good+badparse retained','All8fixedplans union512 with0observed retains512missing','2observed keeps510missing and full256perarm','Diagnosticrecording cannot qualify Model/lifecycle/default']
def reject(name,fn):
    try:fn()
    except(AssertionError,KeyError,ValueError):checks.append(name);return
    raise AssertionError('Expected rejection:'+name)
entry=entries[0];row=oldclient['rows'][0];owner=recording_owners[0];wire=original('/changed-client-recording/0000.wire');body=original('/changed-client-recording/0000.request.json')
bad=copy.deepcopy(row);bad['wire_SHA256']='0'*64
reject('Wrongwirehash rejected',lambda:mod.response_row(entry,cases[0],assets['requests'][0]['body'],bad,owner,body,wire,modules))
bad=copy.deepcopy(row);bad['correct']=False
reject('Forgedclientscore rejected byoriginalwire',lambda:mod.response_row(entry,cases[0],assets['requests'][0]['body'],bad,owner,body,wire,modules))
bad=copy.deepcopy(owner);bad['r450_actual_owner_counts'][7]['fullpre']['active']=1
reject('Unquiescentowner rejected',lambda:mod.response_row(entry,cases[0],assets['requests'][0]['body'],row,bad,body,wire,modules))
bad=copy.deepcopy(row);bad['mode']=1
reject('Foreignmode rejected',lambda:mod.response_row(entry,cases[0],assets['requests'][0]['body'],bad,owner,body,wire,modules))
files['j0/owner.json']['path']='../outside.json';seal()
reject('Bundlepathescape rejected',lambda:mod.analyze(bundle,deployment_sha))
q=dict(passed=True,checks=checks,recording_only=True,old_HTTP_replayed=False,Native=False,GPU=0,actual_Model=False,
    source_SHA256=H(path.read_bytes()),original_CPU945_archive_SHA256=H(blob),new_contract_cases=len(checks),global_fixed_requests=512,observed_recording_responses=2)
(work/'result.json').write_bytes((json.dumps(q,indent=2)+'\n').encode());print(json.dumps(q))
