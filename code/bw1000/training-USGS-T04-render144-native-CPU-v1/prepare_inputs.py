"""Private local adapter: train72 supervised; dev/blind72 never open labels."""
import hashlib,json,os,re,stat
from pathlib import Path,PurePosixPath
D=Path(__file__).resolve().parent
G=D.parents[2]
def need(ok,why):
    if not ok:raise ValueError(why)
def canonical(v):return (json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode()
def digest(b):return hashlib.sha256(b).hexdigest()
sha=digest
def pinned(path,pin,cap=2*1024**2,check=lambda:None):
    path=Path(path);check();need(type(pin['bytes']) is int and 0<pin['bytes']<=cap,'BOUNDED_PIN')
    if os.name=='nt':
        from windows_read_identity import read_stable
        raw,_=read_stable(path,pin['bytes'],check,lambda n:None)
    else:
        before=path.lstat();need(stat.S_ISREG(before.st_mode) and not path.is_symlink(),'REGULAR_SOURCE')
        key=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        with path.open('rb') as f:
            need(key(before)==key(os.fstat(f.fileno())),'OPEN_IDENTITY');raw=f.read(pin['bytes']+1)
            need(key(before)==key(os.fstat(f.fileno()))==key(path.lstat()),'STABLE_IDENTITY')
    check();need(len(raw)==pin['bytes'] and digest(raw)==pin['SHA256'],'EXACT_FILE_PIN');return raw
def verify_source(check=lambda:None):
    m=json.loads((D/'SOURCE_FILES.json').read_bytes())['files']
    need(set(m)=={p.name for p in D.iterdir() if p.is_file() and p.name!='SOURCE_FILES.json'},'EXACT_SOURCE_INVENTORY')
    for name,pin in m.items():
        need(Path(name).name==name,'SOURCE_NAME');pinned(D/name,pin,2*1024**2,check)

def source_manifest(check):
    plan=json.loads((D/'PLAN.json').read_bytes());v=json.loads((D/'INPUT_ONLY_MANIFEST.json').read_bytes())
    need(v['payload_files']==28 and len(v['payloads'])==28 and v['planned']==plan['planned'] and v['source_manifest']==plan['dataset_manifest'],'EXACT_INPUT_ONLY28_MANIFEST')
    need(all(x['read'] is x['upload'] is x['required_on_remote'] is False for x in v['omitted_heldout_gold'].values()) and len(v['omitted_heldout_gold'])==2,'HELDOUT_GOLD_NOT_REQUIRED')
    return plan,v

def stage(check):
    verify_source(check);plan,v=source_manifest(check)
    public=json.loads(pinned(G/plan['dataset_audit']['path'],plan['dataset_audit'],16384,check))
    need(public['dataset_artifact_preparation_passed'] is True and public['images']==24 and public['QA']==144 and public['QA_split_counts']==plan['planned'] and public['independent_math_rows']==144 and public['independent_PNG_byte_dimension_structure_nonblank_pass']==24,'ACTUAL_RENDER144_DATA_ARTIFACT_PASS')
    root=G/plan['dataset_root'];m=json.loads(pinned(root/'manifest-private.json',plan['dataset_manifest'],32768,check))
    need(m['actual_images']==24 and m['actual_QA']==144 and m['actual_split_QA']==plan['planned'] and len(m['outputs'])==30 and all(m['outputs'][n]==p for n,p in v['payloads'].items()),'ORIGINAL_MANIFEST_BINDING')
    # Only the explicit28 whitelist is opened. No inventory scan or heldout gold handle.
    data={n:pinned(root/n,p,1024**2,check) for n,p in v['payloads'].items()}
    need(len(data)==28 and sum(map(len,data.values()))==v['payload_bytes']==1031596,'EXACT28_RAW_PAYLOAD_BYTES')
    return data

def prepare_native(root,out,target_data,check):
    verify_source(check);plan,v=source_manifest(check);target=PurePosixPath(str(target_data))
    need(target==PurePosixPath(plan['namespace'])/'data' and target.is_absolute() and '..' not in target.parts,'NEW_PRIVATE_VIEW')
    need(not out.exists(),'REMOTE_ADAPTER_ONCE');out.mkdir(mode=0o700)
    staged=pinned(root/'input-only-manifest.json',dict(bytes=(D/'INPUT_ONLY_MANIFEST.json').stat().st_size,SHA256=digest((D/'INPUT_ONLY_MANIFEST.json').read_bytes())),32768,check)
    need(json.loads(staged)==v,'EXACT_STAGED_INPUT_ONLY_METADATA')
    protocol_pin=plan['protocol'];pinned(D/'render-contracts.reference.txt',protocol_pin,16384,check)
    trainraw=pinned(root/'train/labels-private.jsonl',plan['train_labels'],65536,check)
    labels=[json.loads(x) for x in trainraw.splitlines()];gold={r['id']:r for r in labels};need(len(labels)==len(gold)==72,'TRAIN72_LABEL_DENOMINATOR')
    media={};files={};ordered={};bindings={};allids=set();chars={r['id_SHA256']:r for r in plan['chart_binding']}
    def write(name,body):
        p=out/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as f:f.write(body);f.flush();os.fsync(f.fileno())
    kinds=['maximum_reported_number','minimum_reported_number','maximum_depth','minimum_depth','depth_change','reported_number_change']
    for split,count in plan['planned'].items():
        raw=pinned(root/split/'inputs.jsonl',plan['source_inputs'][split],65536,check)
        rows=[json.loads(x) for x in raw.splitlines()];need(len(rows)==count,'FULL_SPLIT_INPUTS')
        adapted=[];ordered[split]=[];bindings[split]=[];seen_charts={}
        for r in rows:
            check();need(set(r)=={'id','chart_id','group','kind','event_ids','images','messages','labels'} and r['labels'] is None and r['id'] not in allids,'RENDER144_SOURCE_SCHEMA_UNIQUE_ID')
            allids.add(r['id']);chartsha=digest(r['chart_id'].encode());groupsha=digest(r['group'].encode());c=chars[chartsha]
            need(c['split']==split and c['group_SHA256']==groupsha and r['images']==[c['media_name']] and r['event_ids']==['E%02d'%i for i in range(1,7)] and r['kind'] in kinds and r['id']==r['chart_id']+':q'+str(kinds.index(r['kind'])),'EXACT_CHART_GROUP_MEDIA_KIND')
            seen_charts[chartsha]=seen_charts.get(chartsha,0)+1
            need(len(r['messages'])==1 and set(r['messages'][0])=={'role','content'},'ONLY_USER_INPUT')
            user=r['messages'][0];need(user['role']=='user' and isinstance(user['content'],str) and user['content'].startswith('<image>\n') and user['content'].endswith(' Answer FINAL: followed by the answer.'),'ORIGINAL_RENDER_PROMPT_UNMODIFIED')
            messages=[dict(user)]
            if split=='train':
                label=gold[r['id']];need(label['data_status']=='QUALIFIED' and label['chart_id']==r['chart_id'] and label['group']==r['group'] and label['kind']==r['kind'] and label['event_ids']==r['event_ids'] and user['content']=='<image>\n'+label['question'],'TRAIN_EXACT_LABEL_JOIN')
                answer=label['answer'];pattern=r'-?(?:0|[1-9][0-9]*)\.[0-9]{2}' if label['answer_type']=='decimal2' else r'E0[1-6](?: E0[1-6])*'
                need(label['answer_type'] in {'decimal2','labels'} and isinstance(answer,str) and len(answer)<=64 and re.fullmatch(pattern,answer),'TRAIN_FROZEN_ANSWER_FORMAT')
                messages.append(dict(role='assistant',content='FINAL: '+answer))
            name=c['media_name'];adapted.append(dict(id=r['id'],messages=messages,images=[str(target/name)]));ordered[split].append(digest(r['id'].encode()))
            bindings[split].append(dict(source_id_SHA256=ordered[split][-1],source_row_SHA256=digest(canonical(r)),adapter_row_SHA256=digest(canonical(adapted[-1])),media_name=name,media_SHA256=plan['media'][name]['SHA256'],chart_SHA256=chartsha,group_SHA256=groupsha))
        need(len(seen_charts)==plan['charts_per_split'][split] and set(seen_charts.values())=={6},'ALL_CHARTS_SIX_QA_EACH')
        body=b''.join(canonical(r) for r in adapted);name=split+'-swift.jsonl';write(name,body);files[name]=dict(bytes=len(body),SHA256=digest(body),rows=count)
    need(len(allids)==144,'ALL144_UNIQUE_INPUTS')
    for name,pin in plan['media'].items():
        check();need(PurePosixPath(name).parts[0] in plan['planned'] and '..' not in PurePosixPath(name).parts,'EXACT_MEDIA_SCOPE')
        write(name,pinned(root/name,pin,1024**2,check));media[name]=pin
    binding=dict(passed=True,target_data=str(target),dataset_manifest_SHA256=plan['dataset_manifest']['SHA256'],dataset_audit_SHA256=plan['dataset_audit']['SHA256'],source_inputs=plan['source_inputs'],train_labels_pin=plan['train_labels'],protocol_SHA256=plan['protocol']['SHA256'],files=files,media=media,ordered_source_id_SHA256=ordered,row_bindings=bindings,planned=plan['planned'],train_labels_read=True,dev_gold_read=False,blind_gold_read=False,blind_inputs_and_media_read=True,model_predictions=0,scores=0,input_only_manifest_SHA256=digest(staged),staged_payload_files=28,staged_payload_bytes=v['payload_bytes'],omitted_gold_required_on_remote=False)
    body=canonical(binding);need(len(body)<=262144,'SMALL_BINDING');write('data-binding-private.json',body);return binding
