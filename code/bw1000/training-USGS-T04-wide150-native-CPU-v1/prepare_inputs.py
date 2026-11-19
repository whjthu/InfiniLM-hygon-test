"""Private local adapter: train50 supervised; dev/blind100 never open labels."""
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
def prepare(out,target_data,check):
    verify_source(check);plan=json.loads((D/'PLAN.json').read_bytes())
    target=PurePosixPath(target_data.as_posix() if hasattr(target_data,'as_posix') else target_data)
    need(target==PurePosixPath(plan['namespace'])/'data' and target.is_absolute() and '..' not in target.parts,'NEW_PRIVATE_VIEW')
    need(not out.exists(),'LOCAL_ADAPTER_ONCE');out.mkdir(mode=0o700)
    public=json.loads(pinned(G/plan['dataset_audit']['path'],plan['dataset_audit'],16384,check))
    need(public['data_integrity_qualified'] is True and public['verified']['verified_QA']==150 and public['verified']['verified_charts']==30,'ACTUAL150_DATA_INTEGRITY')
    root=G/plan['dataset_root'];manifest=json.loads(pinned(root/'manifest-private.json',plan['dataset_manifest'],32768,check))
    need(manifest['media']==plan['media'] and manifest['actual_inputs']==150,'ORIGINAL_MANIFEST_BINDING')
    protocol=json.loads(pinned(G/plan['protocol']['path'],plan['protocol'],16384,check))
    # Only the train labels path is constructed/read. No dev/blind label handle exists.
    trainraw=pinned(root/'train/labels-private.jsonl',plan['train_labels'],65536,check)
    labels=[json.loads(x) for x in trainraw.splitlines()];gold={r['id']:r for r in labels}
    need(len(labels)==len(gold)==50,'TRAIN50_LABEL_DENOMINATOR')
    media={};files={};ordered={};bindings={};allids=set();chars={r['id']:r for r in plan['chart_binding']}
    def write(name,body):
        p=out/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as f:f.write(body);f.flush();os.fsync(f.fileno())
    for split in ('train','dev','blind'):
        raw=pinned(root/split/'inputs.jsonl',plan['source_inputs'][split],65536,check)
        rows=[json.loads(x) for x in raw.splitlines()];need(len(rows)==50,'ALL50_SPLIT_INPUTS')
        adapted=[];ordered[split]=[];bindings[split]=[];seen_charts={}
        for r in rows:
            check();need(set(r)=={'id','chart_group','source_group','messages','images'} and r['id'] not in allids,'SOURCE_SCHEMA_UNIQUE_ID')
            allids.add(r['id']);c=chars[r['chart_group']]
            need(c['split']==split and c['group']==r['source_group'] and r['images']==[c['media_name']],'EXACT_CHART_GROUP_MEDIA')
            seen_charts[c['id']]=seen_charts.get(c['id'],0)+1
            need(len(r['messages'])==1 and set(r['messages'][0])=={'role','content'},'ONLY_USER_INPUT')
            user=r['messages'][0];need(user['role']=='user' and isinstance(user['content'],str) and user['content'].startswith('<image>\n') and user['content'].endswith(protocol['prompt_suffix']),'FROZEN_PROMPT_SUFFIX')
            messages=[dict(user)]
            if split=='train':
                label=gold[r['id']];need(label['data_status']=='QUALIFIED' and label['split']=='train' and label['chart_id']==r['chart_group'] and label['group']==r['source_group'] and user['content']=='<image>\n'+label['question']+protocol['prompt_suffix'],'TRAIN_EXACT_LABEL_JOIN')
                answer=label['answer'];pattern=r'-?(?:0|[1-9][0-9]*)\.[0-9]{2}' if label['answer_type']=='decimal2' else r'E[0-9]{2}(?: E[0-9]{2})*'
                need(label['answer_type'] in {'decimal2','labels'} and isinstance(answer,str) and len(answer)<=64 and re.fullmatch(pattern,answer),'TRAIN_FROZEN_ANSWER_FORMAT')
                messages.append(dict(role='assistant',content='FINAL: '+answer))
            name=c['media_name'];adapted.append(dict(id=r['id'],messages=messages,images=[str(target/name)]))
            ordered[split].append(digest(r['id'].encode()))
            bindings[split].append(dict(source_id_SHA256=ordered[split][-1],source_row_SHA256=digest(canonical(r)),adapter_row_SHA256=digest(canonical(adapted[-1])),media_name=name,media_SHA256=plan['media'][name]['SHA256'],chart_SHA256=digest(c['id'].encode()),group_SHA256=digest(c['group'].encode())))
        need(len(seen_charts)==10 and set(seen_charts.values())=={5},'TEN_CHARTS_FIVE_QA_EACH')
        body=b''.join(canonical(r) for r in adapted);name=split+'-swift.jsonl';write(name,body);files[name]=dict(bytes=len(body),SHA256=digest(body),rows=50)
    for name,pin in plan['media'].items():
        check();need(PurePosixPath(name).parts[0] in plan['planned'] and '..' not in PurePosixPath(name).parts,'EXACT_MEDIA_SCOPE')
        write(name,pinned(root/name,pin,1024**2,check));media[name]=pin
    binding=dict(passed=True,target_data=str(target),dataset_manifest_SHA256=plan['dataset_manifest']['SHA256'],dataset_audit_SHA256=plan['dataset_audit']['SHA256'],source_inputs=plan['source_inputs'],train_labels_pin=plan['train_labels'],protocol_SHA256=plan['protocol']['SHA256'],files=files,media=media,ordered_source_id_SHA256=ordered,row_bindings=bindings,planned=plan['planned'],train_labels_read=True,dev_gold_read=False,blind_gold_read=False,blind_inputs_and_media_read=True,model_predictions=0,scores=0)
    body=canonical(binding);need(len(body)<=262144,'SMALL_BINDING');write('data-binding-private.json',body)
    return binding
