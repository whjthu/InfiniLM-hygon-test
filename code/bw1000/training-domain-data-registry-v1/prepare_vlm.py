"""One common split-safe dataset adapter for the five native VLM recipes; no training loop."""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath

CASES={'T01.sft':4096,'T02.sft':512,'T03.sft':4096,'T04.sft':4096,'T05.sft':4096}
def need(ok,why):
    if not ok:raise ValueError(why)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()+b'\n'
def stable(path,pin,cap=2*1024**2):
    before=path.lstat();need(path.is_file() and not path.is_symlink() and 0<before.st_size==pin['bytes']<=cap,'bounded exact local data member')
    raw=path.read_bytes();after=path.lstat()
    need(all(getattr(before,k)==getattr(after,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')) and sha(raw)==pin['SHA256'],'data member changed')
    return raw
def relative(name,split):
    p=PurePosixPath(name);need(not p.is_absolute() and '..' not in p.parts and p.as_posix()==name and p.parts[:2]==(split,'media'),'same-split exact relative media path required')
    return p
def rows(raw):return [json.loads(x) for x in raw.splitlines()]

def join(train_inputs,train_labels,dev_inputs):
    """Pure ordered join; never accepts dev/blind labels, no shuffle or truncation."""
    lookup={r['id']:r for r in train_labels};need(len(lookup)==len(train_labels)==len(train_inputs),'unique complete train label denominator')
    output=[];seen=set()
    for x in train_inputs:
        need(x['id'] not in seen and x['id'] in lookup,'duplicate/missing training ID');seen.add(x['id']);q=lookup[x['id']]
        need(x['chart_id']==q['chart_id'] and x['group']==q['group'] and x['kind']==q['kind'],'cross-source training label join')
        need(x['labels'] is None and len(x['messages'])==1 and x['messages'][0]['role']=='user','training source must be answer-free user input')
        need(q['data_status']=='QUALIFIED' and isinstance(q['answer'],str) and q['answer'] and not q['answer'].startswith('FINAL:'),'qualified unprefixed exact label required')
        r=dict(x);r.pop('labels');r['messages']=[dict(x['messages'][0]),dict(role='assistant',content='FINAL: '+q['answer'])];output.append(r)
    need(len({x['id'] for x in dev_inputs})==len(dev_inputs) and not seen&{x['id'] for x in dev_inputs},'train/dev ID overlap')
    dev=[]
    for x in dev_inputs:
        need(x['labels'] is None and len(x['messages'])==1 and x['messages'][0]['role']=='user','dev must remain user-only')
        r=dict(x);r.pop('labels');dev.append(r)
    need(not {x['group'] for x in train_inputs}&{x['group'] for x in dev_inputs},'train/dev closure group overlap')
    return output,dev

def prepare(a):
    need(a.case in CASES,'registered native VLM case required')
    target=PurePosixPath(a.target_data);need(target.is_absolute() and '..' not in target.parts and str(target)==a.target_data and '\\' not in a.target_data,'explicit canonical Linux target namespace, not Windows Path.is_absolute')
    manifest_path=a.dataset/'manifest-private.json';raw=stable(manifest_path,dict(bytes=a.manifest_bytes,SHA256=a.manifest_sha))
    m=json.loads(raw);need(m['Software']=='training-USGS-multievent-render144-v2' and m['actual_images']==24 and m['actual_QA']==144,'only actual layout-v2 dataset; never old visually failed v1')
    need(m['actual_split_QA']=={'train':72,'dev':36,'blind':36},'frozen72/36/36 counts')
    names=['train/inputs.jsonl','train/labels-private.jsonl','dev/inputs.jsonl']
    data={n:rows(stable(a.dataset/n,m['outputs'][n])) for n in names}
    need([len(data[n]) for n in names]==[72,72,36],'actual complete unique-row counts')
    train,dev=join(data[names[0]],data[names[1]],data[names[2]])
    media={}
    for split,items in [('train',train),('dev',dev)]:
        for r in items:
            need(len(r['images'])==1,'one complete image per actual source input')
            name=r['images'][0];relative(name,split);pin=m['outputs'][name]
            if name not in media:
                stable(a.dataset/name,pin,1048576)
                media[name]=dict(pin,source_path=str((a.dataset/name).resolve()),target_path=str(target/name),target_presence_verified=False)
            r['images']=[str(target/name)]
    need(len(media)==18,'exact12train+6dev media, blind never staged')
    a.output.mkdir(mode=0o700);files={}
    for name,items in [('train-swift.jsonl',train),('dev-swift.jsonl',dev)]:
        body=b''.join(canonical(r) for r in items)
        with (a.output/name).open('xb') as f:f.write(body)
        files[name]=dict(bytes=len(body),SHA256=sha(body),rows=len(items))
    binding=dict(schema='FIVE_NATIVE_VLM_DOMAIN_INPUT_VIEW_V1',case=a.case,dataset_manifest=dict(bytes=len(raw),SHA256=sha(raw)),
        source_files={n:m['outputs'][n] for n in names},files=files,media=media,target_data=str(target),
        training_messages='user + assistant FINAL',dev_messages='user only',source_order_unchanged=True,literal_special_tokens_added=False,
        native_EOS_handling='original template only, actual per-case verification pending',max_sequence_length=CASES[a.case],truncation_performed=False,
        actual_encode_calls=0,actual_target_copy_calls=0,target_presence_verified=False,train_effective_supervised_tokens='UNKNOWN_UNTIL_REAL_NATIVE_ENCODER',
        dev_gold_read=False,blind_inputs_gold_or_media_read=False,training_GO=False,GPU_GO=False,Model_GO=False)
    with (a.output/'data-binding-private.json').open('xb') as f:f.write(canonical(binding))
    return {k:binding[k] for k in ('case','files','target_presence_verified','actual_encode_calls','actual_target_copy_calls','dev_gold_read','blind_inputs_gold_or_media_read','training_GO')}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',choices=sorted(CASES),required=True);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--manifest-sha',required=True);p.add_argument('--manifest-bytes',type=int,required=True);p.add_argument('--target-data',required=True);p.add_argument('--output',type=Path,required=True)
    print(json.dumps(prepare(p.parse_args()),sort_keys=True))
