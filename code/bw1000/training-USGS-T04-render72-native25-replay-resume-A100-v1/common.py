"""Small bounded identity contracts; preparation imports stdlib only."""
import hashlib,json,os,stat,time
from pathlib import Path
D=Path(__file__).resolve().parent
P=json.loads((D/'PLAN.json').read_bytes())
def need(v,w):
    if not v:raise ValueError(w)
def canonical(v):return (json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def sha(b):return hashlib.sha256(b).hexdigest()
def stamp(s):return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def left(c,end=790):
    v=c['epoch']+end-time.time();need(v>0 and c['deadline']==c['epoch']+900,'ONE_ORIGINAL900');return v
def pinned(path,pin,cap=2*1024**2):
    p=Path(path)
    if os.name=='nt':
        from windows_read_identity import read_stable
        need(0<pin['bytes']<=cap,'BOUNDED_WINDOWS_PIN');b,_=read_stable(p,pin['bytes'],lambda:None,lambda n:None);need(sha(b)==pin['SHA256'],'EXACT_WINDOWS_PIN');return b
    a=p.lstat();need(stat.S_ISREG(a.st_mode) and not p.is_symlink() and a.st_size==pin['bytes']<=cap,'BOUNDED_REGULAR_PIN')
    with p.open('rb') as f:
        need(stamp(a)==stamp(os.fstat(f.fileno())),'OPEN_STABLE');b=f.read(pin['bytes']+1);need(stamp(a)==stamp(os.fstat(f.fileno()))==stamp(p.lstat()),'READ_STABLE')
    need(len(b)==pin['bytes'] and sha(b)==pin['SHA256'],'EXACT_PIN');return b
def obj(path,pin=None):return json.loads(pinned(path,pin) if pin else Path(path).read_bytes())
def save(path,v,replace=False):
    p=Path(path);b=canonical(v);need(len(b)<=2*1024**2,'SMALL_METADATA');t=p.with_name(p.name+'.tmp') if replace else p
    with t.open('wb' if replace else 'xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    if replace:t.replace(p)
    return dict(bytes=len(b),SHA256=sha(b))
def own():
    f=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split();return dict(PID=os.getpid(),PGID=os.getpgrp(),startticks=int(f[19]),uid=os.getuid(),host=os.uname().nodename,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
def verify_source():
    m=obj(D/'SOURCE_FILES.json')['files'];need(set(m)=={p.name for p in D.iterdir() if p.is_file() and p.name!='SOURCE_FILES.json'},'EXACT_NEW_SOURCE_MEMBERS')
    for n,p in m.items():pinned(D/n,p)
    return m
def native_inputs(c):
    q=obj(D/'native144-public.json');need(q['producer_passed'] is q['independent_metadata_verified'] is True and q['native']['completed']==dict(train=72,dev=36,blind=36),'ACTUAL_NATIVE144_PASS')
    b=obj(P['databinding']['path'],P['databinding']);r=obj(P['native_identities']['path'],P['native_identities']);need(b['planned']==dict(train=72,dev=36,blind=36) and b['dev_gold_read'] is b['blind_gold_read'] is False,'NO_EVAL_GOLD')
    rr=[x for x in r if x['split']=='train'];need(len(rr)==72 and [x['ordinal'] for x in rr]==list(range(72)),'ALL72_NATIVE_ROWS')
    # Only previously staged TRAIN swift and TRAIN media are opened. No gold or heldout input handle.
    root=Path(b['target_data']);raw=pinned(root/'train-swift.jsonl',b['files']['train-swift.jsonl']);rows=[json.loads(x) for x in raw.splitlines()];need(len(rows)==72,'TRAIN72')
    for i,(x,y) in enumerate(zip(rows,rr)):
        need(set(x)=={'id','messages','images'} and sha(x['id'].encode())==y['source_id_SHA256'] and sha(canonical(x))==y['adapter_row_SHA256'],'NATIVE_ORDERED_ROW')
        need(len(x['messages'])==2 and x['messages'][0]['role']=='user' and x['messages'][1]['role']=='assistant' and x['messages'][1]['content'].startswith('FINAL: ') and len(x['images'])==1,'TRAIN_ONLY_SUPERVISION')
        need(y['labels_None'] is False and y['effective_supervised_tokens']>0 and y['supervised_EOS_count']>0,'NATIVE_NONEMPTY_MASK_EOS')
        name=b['row_bindings']['train'][i]['media_name'];need(name.startswith('train/media/') and x['images']==[str(root/name)],'TRAIN_IMAGE_SCOPE');pinned(root/name,b['media'][name],1024**2)
    return b,rr,root/'train-swift.jsonl'
