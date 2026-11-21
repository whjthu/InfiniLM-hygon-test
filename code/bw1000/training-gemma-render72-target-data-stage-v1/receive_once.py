"""Bounded exclusive installation of exactly the already frozen T02 training data."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import signal
import stat
import struct
import sys
import time

D=Path(__file__).resolve().parent
TARGET='/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-gemma-usgs-render72-native-CPU-v1'
REQUEST='0f9b465a7c60419c8350d44e8f2f7dfd'
MAGIC=b'GEMMA72DATA1\n'
MAX_BODY=4*1024*1024


def need(ok,why):
    if not ok:raise ValueError(why)


def pin(raw):return dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())


def json_file(path,obj):
    raw=(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return pin(raw)


def exact(stream,count,deadline):
    out=[];read=0
    while read<count:
        need(time.time()<deadline,'original staging epoch exhausted')
        part=stream.read(min(65536,count-read))
        need(part,'truncated payload')
        out.append(part);read+=len(part)
    return b''.join(out)


def header(stream,inputs,deadline):
    need(exact(stream,len(MAGIC),deadline)==MAGIC,'payload magic differs')
    length=struct.unpack('<I',exact(stream,4,deadline))[0]
    need(0<length<=16384,'header exceeds cap')
    obj=json.loads(exact(stream,length,deadline))
    expected={'request_id':REQUEST,'target':TARGET,'files':[
        {'name':r['target_relative'],'bytes':r['bytes'],'SHA256':r['SHA256']} for r in inputs['files']]}
    need(obj==expected,'payload roster differs from frozen exact sixteen inputs')
    need(len(obj['files'])==16 and sum(x['bytes'] for x in obj['files'])<=MAX_BODY,'payload count/size differs')
    names=set()
    for item in obj['files']:
        p=PurePosixPath(item['name'])
        need(not p.is_absolute() and '..' not in p.parts and p.as_posix()==item['name']
             and item['name'] not in names,'payload path escapes or repeats')
        names.add(item['name'])
    return obj


def install(stream,root,inputs,epoch,*,local_test=False):
    deadline=epoch+120
    need(type(epoch) in (float,int) and math.isfinite(epoch) and 0<epoch<=time.time()<deadline,'bad origin epoch')
    need(local_test or str(root)==TARGET,'target root differs')
    need(not root.exists() and not root.is_symlink(),'existing stage requires reconciliation; no overwrite')
    h=header(stream,inputs,deadline)
    root.mkdir(mode=0o700,parents=False)
    result={'schema':'GEMMA_RENDER72_TRAIN_DATA_STAGE_V1','request_id':REQUEST,'epoch':epoch,'deadline':deadline,
        'planned_files':16,'completed_files':0,'actual_bytes_written':0,'passed':False,'local_test':local_test,
        'actual_target_installation':not local_test,'train_rows':72,'view_rows':800,'train_media':12,
        'dev_or_blind_file_bodies':0,'model_or_initial_files':0,'native_encodes':0,'optimizer_updates':0,
        'GPU':0,'Slurm':0,'READY':False,'training_GO':False,'GPU_GO':False}
    json_file(root/'stage-intent.json',{'request_id':REQUEST,'epoch':epoch,'deadline':deadline,
        'files':h['files'],'local_test':local_test,'replay_allowed':False})
    installed={}
    try:
        for item in h['files']:
            raw=exact(stream,item['bytes'],deadline)
            need(pin(raw)=={k:item[k] for k in ('bytes','SHA256')},'payload file differs')
            dest=root/item['name'];dest.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
            with dest.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
            installed[item['name']]=pin(raw)
            result['completed_files']+=1;result['actual_bytes_written']+=len(raw)
        need(time.time()<deadline and not stream.read(1),'extra bytes or exhausted original epoch')
        for name,wanted in installed.items():
            dest=root/name
            need(dest.resolve()==dest.absolute() and stat.S_ISREG(dest.lstat().st_mode),'installed file aliases')
            with dest.open('rb') as f:actual=pin(f.read(wanted['bytes']+1))
            need(actual==wanted,'installed file verification differs')
        binding=json.loads((root/'data/data-binding-private.json').read_bytes())
        need(binding['target_data']==TARGET+'/data' and binding['case']=='T02.sft'
             and binding['max_sequence_length']==512,'frozen native data binding differs')
        result.update(passed=True,state='LOCAL_ROUNDTRIP_ONLY' if local_test else 'GEMMA72_TRAIN_DATA_STAGED',
                      installed=installed,implicit_GPU_activity='UNKNOWN_NOT_PROBED')
    except Exception as error:
        result.update(state='FAILED_PARTIAL_REQUIRES_RECONCILIATION',error_type=type(error).__name__,reason=str(error)[:256])
    json_file(root/'stage-outcome.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--epoch',type=float,required=True);p.add_argument('--root-go',action='store_true')
    a=p.parse_args()
    source=json.loads((D/'SOURCE_FILES.json').read_bytes())['files']
    for name,wanted in source.items():
        need(Path(name).name==name and pin((D/name).read_bytes())==wanted,'published staging source differs')
    claim=json.loads((D/'CLAIM.json').read_bytes())
    need(a.root_go and claim['root_GO'] is True and claim['data_stage_GO'] is True,'root stage GO is false')
    need(os.name=='posix' and math.isfinite(a.epoch) and 0<a.epoch<=time.time()<a.epoch+120,'bad HOST/epoch')
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('original staging deadline')))
    signal.alarm(max(1,math.ceil(a.epoch+120-time.time())))
    allowed=sorted(os.sched_getaffinity(0));need(len(allowed)>=2,'two CPUs unavailable');os.sched_setaffinity(0,allowed[:2])
    os.umask(0o077)
    inputs=json.loads((D/'INPUTS.json').read_bytes())
    r=install(sys.stdin.buffer,Path(TARGET),inputs,a.epoch)
    print(json.dumps({k:r[k] for k in ('state','passed','request_id','planned_files','completed_files','actual_bytes_written','READY')},sort_keys=True))
    return 0 if r['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
