"""Independently hash all ten final original files after the sole tail assembly."""
import json,subprocess
from pathlib import Path
BASE=Path(__file__).resolve().parent
pins=json.loads((BASE/'MODEL_FILES.json').read_bytes())['files'];intent=json.loads((BASE/'launch-intent.json').read_bytes())
ns={'__file__':str(Path(__file__).resolve())};prelude=(BASE.parent/'i03-deep-c4-history-staging-v1/read_source_once.py').read_text(encoding='utf8')
exec(compile(prelude.split('REMOTE=',1)[0],'existing_cluster_ssh','exec'),ns)
code=r'''
import datetime,hashlib,json,os,signal,socket,time
from pathlib import Path
root=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-minicpm-original-model-v1');repair=root/'tail-repair-v2'
raw=(repair/'receipt.json').read_bytes();receipt=json.loads(raw)
assert receipt['claim_commit']==INTENT['claim_commit'] and receipt['state']=='RETAINED_BYTES_AND_TAIL_ASSEMBLED_FULL_SHA_VERIFIED'
assert receipt['epoch']==INTENT['epoch'] and receipt['deadline']==INTENT['deadline'] and receipt['deadline']-receipt['epoch']==10800
assert receipt['retained_bytes']==2600776115 and receipt['tail_bytes']==181413 and receipt['files']==PINS
assert hashlib.sha256((root/'failure.json').read_bytes()).hexdigest()=='bc3442cd990d713dfd62951eb41cba50509c480e2f082ce80d647795cbf3d3c3'
assert hashlib.sha256((root/'resume-v1-failure.json').read_bytes()).hexdigest()=='4129c4c9cf275efcadf92b290b7e6d5fff3d9b03bddca64e7a5930197d3d15e3'
with (repair/'independent-reader-intent.json').open('x') as f:json.dump({'epoch':time.time(),'seconds':600,'host':socket.gethostname(),'PID':os.getpid(),'startticks':int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),'receipt_SHA256':hashlib.sha256(raw).hexdigest()},f)
def expired(*args):raise TimeoutError('independent model digest budget expired')
signal.signal(signal.SIGALRM,expired);signal.alarm(600)
model=root/'MiniCPM-V-4.6'
try:
    assert sorted(x.name for x in model.iterdir())==sorted(PINS)
    for name,row in PINS.items():
        p=model/name;before=p.stat();assert p.is_file() and not p.is_symlink() and before.st_size==row['bytes']
        digest=hashlib.sha256()
        with p.open('rb') as f:
            for data in iter(lambda:f.read(1024*1024),b''):digest.update(data)
        after=p.stat();assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
        assert digest.hexdigest()==row['SHA256'],name
    result={'state':'INDEPENDENT_ALL_TEN_ORIGINAL_MINICPM_MODEL_FILES_VERIFIED','observed_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'claim_commit':receipt['claim_commit'],'receipt_SHA256':hashlib.sha256(raw).hexdigest(),'files':PINS,'retained_bytes':receipt['retained_bytes'],'tail_bytes':receipt['tail_bytes'],'original_two_failures_preserved':True,'model_directory':str(model),'GPU_execution':False,'model_loaded':False}
    with (repair/'independent-model.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
except BaseException as error:
    with (repair/'independent-reader-failure.json').open('x') as f:json.dump({'error_type':type(error).__name__},f)
    raise
'''
with (BASE/'independent-reader-local-intent.json').open('x') as f:json.dump({'claim_commit':intent['claim_commit']},f)
r=subprocess.run(ns['ssh'],input=('PINS='+repr(pins)+'\nINTENT='+repr(intent)+'\n'+code).encode(),capture_output=True,timeout=650)
with (BASE/'independent-reader.stderr').open('xb') as f:f.write(r.stderr)
assert r.returncode==0,'Independent reader failed; reconcile saved owner and shared outcome, never replay'
v=json.loads(r.stdout)
with (BASE/'independent-completed-model.json').open('x') as f:json.dump(v,f,indent=2)
print(json.dumps({k:x for k,x in v.items() if k!='files'}))
