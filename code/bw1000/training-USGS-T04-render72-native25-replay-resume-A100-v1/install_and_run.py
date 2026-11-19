"""One bounded EOF source install; no raw inputs or weight payload on PC."""
import base64,json,os,signal,sys,time
from pathlib import Path

def main():
    x=json.loads(base64.b64decode(INSTALL_B64,validate=True));c=x['context'];need=lambda v,w:None if v else (_ for _ in ()).throw(ValueError(w));need(c['deadline']==c['epoch']+900 and c['epoch']<=time.time()<c['epoch']+100,'ONE_ADMISSION_EPOCH');signal.signal(signal.SIGALRM,lambda *_:os._exit(124));signal.setitimer(signal.ITIMER_REAL,c['epoch']+860-time.time())
    files={n:base64.b64decode(b,validate=True) for n,b in x['files'].items()};import hashlib
    sha=lambda b:hashlib.sha256(b).hexdigest();need(sum(map(len,files.values()))<2*1024**2 and sha(files['SOURCE_FILES.json'])==c['source_manifest_SHA256'],'EXACT_SOURCE');mf=json.loads(files['SOURCE_FILES.json'])['files'];need(set(files)==set(mf)|{'SOURCE_FILES.json'},'EXACT_MEMBERS')
    for n,p in mf.items():need(Path(n).name==n and len(files[n])==p['bytes'] and sha(files[n])==p['SHA256'],'SOURCE_MEMBER_PIN')
    p=json.loads(files['PLAN.json']);claim=json.loads(files['CLAIM.json']);need(all(p[k] is claim[k] is True for k in ['root_GO','GPU_GO','READ_GO']) and p['Model_GO'] is p['Hygon_GPU_GO'] is False,'NEW_GPU_ADOPTION');need(sha(files['CLAIM.json'])==c['claim_SHA256'],'CLAIM_PIN');root=Path(p['namespace']);need(root.parent==Path('/home/whj/nvidia-baseline/alignment-assets') and not root.exists(),'EXCLUSIVE_NEW_NAMESPACE');root.mkdir(mode=0o700);(root/'src').mkdir(mode=0o700)
    for n,b in files.items():
        with (root/'src'/n).open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    with (root/'intent.json').open('xb') as f:f.write((json.dumps(c,sort_keys=True,separators=(',',':'))+'\n').encode());f.flush();os.fsync(f.fileno())
    env=dict(os.environ);env.update(PYTHONPATH=str(root/'src'),PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1');os.execve(sys.executable,[sys.executable,'-B','-X','utf8',str(root/'src/controller.py')],env)
if __name__=='__main__':main()
