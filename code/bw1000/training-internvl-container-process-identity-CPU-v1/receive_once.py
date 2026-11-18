import base64,hashlib,json,math,os,signal,sys,time
from pathlib import Path,PurePosixPath
RUN=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-internvl-container-process-identity-CPU-v1')
def main():
    epoch=CONTEXT['epoch'];assert math.isfinite(epoch) and 0<epoch<=time.time()<epoch+20
    signal.signal(signal.SIGALRM,lambda *_:os._exit(124));signal.setitimer(signal.ITIMER_REAL,epoch+115-time.time())
    os.umask(0o077);assert not RUN.exists() and not RUN.is_symlink()
    decoded={k:base64.b64decode(v,validate=True) for k,v in FILES.items()};assert sum(map(len,decoded.values()))<=262144
    raw=decoded['SOURCE_FILES.json'];assert hashlib.sha256(raw).hexdigest()==CONTEXT['source_manifest_SHA256']
    mf=json.loads(raw)['files'];assert set(decoded)==set(mf)|{'SOURCE_FILES.json'}
    for name,pin in mf.items():
        path=PurePosixPath(name);assert not path.is_absolute() and len(path.parts)==1 and '..' not in path.parts
        assert len(decoded[name])==pin['bytes'] and hashlib.sha256(decoded[name]).hexdigest()==pin['SHA256']
    claim=json.loads(decoded['CLAIM.json']);assert claim['root_GO'] is claim['CPU_GO'] is True and claim['GPU_GO'] is claim['Model_GO'] is claim['training_GO'] is False
    RUN.mkdir(mode=0o700);source=RUN/'source';source.mkdir()
    for name,value in decoded.items():
        with (source/name).open('xb') as f:f.write(value);f.flush();os.fsync(f.fileno())
    sys.path.insert(0,str(source))
    from common import save,own
    save(RUN/'install-intent.json',dict(CONTEXT,installer=own(),source_members=len(mf)))
    from verify_package import verify
    verify(source)
    from host_launch import main as launch
    return launch(CONTEXT)
if __name__=='__main__':raise SystemExit(main())
