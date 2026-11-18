"""Root invokes this local builder before the one ordinary held SSH transport."""
import base64,hashlib,json,math,re,subprocess,time
from pathlib import Path
D=Path(__file__).resolve().parent;G=D.parents[2]
def make_body(publication,request,epoch):
    assert re.fullmatch('[0-9a-f]{40}',publication) and re.fullmatch('[0-9a-f]{32}',request)
    assert type(epoch) in (int,float) and math.isfinite(epoch) and 0<epoch<=time.time()<epoch+12
    mono=time.monotonic()+epoch+20-time.time()
    def left():
        v=min(epoch+20-time.time(),mono-time.monotonic());assert v>0;return min(v,8)
    from verify_package import verify
    verify(D)
    mf=(D/'SOURCE_FILES.json').read_bytes();name=D.relative_to(G).as_posix()
    def git(*args):return subprocess.check_output(['git','--no-optional-locks','-c','safe.directory='+G.as_posix(),'-C',str(G),*args],timeout=left())
    assert git('cat-file','blob',publication+':'+name+'/SOURCE_FILES.json')==mf
    assert not git('status','--porcelain=v1','--untracked-files=all','--',name)
    plan=json.loads((D/'PLAN.json').read_bytes());prior=plan['prior_failure']
    old=git('cat-file','blob',publication+':'+prior['path']);assert len(old)==prior['bytes'] and hashlib.sha256(old).hexdigest()==prior['SHA256']
    values={k:base64.b64encode((D/k).read_bytes()).decode() for k in json.loads(mf)['files']};values['SOURCE_FILES.json']=base64.b64encode(mf).decode()
    context=dict(publication=publication,request_id=request,epoch=epoch,deadline=epoch+120,source_manifest_SHA256=hashlib.sha256(mf).hexdigest())
    body=('FILES='+repr(values)+'\nCONTEXT='+repr(context)+'\n').encode()+(D/'receive_once.py').read_bytes()
    assert len(body)<=524288;left();return body,context
