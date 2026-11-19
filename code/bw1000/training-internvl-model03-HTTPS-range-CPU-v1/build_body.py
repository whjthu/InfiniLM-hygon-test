"""One source-only body after ordinary publication; no HTTP or payload on PC."""
import base64,hashlib,json,math,re,subprocess,time
from pathlib import Path
D=Path(__file__).resolve().parent;G=D.parents[2]

def make_body(publication,request,epoch):
    assert re.fullmatch('[0-9a-f]{40}',publication) and re.fullmatch('[0-9a-f]{32}',request)
    assert type(epoch)in(int,float) and math.isfinite(epoch) and 0<epoch<=time.time()<epoch+12
    def left():
        v=epoch+20-time.time();assert v>0;return min(8,v)
    from verify_package import verify
    verify(D)
    mf=(D/'SOURCE_FILES.json').read_bytes();rel=D.relative_to(G).as_posix()
    def git(*a):return subprocess.check_output(['git','--no-optional-locks','-c','safe.directory='+G.as_posix(),'-C',str(G),*a],timeout=left())
    assert git('remote','get-url','origin').decode().strip()=='git@github.com:whjthu/InfiniLM-hygon-test.git'
    assert not git('diff','--cached','--name-only').strip()
    assert git('cat-file','blob',publication+':'+rel+'/SOURCE_FILES.json')==mf
    assert not git('status','--porcelain=v1','--untracked-files=all','--',rel)
    assert git('rev-parse','HEAD').decode().strip()==publication
    assert git('ls-remote','origin','refs/heads/bw1000-experiments').decode().split()==[publication,'refs/heads/bw1000-experiments']
    values={n:base64.b64encode((D/n).read_bytes()).decode() for n in json.loads(mf)['files']};values['SOURCE_FILES.json']=base64.b64encode(mf).decode()
    context=dict(publication=publication,request_id=request,epoch=epoch,deadline=epoch+120,source_manifest_SHA256=hashlib.sha256(mf).hexdigest())
    body=('FILES='+repr(values)+'\nCONTEXT='+repr(context)+'\n').encode()+(D/'receive_once.py').read_bytes();assert len(body)<=262144;left();return body,context
