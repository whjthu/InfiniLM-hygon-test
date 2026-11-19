"""Stage one fixed, verified prefix of the public LLaVA checkpoint."""
import hashlib
import json
from pathlib import Path
import runpy
import shlex
import subprocess
import sys
import time
import urllib.parse
import urllib.request


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUN = HERE.name
HELPER = Path('C:/Users/WangHaojie/.codex/visualizations/2026/10/12/01a130bd-3b97-73c2-8ecf-eda5151f1ba7/glm-paired-readonly-connection.py')
HELPER_SHA = '3f00e0eac9b40eb3fee7d788776389bc22c3e87d20de0441312f04a2566178d1'
RECEIVER = '''import hashlib,json,os,pathlib,signal,sys,time
p=PARAMETERS
assert p['deadline'] > time.time() and p['deadline'] == p['epoch'] + 150
signal.signal(signal.SIGALRM,lambda *_: os._exit(124))
signal.setitimer(signal.ITIMER_REAL,max(.001,p['deadline']-time.time()))
root=pathlib.Path(p['target'])
root.mkdir(mode=0o700)
def write(name,d):
 with (root/name).open('x',encoding='utf-8') as f:
  json.dump(d,f,sort_keys=True,allow_nan=False);f.write('\\n');f.flush();os.fsync(f.fileno())
write('intent.json',p)
path=root/'shard01.offset0.part.incomplete'
h=hashlib.sha256();remaining=p['length'];head=b''
with path.open('xb') as out:
 while remaining:
  b=sys.stdin.buffer.read(min(1048576,remaining));assert b
  if not head:head=b[:p['header_bytes']+8]
  out.write(b);h.update(b);remaining-=len(b)
 out.flush();os.fsync(out.fileno())
assert sys.stdin.buffer.read(1)==b'' and h.hexdigest()==p['range_SHA256']
assert len(head)==p['header_bytes']+8 and int.from_bytes(head[:8],'little')==p['header_bytes']
assert hashlib.sha256(head[8:]).hexdigest()==p['header_SHA256']
with path.open('rb') as f:
 assert os.fstat(f.fileno()).st_size==p['length'] and hashlib.file_digest(f,'sha256').hexdigest()==p['range_SHA256']
final=root/'shard01.offset0.part'
os.link(path,final);path.unlink()
d=os.open(root,os.O_RDONLY | os.O_DIRECTORY)
try:os.fsync(d)
finally:os.close(d)
result=dict(p,bytes_saved=final.stat().st_size,stable_target_range_SHA_matches=True,target_header_matches_A100=True,
            complete_shards=0,weights_admitted=False,finished_epoch=time.time())
write('receipt.json',result)
print(json.dumps(result,allow_nan=False))
'''


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, value):
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,sort_keys=True,allow_nan=False)
        f.write('\n')


class OfficialRedirect(urllib.request.HTTPRedirectHandler):
    """Only accept the exact HTTPS CDN host returned by the official repo."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        location=urllib.parse.urlsplit(newurl)
        if location.scheme!='https' or location.hostname!='us.aws.cdn.hf.co' or location.username is not None or location.password is not None:
            raise ValueError('UNEXPECTED_OFFICIAL_REDIRECT')
        return urllib.request.Request(newurl,headers={'Range':req.headers['Range'],'Accept-Encoding':'identity'})


def public_range(binding, left):
    limit=binding['length']
    assert limit==8*1024**2 and binding['offset']==0
    source='https://huggingface.co/'+binding['repository']+'/resolve/'+binding['revision']+'/'+binding['file']
    req=urllib.request.Request(source,headers={'Range':f'bytes=0-{limit-1}','Accept-Encoding':'identity'})
    specific=f"bytes 0-{limit-1}/{binding['shard_bytes']}"
    started=time.monotonic()
    with urllib.request.build_opener(OfficialRedirect()).open(req,timeout=min(15,left(30))) as response:
        if response.status!=206 or response.headers.get('Content-Range')!=specific or response.headers.get('Content-Length')!=str(limit):
            raise ValueError('OFFICIAL_RANGE_RESPONSE')
        size=0
        parts=[]
        while size<limit:
            left(30)
            block=response.read(min(1048576,limit-size))
            if not block:
                raise EOFError('SHORT_OFFICIAL_RANGE')
            size+=len(block)
            parts.append(block)
    data=b''.join(parts)
    n=binding['header_bytes']
    assert int.from_bytes(data[:8],'little')==n and sha(data[8:8+n])==binding['header_SHA256']
    return data,time.monotonic()-started


def main(publication):
    rel=HERE.relative_to(REPO).as_posix()
    git=['git','-c','safe.directory='+str(REPO)]
    def g(*args):return subprocess.check_output([*git,*args],cwd=REPO,timeout=12)
    assert g('rev-parse','HEAD').decode().strip()==publication
    assert not g('diff','--cached','--name-only').strip() and not g('status','--short','--',rel).strip()
    manifest=(HERE/'SOURCE_FILES.json').read_bytes()
    assert manifest==g('cat-file','blob',publication+':'+rel+'/SOURCE_FILES.json')
    for name, pin in json.loads(manifest)['files'].items():
        raw=(HERE/name).read_bytes()
        assert len(raw)==pin['bytes'] and sha(raw)==pin['SHA256']
    assert sha(HELPER.read_bytes())==HELPER_SHA
    binding=json.loads((HERE/'BINDING.json').read_bytes())
    local=REPO/'local-only'/RUN
    local.mkdir(exist_ok=False)
    epoch=time.time(); start=time.monotonic()
    def left(reserve=0):
        remaining=min(epoch+150-time.time(),start+150-time.monotonic())-reserve
        if remaining<=0:raise TimeoutError('ONE_EPOCH_EXHAUSTED')
        return remaining
    intent=dict(binding,publication=publication,source_manifest_SHA256=sha(manifest),epoch=epoch,deadline=epoch+150)
    save(local/'intent.json',intent)
    result=dict(request=binding['request'],epoch=epoch,state='LOCAL_SOURCE_NOT_COMPLETE',received_metadata=False)
    try:
        payload, seconds=public_range(binding,left)
        parameters=dict(intent,range_SHA256=sha(payload),public_range_read_seconds=seconds,read_bytes=len(payload))
        save(local/'source-range.json',parameters)
        boot=RECEIVER.replace('PARAMETERS',repr(parameters),1)
        ssh=runpy.run_path(str(HELPER))['ssh']
        command='/usr/local/python3.12/bin/python3 -I -S -B -c '+shlex.quote(boot)
        save(local/'target-attempt.json',parameters)
        result['state']='TARGET_OUTCOME_UNKNOWN'
        r=subprocess.run([ssh[0],'-o','StrictHostKeyChecking=yes',*ssh[1:],command],cwd=REPO,
                         input=payload,capture_output=True,timeout=left(3))
        result.update(rc=r.returncode,stderr_bytes=len(r.stderr),stderr_SHA256=sha(r.stderr))
        if r.returncode==0:
            receipt=json.loads(r.stdout)
            assert all(receipt[k]==v for k,v in parameters.items()) and receipt['bytes_saved']==len(payload)
            assert receipt['stable_target_range_SHA_matches'] and receipt['target_header_matches_A100']
            save(local/'receiver.json',receipt)
            result.update(state='PREFIX_COPIED_AND_VERIFIED',received_metadata=True)
    except Exception as e:
        result.update(error_type=type(e).__name__,message_SHA256=sha(str(e).encode()))
    result['whole_seconds']=time.monotonic()-start
    save(local/'outcome.json',result)
    print(json.dumps(result,allow_nan=False))
    return int(not result['received_metadata'])


if __name__=='__main__':
    raise SystemExit(main(sys.argv[1]))
