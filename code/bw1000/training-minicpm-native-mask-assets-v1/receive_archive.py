"""Receive one exact compressed stream, then verify every private original byte."""
import argparse, hashlib, json, os, signal, sys, tarfile, time
from pathlib import Path
from build_archive import safe_name


def extract_pinned_archive(archive, root, plan, raw, bound, completed):
    expected={'assets/'+x['name']:x for x in plan['files']};assert len(expected)==76
    with tarfile.open(archive,mode='r|gz') as tar:
        seen=set()
        for member in tar:
            bound();name=safe_name(member.name)
            assert name not in seen and member.isfile() and not member.issym() and not member.islnk()
            assert member.mode==0o600 and member.uid==member.gid==member.mtime==0
            seen.add(name)
            if name=='ASSET_PLAN.json':
                assert member.size==len(raw) and tar.extractfile(member).read()==raw;continue
            assert name in expected;entry=expected[name];assert member.size==entry['bytes']
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
            digest=hashlib.sha256();count=0
            with path.open('xb') as out:
                source=tar.extractfile(member)
                while True:
                    bound();data=source.read(1024**2)
                    if not data:break
                    count+=len(data);assert count<=entry['bytes'];out.write(data);digest.update(data)
            assert count==entry['bytes'] and digest.hexdigest()==entry['SHA256'],name
            completed.append(entry)
    assert seen==set(expected)|{'ASSET_PLAN.json'} and len(completed)==76
    return completed


def main(plan_path, expected_SHA256, expected_bytes):
    raw=Path(plan_path).read_bytes();plan=json.loads(raw);root=Path(plan['receiver_root'])
    assert not root.exists(),'reconcile existing namespace; never replay transfer'
    assert 0<expected_bytes<=plan['maximum_archive_bytes']
    assert len(expected_SHA256)==64 and all(c in '0123456789abcdef' for c in expected_SHA256)
    root.mkdir(mode=0o700);epoch=time.time();deadline=epoch+360;mono=time.monotonic()
    def expired(*unused):raise TimeoutError('single receiver360s deadline')
    signal.signal(signal.SIGALRM,expired);signal.alarm(360)
    def bound():assert time.time()<deadline and time.monotonic()-mono<360
    def save(name,value):
        with (root/name).open('x') as f:json.dump(value,f,indent=2)
    save('receive-intent.json',dict(epoch=epoch,deadline=deadline,PID=os.getpid(),
         startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),
         archive_SHA256=expected_SHA256,archive_bytes=expected_bytes,plan_SHA256=hashlib.sha256(raw).hexdigest()))
    h=hashlib.sha256();remaining=expected_bytes;completed=[]
    try:
        with (root/'archive.tar.gz.part').open('xb') as f:
            while remaining:
                bound();data=sys.stdin.buffer.read(min(1024**2,remaining));assert data,'incomplete compressed stream'
                f.write(data);h.update(data);remaining-=len(data)
            assert not sys.stdin.buffer.read(1),'unexpected compressed stream suffix'
            bound();f.flush();os.fsync(f.fileno())
        assert h.hexdigest()==expected_SHA256,'compressed original bytes differ'
        (root/'archive.tar.gz.part').rename(root/'archive.tar.gz')
        extract_pinned_archive(root/'archive.tar.gz', root, plan, raw, bound, completed)
        result=dict(completed=True,planned_files=76,completed_files=76,source_bytes=plan['source_bytes'],
                    archive_SHA256=expected_SHA256,archive_bytes=expected_bytes,plan_SHA256=hashlib.sha256(raw).hexdigest(),
                    epoch=epoch,deadline=deadline,wall_seconds=time.monotonic()-mono,GPU_initialization=False,
                    qualification='Private native mask/initial trainable bytes installed; no Model or numerical qualification.')
        save('receive-complete.json',result);signal.alarm(0);print(json.dumps(result))
    except BaseException as e:
        save('receive-failure.json',dict(type=type(e).__name__,message=str(e)[:250],planned_files=76,
                                       completed_files=len(completed),remaining_compressed_bytes=remaining))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--sha256',required=True);p.add_argument('--bytes',type=int,required=True)
    a=p.parse_args();main(a.plan,a.sha256,a.bytes)
