"""One bounded, byte-pinned CPU archive. Never imports Torch or initializes a GPU."""
import argparse, gzip, hashlib, io, json, os, signal, tarfile, time
from pathlib import Path, PurePosixPath


def safe_name(name):
    p=PurePosixPath(name)
    assert name and not p.is_absolute() and '..' not in p.parts and str(p)==name
    assert '\\' not in name
    return name


def main(plan_path, output, claim):
    plan_path=Path(plan_path);root=Path(output);plan_raw=plan_path.read_bytes();plan=json.loads(plan_raw)
    assert not (root/'archive.tar.gz').exists() and not (root/'build-intent.json').exists()
    source=Path(plan['source_root']);assert source.is_dir() and not source.is_symlink()
    entries=plan['files'];assert len(entries)==76 and len({x['name'] for x in entries})==76
    assert sum(x['bytes'] for x in entries)==plan['source_bytes']<=8*1024**3
    assert all(safe_name(x['name']) and type(x['bytes']) is int and 0<x['bytes']<2*1024**3 for x in entries)
    epoch=time.time();mono=time.monotonic();deadline=epoch+240;completed=[]
    def bounded():
        assert time.time()<deadline and time.monotonic()-mono<240,'original CPU archive deadline'
    def expired(*unused):raise TimeoutError('CPU archive original240s deadline')
    signal.signal(signal.SIGALRM,expired);signal.alarm(240)
    def save(name,value,replace=False):
        p=root/name
        if replace:
            t=root/(name+'.tmp');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(p)
        else:
            with p.open('x') as f:json.dump(value,f,indent=2)
    save('build-intent.json',dict(claim=claim,epoch=epoch,deadline=deadline,PID=os.getpid(),
         startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),
         plan_SHA256=hashlib.sha256(plan_raw).hexdigest(),planned_files=76,planned_source_bytes=plan['source_bytes']))
    class HashWriter:
        def __init__(self,stream):self.stream=stream;self.h=hashlib.sha256();self.n=0
        def write(self,data):
            bounded();self.n+=len(data);assert self.n<=plan['maximum_archive_bytes'];self.h.update(data)
            return self.stream.write(data)
        def flush(self):return self.stream.flush()
    class PinnedReader:
        def __init__(self,stream):self.stream=stream;self.h=hashlib.sha256();self.n=0
        def read(self,n):
            bounded();data=self.stream.read(n);self.n+=len(data);self.h.update(data);return data
    try:
        with (root/'archive.tar.gz.part').open('xb') as raw:
            hashed=HashWriter(raw)
            with gzip.GzipFile(filename='',mode='wb',compresslevel=1,fileobj=hashed,mtime=0) as compressed:
                with tarfile.open(fileobj=compressed,mode='w|',format=tarfile.USTAR_FORMAT) as tar:
                    for entry in entries:
                        bounded();p=source/entry['name'];assert p.is_file() and not p.is_symlink()
                        assert p.resolve().is_relative_to(source.resolve()) and p.stat().st_size==entry['bytes']
                        info=tarfile.TarInfo('assets/'+entry['name']);info.size=entry['bytes'];info.mode=0o600
                        with p.open('rb') as f:
                            pinned=PinnedReader(f);tar.addfile(info,pinned)
                            assert pinned.n==entry['bytes'] and pinned.h.hexdigest()==entry['SHA256'],entry['name']
                        completed.append(entry)
                        save('build-progress.json',dict(planned_files=76,completed_files=len(completed),
                             completed_source_bytes=sum(x['bytes'] for x in completed),last_name=entry['name']),True)
                    info=tarfile.TarInfo('ASSET_PLAN.json');info.size=len(plan_raw);info.mode=0o600
                    tar.addfile(info,io.BytesIO(plan_raw))
            bounded();raw.flush();os.fsync(raw.fileno())
        (root/'archive.tar.gz.part').rename(root/'archive.tar.gz')
        result=dict(completed=True,planned_files=76,completed_files=len(completed),source_bytes=plan['source_bytes'],
                    archive_bytes=hashed.n,archive_SHA256=hashed.h.hexdigest(),plan_SHA256=hashlib.sha256(plan_raw).hexdigest(),
                    epoch=epoch,deadline=deadline,wall_seconds=time.monotonic()-mono,claim=claim,GPU_initialization=False)
        save('archive-receipt.json',result);signal.alarm(0);print(json.dumps(result))
    except BaseException as e:
        save('build-failure.json',dict(type=type(e).__name__,message=str(e)[:200],planned_files=76,
             completed_files=len(completed),completed_source_bytes=sum(x['bytes'] for x in completed)))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True);p.add_argument('--claim',required=True)
    a=p.parse_args();main(a.plan,a.output,a.claim)
