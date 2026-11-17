"""Future seven-wheel install worker; only runs after parent GO and owned-job handshake."""
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time
import urllib.request
from urllib.parse import urlsplit
import zipfile
from overlay_env import D,RUN,OVERLAY,environment
from job_guard import current_affinity


def require(ok,reason):
    if not ok:raise ValueError(reason)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,request,response,code,message,headers,newurl):
        return None


def main(epoch):
    lock=json.loads((D/'WHEEL_LOCK.json').read_bytes());deadline=epoch+300
    def left():
        seconds=deadline-time.time()-10;require(seconds>0,'original300s install deadline');return seconds
    gate=RUN/'job-gate.json'
    while not gate.exists():
        left();time.sleep(.02)
    ownership=json.loads(gate.read_bytes())
    require(ownership['deadline']==deadline and ownership['CPU_count']==2 and ownership['job_affinity_verified']
            and ownership['worker_affinity_mask']==current_affinity()['process_mask'],'parent native job/CPU gate differs')
    require(sys.version_info[:3]==(3,12,14),'fixed Python version differs')
    for name,version in lock['existing_runtime'].items():require(md.version(name)==version,'existing runtime version changed')
    total=0;unpacked=0;members=0;wheel_pins={}
    for name,row in sorted(lock['wheels'].items()):
        left();url=urlsplit(row['url']);require(url.scheme=='https' and url.hostname=='files.pythonhosted.org','official wheel origin differs')
        target=RUN/'wheels'/row['filename'];digest=hashlib.sha256();used=0
        request=urllib.request.Request(row['url'],headers={'User-Agent':'Hygon-workspace-overlay/1.0'})
        opener=urllib.request.build_opener(NoRedirect())
        with opener.open(request,timeout=min(12,left())) as response,target.open('xb') as stream:
            require(response.status==200 and response.url==row['url'],'exact official wheel response differs')
            while True:
                left();block=response.read(min(65536,row['bytes']-used+1))
                if not block:break
                used+=len(block);total+=len(block)
                require(used<=row['bytes'] and total<=lock['wheel_download_bytes'],'download exact byte budget exceeded')
                digest.update(block);stream.write(block)
        require(used==row['bytes'] and digest.hexdigest()==row['SHA256'],'download bytes/SHA differs')
        with zipfile.ZipFile(target) as archive:
            items=archive.infolist();require(len(items)<20000,'wheel member count exceeded')
            for item in items:
                path=PurePosixPath(item.filename)
                require(not path.is_absolute() and '..' not in path.parts and '\\' not in item.filename,'unsafe wheel member')
                require((item.external_attr>>16)&0o170000!=0o120000,'wheel symlink forbidden')
            unpacked+=sum(item.file_size for item in items);members+=len(items)
        wheel_pins[name]=dict(filename=row['filename'],bytes=used,SHA256=digest.hexdigest())
    # Conservative double extraction/download plus metadata/cache reserve; no filesystem quota is claimed.
    upper=2*unpacked+2*total+8*1024**2
    require(upper<=128*1024**2,'conservative128MiB owned-scope disk budget would be exceeded')
    (RUN/'wheel-layout.json').write_text(json.dumps(dict(wheels=wheel_pins,uncompressed_bytes=unpacked,members=members,
          conservative_peak_bound_bytes=upper,scope_cap_bytes=128*1024**2),indent=2)+'\n',encoding='utf8')
    command=[sys.executable,'-I','-B','-m','pip','--isolated','install','--no-index','--find-links',str(RUN/'wheels'),
             '--only-binary=:all:','--no-deps','--require-hashes','--no-compile','--no-cache-dir',
             '--target',str(OVERLAY),'--requirement',str(D/'requirements.lock')]
    env=environment();env['PIP_DISABLE_PIP_VERSION_CHECK']='1';env['PIP_CONFIG_FILE']=os.devnull
    with (RUN/'pip.stdout').open('xb') as out,(RUN/'pip.stderr').open('xb') as err:
        result=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=out,stderr=err,env=env,timeout=left())
    require(result.returncode==0,'exact no-deps pip install failed')
    code=(D/'smoke_overlay.py').read_text()
    with (RUN/'smoke.stdout').open('xb') as out,(RUN/'smoke.stderr').open('xb') as err:
        result=subprocess.run([sys.executable,'-B','-X','utf8','-c',code],stdin=subprocess.DEVNULL,
                              stdout=out,stderr=err,env=env,timeout=left())
    require(result.returncode==0,'per-command overlay/font smoke failed')
    (RUN/'worker-result.json').write_text(json.dumps(dict(passed=True,wheel_pins=wheel_pins,
         wheel_bytes=total,conservative_peak_bound_bytes=upper,installation_target=str(OVERLAY),
         shared_runtime_changed=False,global_environment_changed=False,WDI_GETs=0,Torch_imports=0,GPU=0),indent=2)+'\n',encoding='utf8')
    return 0


if __name__=='__main__':
    require(len(sys.argv)==2,'original epoch required')
    raise SystemExit(main(float(sys.argv[1])))
