"""One bounded official reference-code acquisition; never retrieves motion recordings."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

D=Path(__file__).resolve().parent
G=D.parents[2]
RUN=G/'local-only/training-hymotion-cmu-asf-fk-candidate-v1-reference-original'
URL='https://www.cs.cmu.edu/~15464-s13/assignments/assignment1/AMCViewer.zip'
CAP=16*1024**2
REQUEST='bfec0563f3024563ac92a1f1d22fc30f'


def save(name,obj):
    (RUN/name).write_text(json.dumps(obj,sort_keys=True,indent=2)+'\n',encoding='utf8')


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args):return None


def worker(deadline):
    while not (RUN/'job-gate.json').exists():
        if time.time()>=deadline:raise TimeoutError('Job gate deadline')
        time.sleep(.01)
    r=dict(passed=False,URL=URL,GET_attempted=False,received_bytes=0,EOF=False,status=None)
    try:
        gate=json.loads((RUN/'job-gate.json').read_bytes())
        if gate['CPU_count']!=2 or gate['memory_limit_bytes']!=2147483648:raise ValueError('Job admission differs')
        r['GET_attempted']=True;save('progress.json',r)
        with urllib.request.build_opener(NoRedirect()).open(urllib.request.Request(URL,headers={'Accept-Encoding':'identity'}),timeout=min(20,deadline-time.time())) as response:
            r.update(status=response.status,Content_Length=response.headers.get('Content-Length'),Content_Type=response.headers.get('Content-Type'))
            if response.status!=200 or response.headers.get('Content-Encoding','identity')!='identity':raise ValueError('HTTP framing')
            if r['Content_Length'] is not None and (not r['Content_Length'].isdigit() or not 0<int(r['Content_Length'])<=CAP):raise ValueError('reference exceeds fixed cap before body')
            h=hashlib.sha256()
            with (RUN/'AMCViewer.zip').open('xb') as f:
                while True:
                    if time.time()>=deadline:raise TimeoutError('original epoch')
                    b=response.read1(min(65536,CAP+1-r['received_bytes']))
                    if not b:r['EOF']=True;break
                    r['received_bytes']+=len(b)
                    if r['received_bytes']>CAP:raise ValueError('reference body cap')
                    f.write(b);h.update(b)
                f.flush();os.fsync(f.fileno())
            if r['Content_Length'] is not None and int(r['Content_Length'])!=r['received_bytes']:raise ValueError('length incomplete')
            r.update(passed=True,SHA256=h.hexdigest())
    except urllib.error.HTTPError as e:
        r.update(status=e.code,error_type=type(e).__name__);e.close()
    except Exception as e:r.update(error_type=type(e).__name__,reason=str(e)[:256])
    save('result.json',r);return 0 if r['passed'] else 1


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--deadline',type=float);a=p.parse_args()
    if a.worker:return worker(a.deadline)
    epoch=time.time();mono=time.monotonic();RUN.mkdir()
    save('intent.json',dict(request_id=REQUEST,epoch=epoch,deadline=epoch+90,URL=URL,max_GETs=1,max_body_bytes=CAP,no_retry=True,no_redirect=True,source_recording_download=False))
    helper=G/'code/bw1000/training-USGS-multievent-render144-v2/job_guard.py'
    if hashlib.sha256(helper.read_bytes()).hexdigest()!='c52244b5c239852c09ec2b7c5bd6aa1ffa4056eaf4e822946aaac46c9cd12b71':raise ValueError('Job helper source differs')
    spec=importlib.util.spec_from_file_location('qualified_refjob',helper);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    job=module.guarded_run([sys.executable,'-B','-X','utf8',str(Path(__file__).resolve()),'--worker','--deadline',str(epoch+90)],RUN/'job-gate.json',RUN,epoch+90,mono+90)
    save('outcome.json',dict(request_id=REQUEST,epoch=epoch,Job=job,state='KNOWN_TERMINAL'))
    print(json.dumps(dict(returncode=job['returncode'],wall_seconds=job['wall_seconds'],output=str(RUN)),sort_keys=True));return job['returncode']


if __name__=='__main__':raise SystemExit(main())
