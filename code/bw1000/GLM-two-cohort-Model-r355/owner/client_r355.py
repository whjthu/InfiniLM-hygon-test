"""Six finite concurrent pairs, complete frozen bodies and existing strict gold."""
import argparse,concurrent.futures,json,pathlib,time,urllib.request,hashlib,os
from workload_r355 import PLAN,load,marker_identity
from workload_r349 import validate_reply
from source_pin_r290 import verify,D


def publish(path,value):
    with path.open('x')as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--ready',required=True);p.add_argument('--manifest',required=True);a=p.parse_args()
    verify(a.manifest);fixtures=load(D);ready=json.loads(pathlib.Path(a.ready).read_bytes())
    assert ready['run']==a.run and ready['source_manifest_SHA256']==a.manifest
    assert(ready['rows'],ready['context'],ready['TP'],ready['PP'],ready['cards'])==(2,32768,8,4,32)
    assert ready['Native_SHA256']=='038d4ff2e7aee2e6db80f691dd885222f3e3160e25ae9d0e269296daf9596c00'
    R=pathlib.Path('/refactor');out=R/'results'/('GLM-r355-client-'+a.run);out.mkdir(exist_ok=False)
    result=dict(completed=False,passed=False,HTTP_requests=0,logical_requests=12,pairs=[],buffered_HTTP=True,online_SLO_qualified=False)
    deadline=float(os.environ['GLM_R290_GPUSTART_UNIX'])+6120
    def save():
        p=out/'result.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(p)
    def request(row,index):
        left=deadline-time.time()
        if left<=120:raise TimeoutError('Original absolute client deadline')
        f=fixtures[row];req=urllib.request.Request('http://127.0.0.1:%d/v1/chat/completions'%ready['port'],data=f['raw'],headers={'Content-Type':'application/json'})
        started=time.monotonic()
        with urllib.request.urlopen(req,timeout=left-60)as response:status=response.status;raw=response.read()
        ended=time.monotonic();assert status==200
        (out/('pair%d-row%d.wire'%(index,row))).write_bytes(raw)
        wire=json.loads(raw);usage=validate_reply(wire,f['fixture'],f['gold'])
        assert usage['prompt_tokens']==len(f['ids'])
        return dict(row=row,fixture_id=f['fixture']['id'],request_id=wire['id'],common_CPU_HTTP_started=started,common_CPU_HTTP_ended=ended,
                    HTTP_wall_s=ended-started,usage=usage,quality_passed=True,wire_SHA256=hashlib.sha256(raw).hexdigest())
    try:
        for index,enabled in enumerate(PLAN):
            marker=R/'control'/('GLM-r355-quality-'+a.run+'-pair%d.json'%index)
            identity=marker_identity(a.run,index,a.manifest)
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:
                    futures=[pool.submit(request,row,index)for row in range(2)]
                    rows=[x.result()for x in futures]
                assert rows[0]['request_id']!=rows[1]['request_id']
                result['HTTP_requests']+=2;result['pairs'].append(dict(index=index,enabled=enabled,warmup=index<2,requests=rows));save()
                publish(marker,dict(identity=identity,passed=True,actual_HTTP=2,quality='existing actual upstream parser plus unchanged strict independent tool gold'))
            except BaseException as error:
                if not marker.exists():publish(marker,dict(identity=identity,passed=False,error_type=type(error).__name__,error=str(error)))
                raise
        assert result['HTTP_requests']==12
        result['passed']=True
    except BaseException as error:result.update(error_type=type(error).__name__,error=str(error))
    finally:result['completed']=True;save()
    return 0 if result['passed']else 1


if __name__=='__main__':raise SystemExit(main())
