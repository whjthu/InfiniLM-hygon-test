"""One actual worker/strict loader and one synthetic long-prefix RPC session."""
from pathlib import Path
import argparse,hashlib,json,os,socket,subprocess,sys,time,traceback,uuid


def wait_listener(process,port,deadline):
    while time.time()<min(deadline-30,START+120):
        if process.poll() is not None:raise RuntimeError('Owned worker exited before passive listener')
        inodes=set()
        for table in ['/proc/net/tcp','/proc/net/tcp6']:
            for line in Path(table).read_text().splitlines()[1:]:
                fields=line.split()
                if int(fields[1].split(':')[-1],16)==port and fields[3]=='0A':inodes.add(fields[9])
        for fd in Path('/proc',str(process.pid),'fd').iterdir():
            try:link=os.readlink(fd)
            except OSError:continue
            if link.startswith('socket:[')and link[8:-1]in inodes:return
        time.sleep(.1)
    raise TimeoutError('Passive owned worker listener deadline')


def run_rpc(connection,send,recv,full,tail,save):
    """This exact plan is also executed by CPU whole-entry recording providers."""
    sequence=0;rows=[]
    def command(cmd,body=b'',**meta):
        nonlocal sequence
        request=dict(cmd=cmd,seq=sequence)
        if cmd!='close':request.update(epoch=1,request=1)
        request.update(meta);tick=time.perf_counter();send(connection,request,body)
        response,payload=recv(connection)
        assert response['seq']==sequence and not payload
        rows.append(dict(cmd=cmd,seq=sequence,wall_s=time.perf_counter()-tick,
                         response=response,body_bytes=len(body)))
        sequence+=1;save(rows)
        return response
    command('begin',prompt_tokens=32760)
    for index,past in enumerate(range(0,32760,512)):
        width=min(512,32760-past)
        response=command('append_prompt',full if width==512 else tail,
                         past=past,rows=width,capture_epoch=index+1)
        assert response['past_after']==past+width
    command('seal',past=32760)
    answer=command('query',past=32760,anchor=163584,K=7)
    assert answer['mode']=='draft'and len(answer['proposals'])==7
    assert all(type(x)is int and 0<=x<163840 for x in answer['proposals'])
    command('retire');assert command('close')['closed']
    return rows


START=time.time()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest-sha',required=True)
    parser.add_argument('--run',required=True);parser.add_argument('--deadline-unix',required=True,type=float)
    args=parser.parse_args();D=Path(__file__).resolve().parent;R=Path('/refactor')
    assert 30<args.deadline_unix-time.time()<=600
    assert os.environ.get('KIMI_MEMORY_PRESSURE_R362')=='1'
    manifest=D/'SOURCE_MANIFEST.json';assert hashlib.sha256(manifest.read_bytes()).hexdigest()==args.manifest_sha
    m=json.loads(manifest.read_bytes())
    assert all(hashlib.sha256((D/n).read_bytes()).hexdigest()==h for n,h in m['files_SHA256'].items())
    sys.path[:0]=[str(D),str(R/'incoming/Kimi-long-draft-RPC-CPU-r333')]
    from worker import expected_hello
    from batch_wire_r9128 import send,recv
    result=R/'results'/('Kimi-r362-pressure-'+args.run+'.json')
    worker_result=R/'results'/('Kimi-r362-worker-'+args.run+'.json')
    assert not result.exists()and not worker_result.exists()
    q=dict(completed=False,passed=False,run=args.run,actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
           GPU_cards=1,target_cards=0,source_manifest_SHA256=args.manifest_sha,synthetic_seed324_periodic_features=True,
           target_acceptance_qualified=False,Model_speedup_claim=False,profile=False,no_retry_reload=True)
    def save(rows=None):
        if rows is not None:q['commands']=rows
        temporary=result.with_suffix('.tmp');temporary.write_text(json.dumps(q,indent=2)+'\n');temporary.replace(result)
    process=connection=None;save()
    try:
        # CPU-only fixture production; the subprocess alone owns the single GPU.
        import torch
        torch.set_num_threads(1);torch.manual_seed(324)
        base=torch.randn(1,512,35840,dtype=torch.bfloat16)*.1
        full=base.contiguous().view(torch.uint16).numpy().tobytes()
        tail=base[:,:504].contiguous().view(torch.uint16).numpy().tobytes()
        q['fixture_full512_SHA256']=hashlib.sha256(full).hexdigest();q['fixture_tail504_SHA256']=hashlib.sha256(tail).hexdigest()
        nonce=uuid.uuid4().hex+uuid.uuid4().hex;env=dict(os.environ,KIMI_DRAFT_NONCE_R9046=nonce,INFINILM_KIMI_LONG_RPC_R334='1')
        cmd=[sys.executable,'-B','-u',str(D/'worker.py'),'--manifest-sha',args.manifest_sha,'--host','127.0.0.1',
             '--port','46962','--deadline-unix',str(args.deadline_unix),'--output',str(worker_result)]
        process=subprocess.Popen(cmd,env=env);q['owned_worker_PID']=process.pid;save()
        wait_listener(process,46962,args.deadline_unix);q['passive_listener_no_connect_probe']=True
        connection=socket.create_connection(('127.0.0.1',46962),timeout=min(120,args.deadline_unix-time.time()))
        connection.settimeout(min(120,args.deadline_unix-time.time()))
        send(connection,expected_hello(nonce));ready,body=recv(connection)
        assert ready['ready']and ready['runtime_manifest_SHA256']==m['runtime_manifest_SHA256']and not body
        q['authenticated_ready']=True;save()
        run_rpc(connection,send,recv,full,tail,save)
        connection.close();connection=None
        process.wait(timeout=max(1,args.deadline_unix-time.time()))
        assert process.returncode==0
        report=json.loads(worker_result.read_bytes())
        assert report['completed']and report['passed']and report['normal_protocol_close']
        assert report['phase_counts']==dict(begin=1,append_prompt=64,seal=1,query=1,retire=1,close=1)
        assert report['feature_bytes']==32760*35840*2
        q['passed']=True
    except BaseException as error:
        q.update(error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        if connection is not None:connection.close()
        if process is not None and process.poll()is None:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
        q.update(completed=True,worker_exit_code=None if process is None else process.poll(),wall_s=time.time()-START)
        if worker_result.exists():q.update(worker_report_path=str(worker_result),worker_report_SHA256=hashlib.sha256(worker_result.read_bytes()).hexdigest())
        save()
    print('KIMI_R362_PRESSURE_NORMAL',flush=True)


if __name__=='__main__':main()
