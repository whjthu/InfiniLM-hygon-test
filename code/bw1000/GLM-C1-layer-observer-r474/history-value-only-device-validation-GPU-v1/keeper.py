"""Bounded CPU Slurm keeper; one new singleGPU child, never a daemon."""
import os,signal,time
from common import *
def main():
    intent=load(W/'install-intent.json');budget=Budget(intent['epoch']);need(socket.gethostname().split('.')[0]==KEEPER_NODE,'KEEPER_NODE')
    save(W/'keeper-owner.json',owner(os.getpid()),True)
    q=dict(passed=False,epoch=intent['epoch'],keeper_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],GPU=None,state='UNKNOWN',Model_GO=False)
    def stop(*unused):raise TimeoutError('KEEPER_ORIGINAL_STOP')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        budget.left(570,30);pre_raw=(W/('pre-'+KEEPER_NODE+'.json')).read_bytes();pre=json.loads(pre_raw);identity=owner(os.getpid())
        reservation=load(W/'reservation.json');returned_raw=(W/'pre-return.json').read_bytes();returned=json.loads(returned_raw)
        need(reservation['request_id']==intent['request_id'] and reservation['pre_return_SHA256']==H(returned_raw),'OWN_RESERVATION')
        need(all(returned[k]==intent[k] for k in ('request_id','publication','source_manifest_SHA256','claim_SHA256')) and returned['payloads'][KEEPER_NODE]['SHA256']==H(pre_raw) and returned['payloads'][KEEPER_NODE]['boot_id']==pre['boot_id'] and returned['step']==pre['step'],'PRE_RETURN_BINDING')
        need(pre['node']==KEEPER_NODE==identity['host'].split('.')[0] and pre['boot_id']==identity['boot_id'] and 0<=time.monotonic_ns()-pre['monotonic_ns']<=90_000_000_000,'KEEPER_SAME_HOST_BOOT_MONOTONIC_STALE')
        gpu=step(budget,'gpu',GPU_NODE,4,8192,1,480,['/usr/local/python3.12/bin/python3','-B',str(D/'host_probe.py'),'gate']);q['GPU']=gpu
        need(gpu['terminal'],'GPU_UNKNOWN_RETAIN_RESERVATION');q['state']='GPU_TERMINAL'
        q['passed']=gpu['passed']
    except BaseException as error:q['error_type']=type(error).__name__
    finally:save(W/'keeper-result.json',q,True)
    return 0 if q['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
