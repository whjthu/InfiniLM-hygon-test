"""Fresh node ownership and independent post; scoped container supervisor."""
import json,os,signal,socket,sys,time
from pathlib import Path
from snapshot import snapshot
from common import W,D,R,GPU_NODE,KEEPER_NODE,Budget,need,load,save,owner,run_owned
def main(phase):
    intent=load(W/'install-intent.json');budget=Budget(intent['epoch']);budget.left()
    q=snapshot(False);need(q['node'] in (GPU_NODE,KEEPER_NODE) and (phase!='gate' or q['node']==GPU_NODE) and q['kfd_holder_count']==0 and q['same_uid_fd_denied_count']==0,'EMPTY_ASSIGNED_NODE')
    with socket.socket() as s:s.bind(('0.0.0.0',43844))
    q['port_bindable']=True;q['port']=43844;q['observed_epoch']=time.time()
    if phase!='pre':
        before=load(W/('pre-'+q['node']+'.json'));need(before['pci']==q['pci'] and before['boot_id']==q['boot_id'],'HARDWARE_IDENTITY')
        if phase=='gate':need(0<=q['monotonic_ns']-before['monotonic_ns']<=90_000_000_000,'FRESH_PRE_REQUIRED')
    if phase=='post':
        checked=[]
        for name in (('worker-owner.json','container-owner.json') if q['node']==GPU_NODE else ('keeper-owner.json',)):
            p=W/name;need(p.is_file(),'KNOWN_OWNER_RECEIPT_MISSING');record=load(p)
            need(record['host'].split('.')[0]==q['node'] and record['boot_id']==q['boot_id'],'OWNER_HOST_BOOT')
            try:now=owner(record['PID']);need(now['startticks']!=record['startticks'],'OWNED_PID_STILL_LIVE')
            except FileNotFoundError:pass
            checked.append(dict(PID=record['PID'],startticks=record['startticks'],original_absent=True))
        q['known_owners']=checked
    save(W/(phase+'-'+q['node']+'.json'),q,True)
    if phase!='gate':return
    def stop(*unused):raise TimeoutError('ORIGINAL_SUPERVISOR_STOP')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    scope='/refactor/results/'+W.name
    shell='source "$1/GPU_ENV.sh"; exec /usr/bin/python3 -B -u "$1/worker.py"'
    argv=['/public/software/apps/apptainer/1.3.4/bin/apptainer','exec','--rocm','--no-home','--pwd','/',
      '--bind',str(R)+':/refactor:ro','--bind',str(W)+':'+scope+':rw',
      '--bind','/public/home/aclsylqidf/hygon-inference-20261013:/old:ro','--bind','/opt/hyhal:/opt/hyhal:ro',str(R/'containers/dtk26-sandbox-r1'),
      '/bin/bash','--noprofile','--norc','-c',shell,'component',scope+'/src']
    result=run_owned(argv,W/'worker.private.log',budget,420,W/'container-owner.json',preserve_env=True)
    save(W/'supervisor.json',dict(result,step=q['step'],epoch=intent['epoch']),True)
    need(result['rc']==0 and not result['forced'] and result['owned_reaped'],'DEVICE_WORKER_FAILED')
if __name__=='__main__':main(sys.argv[1])
