"""Real GPU-step HOST gate before container SDK initialization. No model/weights."""
import os
from pathlib import Path
import socket
import time
import importlib.util
from contract import *

def main():
    verify('GPU');intent,ipin=stable_json(ROOT/'intent.json');clock=Clock(intent['epoch'])
    step=os.environ.get('SLURM_JOB_ID','')+'.'+os.environ.get('SLURM_STEP_ID','')
    need(numbered([step]) and socket.gethostname().split('.')[0]==NODE,'exact original GPU step required')
    fields=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()
    owner=dict(PID=os.getpid(),startticks=int(fields[19]),host=socket.gethostname(),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    save('gpu-host-entry.json',dict(request_id=REQUEST,epoch=intent['epoch'],step=step,owner=owner,intent=ipin))
    spec=importlib.util.spec_from_file_location('actual_effect_HOST_snapshot',D/'snapshot.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.NODE=NODE
    value=module.snapshot(False)
    before,_=stable_json(ROOT/'before.json')
    need(value['pci']==before['pci'] and len(set(value['pci']))==8 and value['boot_id']==before['boot_id'] and
         value['kfd_holder_count']==0 and value['full_fd_inventory_performed'] is True,'actual same-step eight-physical HOST gate failed')
    with socket.socket() as probe:probe.bind(('127.0.0.1',PORT))
    value.update(request_id=REQUEST,epoch=intent['epoch'],intent=ipin,owner=owner,master_port=PORT,master_port_available=True)
    save('gpu-host-before.json',value)
    clock.left(INSIDE,10)
    from controller import container_command
    argv=container_command();os.execve(argv[0],argv,dict(os.environ))

if __name__=='__main__':main()
