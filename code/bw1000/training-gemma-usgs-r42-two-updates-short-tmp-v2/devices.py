"""Original shared HOST PCI/KFD/FD snapshot with explicit fresh scope binding."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket

from contract import D,ROOT,ASSET,REQUEST,NODE,PORT,stable_json,save,need


def device_snapshot(phase):
    spec=importlib.util.spec_from_file_location('gemma_shared_snapshot',D/'snapshot.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.NODE=NODE
    value=module.snapshot(False)
    with socket.socket() as s:
        s.bind(('127.0.0.1',PORT))
    value.update(request_id=REQUEST,master_port=PORT,master_port_available=True,phase=phase)
    if phase=='post':
        native,_=stable_json(ROOT/'native-outcome.json')
        need(native['supervisor']['host']==socket.gethostname()
             and native['supervisor']['boot_id']==value['boot_id'],'post strong owner host/boot unknown')
        remaining=[]
        for item in native['owners'].values():
            path=Path('/proc')/str(item['PID'])/'stat'
            try:
                row=path.read_text().rsplit(') ',1)[1].split()
            except (FileNotFoundError,ProcessLookupError):
                continue
            if int(row[19])==item['startticks'] and row[0] not in ('Z','X'):
                remaining.append(item['PID'])
        value.update(owned_processes_remaining=remaining,owned_registry_present=True,
                     held_reaped=native['held_cleanup'].get('reaped'),
                     normal_native_exit=native['passed'])
        need(not remaining and value['held_reaped'] is True,'original owned process remains/unknown')
    need(value['kfd_holder_count']==0 and value['full_fd_inventory_performed'] is True,
         'selected HOST KFD/FD inventory not clean and complete')
    save('before.json' if phase=='pre' else 'after.json',value)
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('pre','post'));a=p.parse_args()
    r=device_snapshot(a.phase)
    print(json.dumps(dict(phase=a.phase,request_id=REQUEST,step=r['step'],passed=True)))
