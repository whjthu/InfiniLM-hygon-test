"""New transport/foreign-owner pollution boundaries only; synthetic proc metadata."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import transport
import gpu_admission as gpu


def fixture():
    rows = []
    for pid, ppid, uid, name in [(104,103,1001,'python'),(103,102,1001,'sshd'),(102,101,0,'sshd'),(101,0,0,'sshd')]:
        rows.append(dict(PID=pid,PPID=ppid,PGID=101,startticks=10000+pid,comm=name,proc_uid=uid,
            proc_device=1,proc_inode=pid,fd_uid=1001 if pid==104 else 0,fd_mode=0o500,fd_device=1,fd_inode=20000+pid,
            status_identity=dict(PPid=ppid,Uid=[uid]*4,Gid=[uid]*4,TracerPid=0,NSpid=[pid],CapEff='0000000000000000',NoNewPrivs=0),
            observed_states=['S']*3,status_bytes=500,status_SHA256='a'*64,exe_identity='UNKNOWN',exe_read=False,FD_targets_read=False))
    owner=dict(PID=104,startticks=10104,uid=1001,host='synthetic',boot_id='syntheticboot',scope_request_id='f'*32)
    before=dict(host='synthetic',boot_id='syntheticboot',supervisor_PID=104,scope_request_id='f'*32,chain=rows,
                exe_identity='UNKNOWN',executable_identity_verified=False)
    return owner,before


def main():
    results=[];owner,before=fixture()
    def accepted_chain(value,host='synthetic'):
        try:
            transport.validate_chain(owner,value['chain'],host,'syntheticboot',104,1001);return True
        except ValueError:return False
    results.append(dict(name='current_exact_four_level_chain',passed=accepted_chain(before)))
    parent=before['chain'][1]
    raw_status=('PPid: 102\nUid: 1001 1001 1001 1001\nGid: 1001 1001 1001 1001\n'
        'TracerPid: 0\nNSpid: 103\nCapEff: 0000000000000000\nNoNewPrivs: 0\nState: S (sleeping)\n').encode()
    parsed,state=transport.status_identity(raw_status,parent,1001)
    results.append(dict(name='actual_status_parser_identity_fields',passed=parsed==parent['status_identity'] and state=='S'))
    try:transport.status_identity(raw_status.replace(b'Uid: 1001 1001 1001 1001',b'Uid: 1001 0 1001 1001'),parent,1001);rejected=False
    except ValueError:rejected=True
    results.append(dict(name='status_effective_UID_contamination_rejected',passed=rejected))
    for name,mutate in [
        ('wrong_direct_UID',lambda x:x['chain'][1].update(proc_uid=0)),
        ('wrong_root_UID',lambda x:x['chain'][2].update(proc_uid=1001)),
        ('wrong_root_link',lambda x:x['chain'][2].update(PPID=555)),
        ('wrong_parent_fd_owner',lambda x:x['chain'][1].update(fd_uid=1001)),
        ('wrong_parent_fd_mode',lambda x:x['chain'][1].update(fd_mode=0o555)),
        ('wrong_status_UID',lambda x:x['chain'][1]['status_identity'].update(Uid=[0]*4)),
        ('privileged_sameUID_parent',lambda x:x['chain'][1]['status_identity'].update(CapEff='0000000000000001')),
        ('binary_identity_not_proven',lambda x:x['chain'][1].update(exe_identity='VERIFIED')),
        ('foreign_sshd_not_directparent',lambda x:x['chain'][0].update(PPID=555)),
    ]:
        changed=copy.deepcopy(before);mutate(changed);results.append(dict(name=name,passed=not accepted_chain(changed)))
    results.append(dict(name='wrong_host',passed=not accepted_chain(before,'foreign')))
    for name,mutate in [('reused_parent_PID',lambda x:x['chain'][1].update(startticks=99)),
        ('status_identity_changed',lambda x:x['chain'][1]['status_identity'].update(NoNewPrivs=1)),
        ('boot_changed',lambda x:x.update(boot_id='different'))]:
        after=copy.deepcopy(before);mutate(after)
        try:transport.confirm(before,after);rejected=False
        except ValueError:rejected=True
        results.append(dict(name=name,passed=rejected))
    candidate={k:before['chain'][1][k] for k in ['PID','PPID','startticks','proc_uid']}
    results.append(dict(name='only_actual_FD_directory_PermissionError_eligible',passed=transport.eligible(before,candidate,'fd_directory')
        and not transport.eligible(before,candidate,'proc_identity') and not transport.eligible(before,candidate,'fd_target')))
    # Execute the real FD guard with a synthetic proc directory, never the host /proc.
    original=(gpu.Path,getattr(gpu.os,'getuid',None),gpu.transport.capture)
    class Item:
        def __init__(self,row,blocked):self.row=row;self.name=str(row['PID']);self.blocked=blocked
        def stat(self):return SimpleNamespace(st_uid=self.row['proc_uid'])
        def __truediv__(self,key):return Node(self,key)
    class Node:
        def __init__(self,item,key):self.item=item;self.key=key
        def read_text(self):
            r=self.item.row;v=['S',str(r['PPID']),str(r['PGID'])]+['0']*16+[str(r['startticks'])]
            return str(r['PID'])+' ('+r['comm']+') '+' '.join(v)
        def iterdir(self):
            if self.item.blocked:raise PermissionError('synthetic exact FD directory')
            return iter([])
    class Proc:
        def __init__(self,items):self.items=items
        def iterdir(self):return iter(self.items)
    try:
        gpu.os.getuid=lambda:1001
        for name,blocked_parent,extra,reuse,change,wanted in [
            ('one_own_transport_exception',True,False,False,False,True),
            ('no_exception_complete_UID_scan',False,False,False,False,True),
            ('additional_foreign_unreadable_sshd_rejected',True,True,False,False,False),
            ('reused_candidate_start_rejected',True,False,True,False,False),
            ('ancestry_changes_after_scan_rejected',True,False,False,True,False),
        ]:
            first=copy.deepcopy(before);second=copy.deepcopy(before)
            if change:second['chain'][2]['startticks']+=1
            captures=iter([first,second]);gpu.transport.capture=lambda *_:next(captures)
            own=copy.deepcopy(first['chain'][0]);parent=copy.deepcopy(first['chain'][1])
            if reuse:parent['startticks']+=1
            items=[Item(own,False),Item(parent,blocked_parent)]
            if extra:
                foreign=copy.deepcopy(parent);foreign['PID']=999;items.append(Item(foreign,True))
            gpu.Path=lambda *_:Proc(items)
            try:
                actual=gpu.selected_fd_holders(0,10**20,owner);accepted=True
                if blocked_parent:
                    accepted=(actual['transport_exception_count']==1 and actual['all_UID_FD_complete'] is False
                        and actual['nontransport_UID_FD_complete'] is True and actual['transport_FDs_inspected'] is False
                        and actual['scope_policy']=='scope_admission_satisfied_with_transport_exception')
                else:
                    accepted=(actual['transport_exception_count']==0 and actual['uninspected_transport_ancestor'] is None
                        and actual['all_UID_FD_complete'] is True)
            except ValueError:accepted=False
            results.append(dict(name=name,passed=accepted==wanted))
    finally:
        gpu.Path,gpu.transport.capture=original[0],original[2]
        if original[1] is None:delattr(gpu.os,'getuid')
        else:gpu.os.getuid=original[1]
    value=dict(passed=all(x['passed'] for x in results),cases=results,synthetic_only=True,
        actual_proc_reads=0,actual_exe_reads=0,actual_device_queries=0,actual_GPU=0,actual_remote=0,old_tests_repeated=False)
    print(json.dumps(value,sort_keys=True));return 0 if value['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
