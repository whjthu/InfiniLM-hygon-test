"""New auxiliary FD policy only: synthetic IO, never the actual host proc/device."""
import copy
import errno
import json
from types import SimpleNamespace
from pathlib import Path
import gpu_admission as gpu

D = Path(__file__).resolve().parent


def main():
    results=[]
    saved={key:getattr(gpu,key) for key in ['Path','observe_process','process_identity','fd_directory_metadata',
        'query','driver_information','selected_fd_holders']}
    old_uid=getattr(gpu.os,'getuid',None);old_scandir=gpu.os.scandir;old_readlink=gpu.os.readlink;old_socket=gpu.socket.socket
    class Directory:
        name='501'
        def stat(self):return SimpleNamespace(st_uid=1001)
        def __truediv__(self,key):return key
    directory=Directory()
    class Root:
        def iterdir(self):return iter([directory])
    identity=dict(PID=501,startticks=1000,PPID=20,uid=1001)
    def start(context):
        context.update(uid=1001,stage='fd_directory',identity_before=copy.deepcopy(identity),fd_uid=0,fd_mode=0o500)
    def permission(*args):
        start(args[3]);raise PermissionError(errno.EACCES,'synthetic unreadable')
    def run_observer(observer,after=None):
        gpu.Path=lambda *_:Root();gpu.observe_process=observer
        gpu.process_identity=lambda *_:copy.deepcopy(identity if after is None else after)
        gpu.fd_directory_metadata=lambda *_:dict(fd_uid=0,fd_mode=0o500)
        return saved['selected_fd_holders'](0,10**20)
    try:
        gpu.os.getuid=lambda:1001
        partial=run_observer(permission)
        record=partial['incomplete_records'][0]
        results.append(dict(name='PermissionError_partial_identity_and_errno_saved',passed=partial['FD_coverage_state']=='PARTIAL_UNKNOWN'
            and partial['incomplete_total']==partial['permission_incomplete_total']==partial['incomplete_processes']==1
            and partial['checked_own_UID_processes']==1 and partial['completed_own_UID_processes']==0
            and record['PID']==501 and record['startticks']==1000 and record['PPID']==20 and record['uid']==1001
            and record['errno']==errno.EACCES and record['fd_uid']==0 and record['fd_mode']==0o500
            and record['identity_before']==record['identity_after'] and record['original_PID_termination_proven'] is False
            and partial['all_UID_FD_complete'] is False and partial['nontransport_UID_FD_complete'] is None))
        for name,error in [('unrelated_OSError_not_swallowed',OSError(errno.EIO,'synthetic IO')),
                           ('ValueError_not_swallowed',ValueError('synthetic schema'))]:
            def fail(*args,error=error):raise error
            try:run_observer(fail);rejected=False
            except type(error):rejected=True
            results.append(dict(name=name,passed=rejected))
        changed=dict(identity,startticks=1001)
        try:run_observer(permission,changed);rejected=False
        except ValueError as error:rejected=str(error)=='FD_PROCESS_IDENTITY_CHANGED'
        results.append(dict(name='known_identity_change_hard_rejected',passed=rejected))
        def gone(*args):
            start(args[3]);args[3]['stage']='after_proc_stat';raise ProcessLookupError(errno.ESRCH,'synthetic gone')
        value=run_observer(gone)
        results.append(dict(name='process_lifecycle_is_UNKNOWN_not_terminal',passed=value['gone_or_closed_race_total']==1
            and value['permission_incomplete_total']==0 and value['partial_process_attempts']==1
            and value['incomplete_records'][0]['original_PID_termination_proven'] is False))
        # Real observe_process: one closed FD, then a readable selected GPU FD.
        class Scan:
            def __enter__(self):return iter([SimpleNamespace(name='9',path='selected'),SimpleNamespace(name='8',path='closed'),SimpleNamespace(name='10',path='other')])
            def __exit__(self,*args):return False
        target_calls=[]
        def targets(path):
            target_calls.append(path)
            if path=='closed':raise FileNotFoundError(errno.ENOENT,'synthetic FD closed')
            return '/dev/nvidia0' if path=='selected' else '/dev/null'
        gpu.os.scandir=lambda *_:Scan();gpu.os.readlink=targets
        value=run_observer(saved['observe_process'])
        results.append(dict(name='closed_FD_continues_and_readable_holder_preserved',passed=value['gone_or_closed_race_total']==1
            and value['completed_own_UID_processes']==0 and len(value['selected_device_or_KFD_holders'])==1
            and value['incomplete_records'][0]['stage']=='fd_target' and value['incomplete_records'][0]['fd_number']==8
            and target_calls==['selected','closed','other']))
        holder_observation=value
        # Exercise the actual main snapshot gate with all primary checks synthetic-valid.
        expected=json.loads((D/'INSTALL_PLAN.json').read_bytes())['expected_GPU']
        gpu.query=lambda fields,kind,deadline:[[str(expected['index']),expected['UUID'],expected['PCI'],'40442','40960']] if kind=='gpu' else []
        gpu.driver_information=lambda *_:dict(fields=dict(Device_Minor=0))
        class Socket:
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def bind(self,*args):pass
        gpu.socket.socket=Socket
        gpu.selected_fd_holders=lambda *_:partial
        value=gpu.snapshot(expected,29927,10**20)
        results.append(dict(name='primary_scope_can_pass_partial_auxiliary_without_clean_claim',passed=value['primary_GPU_admission_satisfied'] is True
            and value['all_UID_FD_complete'] is False and value['FD_observation_is_auxiliary'] is True
            and value['exclusive_GPU_proven'] is False))
        gpu.selected_fd_holders=lambda *_:holder_observation
        try:gpu.snapshot(expected,29927,10**20);rejected=False
        except ValueError as error:rejected=str(error)=='SELECTED_GPU_OR_KFD_HAS_EXISTING_OWNER'
        results.append(dict(name='actual_readable_selected_holder_still_hard_rejects',passed=rejected))
        kfd=copy.deepcopy(holder_observation);kfd['selected_device_or_KFD_holders'][0]['target']='/dev/kfd'
        gpu.selected_fd_holders=lambda *_:kfd
        try:gpu.snapshot(expected,29927,10**20);rejected=False
        except ValueError:rejected=True
        results.append(dict(name='actual_readable_KFD_holder_still_hard_rejects',passed=rejected))
        state=dict(total=0,records=[],saved_bytes=2,truncated_count=0,PIDs=set(),permission_total=0,gone_or_closed_race_total=0)
        for pid in range(1,513):
            row=copy.deepcopy(record);row['PID']=pid;row['startticks']=1000+pid
            for key in ['identity_before','identity_after']:row[key].update(PID=pid,startticks=1000+pid)
            gpu.add_incomplete(state,row)
        actual_bytes=len(json.dumps(state['records'],ensure_ascii=True,sort_keys=True,allow_nan=False).encode())
        results.append(dict(name='diagnostic64_and16KiB_caps_keep_all_failure_events',passed=state['total']==512
            and state['permission_total']==512 and len(state['PIDs'])==512 and len(state['records'])<=64
            and actual_bytes==state['saved_bytes']<=16384 and state['truncated_count']==512-len(state['records'])))
    finally:
        for key,value in saved.items():setattr(gpu,key,value)
        gpu.os.scandir=old_scandir;gpu.os.readlink=old_readlink;gpu.socket.socket=old_socket
        if old_uid is None:delattr(gpu.os,'getuid')
        else:gpu.os.getuid=old_uid
    value=dict(passed=all(row['passed'] for row in results),cases=results,synthetic_only=True,
        actual_proc_reads=0,actual_GPU=0,actual_device_queries=0,actual_remote=0,old_fixtures_repeated=False)
    print(json.dumps(value,sort_keys=True));return 0 if value['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
