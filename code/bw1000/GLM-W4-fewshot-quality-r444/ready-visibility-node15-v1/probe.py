"""One terminal-run filesystem differential, no Model/SDK imports or writes.

All output goes to this new CPU candidate directory. Original Run constructor
is not called (it mkdirs); the actual published read/check methods are used on
the existing, identity-bound terminal namespace.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback

S=Path(__file__).resolve().parent
R=Path('/refactor')
HOST=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
B=R/'candidates/GLM444-ready-visibility-node15-v1-CPU'
RUN='startupdiag-20261104-v4-s0'
REL=Path('results/GLM-W4-fewshot-quality-r444')/RUN
PROVIDER=R/'incoming/GLM-W4-fewshot-quality-r444-startup-diagnostic-v4/provider'
DEPLOY=PROVIDER.parent/'GPU-final-v1'
PROVIDER_SHA='2e6804b08a21a5c7e296308d4ba88552b6fd5d194e47c492cfb20de8c9ca12fe'
DEPLOY_SHA='64557b83456512166cd004b1e077f406661df1fd4e52f3d5a92af5b1d1e62c4c'
GPU='910376.879'
EPOCH=1793747525.260628
sha=lambda raw:hashlib.sha256(raw).hexdigest()


def stat_result(st):
    return dict(device=st.st_dev,inode=st.st_ino,size=st.st_size,mode=oct(st.st_mode),
                uid=st.st_uid,gid=st.st_gid,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns)


def main():
    B.mkdir(parents=True,exist_ok=True);out=B/'CPU-result.json';assert not out.exists()
    q=dict(completed=False,passed=False,read_only_original_run=True,
        actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
        hostname=socket.gethostname(),observed_unix=time.time(),paths=[],errors=[],
        Model=False,GPU=0,Native_dlopen=False,SDK=False,weights=False,compile=False,
        original_GPU=GPU,original_keeper='910407.941',original_epoch=EPOCH,
        original_run_constructor_called=False,source_manifest_SHA256=sha((S/'SOURCE_MANIFEST.json').read_bytes()),
        limitation='Post-STOP same-node/container observation. Does not recreate vanished process Path accessor state or identify its old wait(name) local variable.')
    def save():
        tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(q,indent=2)+'\n');tmp.replace(out)
    def operation(fn):
        try:return dict(ok=True,result=fn())
        except BaseException as error:return dict(ok=False,error=repr(error),errno=getattr(error,'errno',None))
    save()
    try:
        assert q['hostname'].split('.')[0]=='a02r4n15'
        assert all(os.environ[k]=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
        manifest=json.loads((S/'SOURCE_MANIFEST.json').read_bytes())
        for n,h in manifest['files_SHA256'].items():assert sha((S/n).read_bytes())==h,n
        assert sha((PROVIDER/'SOURCE_MANIFEST.json').read_bytes())==PROVIDER_SHA
        assert sha((DEPLOY/'SOURCE_MANIFEST.json').read_bytes())==DEPLOY_SHA
        pmanifest=json.loads((PROVIDER/'SOURCE_MANIFEST.json').read_bytes())
        protocol=PROVIDER/'public_glm_native/protocol.py'
        assert sha(protocol.read_bytes())==pmanifest['files_SHA256']['public_glm_native/protocol.py']
        q['protocol_SHA256']=sha(protocol.read_bytes())
        watch=R/'results/GLM-r444-startup-diagnostic-v4-startupdiag-20261104-v4-s0-watch.json'
        w=json.loads(watch.read_bytes());assert w['actual_GPU']==GPU and w['CPU_keeper']=='910407.941'
        assert w['completed']and not w['passed']and w['actual_client']is None and not w['forced_cleanup']
        assert w['original_GPU_start_unix']==EPOCH
        config=json.loads((DEPLOY/'config-slot0.json').read_bytes())
        config.update(stage=1,run=RUN,actual_step=GPU,source_manifest_SHA256=PROVIDER_SHA,
            provider_config_SHA256=sha((DEPLOY/'config-slot0.json').read_bytes()),gpu_start_epoch=EPOCH)
        spec=importlib.util.spec_from_file_location('published_terminal_protocol',protocol)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        obj=module.Run.__new__(module.Run);obj.config=dict(config)
        obj.path=R/REL
        q['expected_identity']=obj.identity()
        q['Path_accessor']=dict(type=str(type(Path('.')._accessor)),
            accessor_stat_repr=repr(Path('.')._accessor.stat),os_stat_repr=repr(os.stat),
            Path_exists_module=Path.exists.__module__)
        q['mountinfo_filtered']=[line for line in Path('/proc/self/mountinfo').read_text().splitlines()
            if '/refactor'in line or str(HOST)in line or' - nfs'in line]
        q['directories']=[]
        for root in [R,HOST]:
            folder=root/REL
            directory=dict(path=str(folder),resolve=operation(lambda: str(folder.resolve(strict=True))),
                lstat=operation(lambda:stat_result(os.lstat(folder))),stat=operation(lambda:stat_result(os.stat(folder))))
            entries={}
            with os.scandir(folder)as scan:
                for entry in scan:
                    if entry.name.startswith('ready-')and entry.name.endswith('.json'):
                        entries[entry.name]=operation(lambda e=entry:stat_result(e.stat(follow_symlinks=False)))
            directory['scandir_ready']=entries;q['directories'].append(directory);save()
            fd=os.open(folder,os.O_RDONLY|os.O_DIRECTORY)
            try:
                for stage in range(2):
                    for rank in range(8):
                        name='ready-%d-%d.json'%(stage,rank);path=folder/name
                        row=dict(path=str(path),stage=stage,rank=rank,observed_unix=time.time(),
                            Path_exists=operation(path.exists),os_stat=operation(lambda:stat_result(os.stat(path))),
                            dirfd_stat=operation(lambda:stat_result(os.stat(name,dir_fd=fd))))
                        def opened(*,relative):
                            handle=os.open(name,os.O_RDONLY,dir_fd=fd)if relative else os.open(path,os.O_RDONLY)
                            try:
                                st=os.fstat(handle);raw=os.read(handle,65537);assert len(raw)<=65536
                            finally:os.close(handle)
                            value=json.loads(raw);assert value['identity']==obj.identity()
                            assert(value['stage'],value['rank'])==(stage,rank)
                            return dict(stat=stat_result(st),bytes=len(raw),SHA256=sha(raw),
                                        identity=value['identity'],rawtext=raw.decode())
                        row['os_open_read']=operation(lambda:opened(relative=False))
                        row['dirfd_open_read']=operation(lambda:opened(relative=True))
                        obj.path=folder
                        row['actual_Run_read']=operation(lambda:obj.read(name))
                        q['paths'].append(row);save()
            finally:os.close(fd)
        q['proc_maps']=Path('/proc/self/maps').read_text()
        assert not any(n in sys.modules for n in ['torch','_infinilm','transformers'])
        assert 'libGLM_V_on_demand_Model_r403.so'not in q['proc_maps']
        q['all_paths_exists_now']=all(row['Path_exists']['ok']and row['Path_exists']['result']for row in q['paths'])
        q['all_methods_succeeded_now']=all(all(row[key]['ok']for key in ['Path_exists','os_stat','dirfd_stat','os_open_read','dirfd_open_read','actual_Run_read'])for row in q['paths'])
        byname={}
        for row in q['paths']:
            if row['os_open_read']['ok']:
                got=row['os_open_read']['result'];key=(row['stage'],row['rank'])
                byname.setdefault(key,[]).append((got['stat']['device'],got['stat']['inode'],got['SHA256']))
        q['both_aliases_same_inode_and_bytes_now']=len(byname)==16 and all(len(v)==2 and len(set(v))==1 for v in byname.values())
        q.update(completed=True,passed=True,old_stall_wait_name='UNKNOWN: no locals/path marker in original stack',
            old_stall_cause='UNKNOWN; present consistency does not prove past cache behavior')
    except BaseException as error:
        q.update(completed=True,passed=False);q['errors'].append(dict(error=repr(error),traceback=traceback.format_exc()))
    finally:save()
    print(json.dumps({k:q.get(k)for k in ['passed','actual_CPU','hostname','all_paths_exists_now','all_methods_succeeded_now','both_aliases_same_inode_and_bytes_now','errors']}))
    return 0 if q['passed']else 1


if __name__=='__main__':raise SystemExit(main())
