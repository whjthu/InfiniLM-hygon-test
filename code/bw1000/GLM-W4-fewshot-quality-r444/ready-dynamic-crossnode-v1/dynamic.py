"""CPU-only delayed hardlink handshake: node15 consumes, node14 publishes.

Original Run.wait/all_ready/publish_once code is imported unchanged. Trace
observations do not replace exists/read or create service MODEL_READY.
"""
import copy
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
B=R/'candidates/GLM444-ready-dynamic-crossnode-v1-CPU'
P=R/'incoming/GLM-W4-fewshot-quality-r444-startup-diagnostic-v4/provider'
D=P.parent/'GPU-final-v1'
OLD=R/'results/GLM-W4-fewshot-quality-r444/startupdiag-20261104-v4-s0'
EPOCH=1793747525.260628  # Original borrowed schema time, never a new GPU epoch.
sha=lambda raw:hashlib.sha256(raw).hexdigest()


def main():
    role=int(os.environ['SLURM_PROCID']);assert role in (0,1)
    node=socket.gethostname().split('.')[0]
    assert node==['a02r4n14','a02r4n15'][role]
    B.mkdir(parents=True,exist_ok=True);out=B/('node-%d.json'%role);assert not out.exists()
    q=dict(completed=False,passed=False,role='producer'if role==0 else'consumer',node=node,
        actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
        recording_metadata_only=True,GPU=0,Model=False,Native_dlopen=False,SDK=False,
        weights=False,compile=False,events=[],errors=[],original_epoch_unchanged=EPOCH,
        source_manifest_SHA256=sha((S/'SOURCE_MANIFEST.json').read_bytes()),
        observation_may_refresh_metadata_cache=True,
        limitation='CPU-only filesystem mechanism. No18min Model imports/loader, no vanished original process state, no historical stall repair claim.')
    def save():
        temp=out.with_suffix('.tmp');temp.write_text(json.dumps(q,indent=2)+'\n');temp.replace(out)
    def event(event_type,**fields):
        q['events'].append(dict(event=event_type,realtime=time.time(),monotonic=time.monotonic(),**fields));save()
    def capture(fn):
        try:return dict(ok=True,result=fn())
        except BaseException as error:return dict(ok=False,error=repr(error),errno=getattr(error,'errno',None))
    save()
    try:
        assert all(os.environ[k]=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
        manifest=json.loads((S/'SOURCE_MANIFEST.json').read_bytes())
        for name,h in manifest['files_SHA256'].items():assert sha((S/name).read_bytes())==h,name
        protocol=P/'public_glm_native/protocol.py'
        assert sha(protocol.read_bytes())==manifest['original_protocol_SHA256']
        spec=importlib.util.spec_from_file_location('original_published_CPU_protocol',protocol)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        config=json.loads((D/'config-slot0.json').read_bytes())
        config.update(stage=role,run='CPU-recording-ready-dynamic-v1',runtime_directory=str(B/'cpu-runs'),
            actual_step=q['actual_CPU'],source_manifest_SHA256=manifest['original_provider_SHA256'],
            provider_config_SHA256=sha((D/'config-slot0.json').read_bytes()),gpu_start_epoch=EPOCH)
        assert time.time()+120<EPOCH+config['service_seconds'],'Original schema deadline expired; no refresh'
        run=module.Run(config);assert run.path.is_relative_to(B.resolve())
        q['recording_Run_identity']=run.identity();q['recording_Run_path']=str(run.path)
        missing=B/'CPU_FIRST_REAL_WAIT_MISS.json'
        if role==0:
            original=[]
            for name,h in manifest['original_ready_SHA256'].items():
                raw=(OLD/name).read_bytes();assert sha(raw)==h,name
                row=json.loads(raw)
                assert row['identity']['actual_step']=='910376.879'and row['identity']['gpu_start_epoch']==EPOCH
                original.append((name,raw,row))
            event('producer.wait_consumer_first_real_miss')
            until=time.monotonic()+30
            while True:
                try:flag=json.loads(missing.read_bytes());break
                except FileNotFoundError:
                    if time.monotonic()>=until:raise TimeoutError('CPU consumer first miss not visible')
                    time.sleep(.05)
            assert flag['recording_metadata_only']and flag['actual_CPU']==q['actual_CPU']and flag['consumer_node']=='a02r4n15'
            event('producer.observed_consumer_miss',consumer_marker=flag)
            time.sleep(2)
            for name,raw,row in original:
                value=copy.deepcopy(row);old_identity=value.pop('identity')
                value.update(recording_metadata_only=True,not_a_current_loaded_owner=True,
                    original_ready_SHA256=sha(raw),original_ready_identity=old_identity)
                # Original Run.write -> original publish_once: fsync temporary,
                # hardlink final, unlink temporary. No copied-model execution.
                run.write(name,value)
                data=(run.path/name).read_bytes()
                event('producer.original_hardlink_publish.return',name=name,path=str(run.path/name),
                    bytes=len(data),SHA256=sha(data),exists_now=(run.path/name).exists())
                time.sleep(.1)
            q['published_recording_ready_files']=16
        else:
            wait_code=module.Run.wait.__code__
            lines=protocol.read_text().splitlines()
            sleep_lines=[i for i,line in enumerate(lines,1)if 'time.sleep(.1)'in line]
            assert sleep_lines==[71],sleep_lines
            misses={};last={};first_published=False
            def trace(frame,kind,arg):
                nonlocal first_published
                if frame.f_code is not wait_code:return None
                name=frame.f_locals.get('name');obj=frame.f_locals.get('self')
                if name is None or obj is None:return trace
                path=obj.path/name
                if kind=='call':event('consumer.actual_Run_wait.enter',name=name,path=str(path))
                elif kind=='line'and frame.f_lineno==71:
                    now=time.monotonic();misses[name]=misses.get(name,0)+1
                    if name not in last or now-last[name]>=1:
                        last[name]=now
                        def opened():
                            fd=os.open(path,os.O_RDONLY)
                            try:raw=os.read(fd,65537);st=os.fstat(fd)
                            finally:os.close(fd)
                            return dict(bytes=len(raw),SHA256=sha(raw),device=st.st_dev,inode=st.st_ino)
                        # The original false predicate was already executed.
                        # Secondary observations are explicit and may refresh
                        # FS caches; they never replace the source predicate.
                        event('consumer.actual_Run_wait.line71_after_false',name=name,path=str(path),
                            polls=misses[name],Path_exists=capture(path.exists),
                            os_stat=capture(lambda:dict(size=os.stat(path).st_size,inode=os.stat(path).st_ino)),
                            os_open=capture(opened))
                    if not first_published:
                        module.publish_once(missing,dict(recording_metadata_only=True,actual_CPU=q['actual_CPU'],
                            consumer_node=node,wait_name=name,wait_path=str(path),realtime=time.time()))
                        first_published=True
                elif kind=='return':
                    event('consumer.actual_Run_wait.return',name=name,path=str(path),
                        returned_recording_only=isinstance(arg,dict)and arg.get('recording_metadata_only',False))
                return trace
            event('consumer.actual_Run_all_ready.enter')
            sys.settrace(trace)
            try:records=run.all_ready()
            finally:sys.settrace(None)
            assert len(records)==16 and all(row['recording_metadata_only']for row in records)
            q.update(actual_all_ready_returned=16,actual_wait_poll_counts=misses)
            event('consumer.actual_Run_all_ready.return',records=16)
        assert not(run.path/'MODEL_READY.json').exists(),'CPU test cannot grant service READY'
        assert not any(n in sys.modules for n in ['torch','_infinilm','transformers'])
        q['proc_maps']=Path('/proc/self/maps').read_text()
        assert 'libGLM_V_on_demand_Model_r403.so'not in q['proc_maps']
        q.update(completed=True,passed=True,Model_READY_created=False,historical_cause_or_repair_proven=False)
    except BaseException as error:
        sys.settrace(None);q.update(completed=True,passed=False)
        q['errors'].append(dict(error=repr(error),traceback=traceback.format_exc()))
    finally:save()
    print(json.dumps({k:q.get(k)for k in ['passed','actual_CPU','node','role','actual_all_ready_returned','published_recording_ready_files','errors']}))
    return 0 if q['passed']else 1


if __name__=='__main__':raise SystemExit(main())
