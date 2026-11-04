"""Actual scalar relay consumers; model/process leaves explicitly recording only."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import socket
import struct
import sys
import threading
import time
import traceback

S=Path(__file__).resolve().parent
B=Path('/refactor/candidates/GLM444-verified-ready-socket-CPU-v1')
OLD=Path('/refactor/results/GLM-W4-fewshot-quality-r444/startupdiag-20261104-v4-s0')
sha=lambda raw:hashlib.sha256(raw).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-sha',required=True);a=p.parse_args()
    role=int(os.environ['SLURM_PROCID']);assert role in(0,1)
    assert socket.gethostname().split('.')[0]==['a02r4n14','a02r4n15'][role]
    B.mkdir(parents=True,exist_ok=True)
    out=B/('node-%d.json'%role);assert not out.exists()
    q=dict(completed=False,passed=False,role=role,node=socket.gethostname(),events=[],cases=[],errors=[],
        source_manifest_SHA256=a.source_sha,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
        recording_metadata_only=True,GPU=0,Model=False,Native_dlopen=False,SDK=False,weights=False,
        original879_unchanged=True,CPU_fixture_epoch=float(os.environ['GLM_READY_CPU_SCHEMA_EPOCH']))
    def save():
        temp=out.with_suffix('.tmp');temp.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');temp.replace(out)
    def event(name,**fields):
        q['events'].append(dict(event=name,monotonic=time.monotonic(),realtime=time.time(),**fields));save()
    def case(name,fn):
        row=dict(name=name,passed=False);q['cases'].append(row);save()
        try:row['result']=fn();row['passed']=True
        except BaseException as error:row['error']=repr(error);save();raise
        save()
    def capture(fn):
        try:return dict(ok=True,result=fn())
        except BaseException as error:return dict(ok=False,error=repr(error),errno=getattr(error,'errno',None))
    def ports(ips):
        rows=[]
        for ip in ips:
            for port in(43843,43844):
                with socket.socket()as sock:sock.bind((ip,port))
                rows.append(dict(IP=ip,port=port,bindable=True))
        return rows
    save()
    try:
        assert all(os.environ[k]=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
        mf=json.loads((S/'SOURCE_MANIFEST.json').read_bytes());assert sha((S/'SOURCE_MANIFEST.json').read_bytes())==a.source_sha
        for name,h in mf['files_SHA256'].items():assert sha((S/name).read_bytes())==h,name
        P=S/'provider';pm=json.loads((P/'SOURCE_MANIFEST.json').read_bytes())
        for name,h in pm['files_SHA256'].items():assert sha((P/name).read_bytes())==h,name
        sys.path.insert(0,str(P))
        from public_glm_native import protocol,ready_socket as wire,provider,actors
        for module in(protocol,wire,provider,actors):assert Path(module.__file__).resolve().is_relative_to(P.resolve())
        q['actual_imports']={v.__name__:dict(path=v.__file__,SHA256=sha(Path(v.__file__).read_bytes()))for v in(protocol,wire,provider,actors)}
        original=[]
        for name,h in mf['original_ready_SHA256'].items():
            raw=(OLD/name).read_bytes();assert sha(raw)==h,name
            row=json.loads(raw);assert row['identity']['actual_step']=='910376.879'
            original.append((name,raw,row))
        config=json.loads((S/'config-CPU.json').read_bytes())
        config.update(stage=role,actual_step=q['actual_CPU'],gpu_start_epoch=q['CPU_fixture_epoch'])
        q['environment']={k:os.environ[k]for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']}
        q['fresh']=ports([config['ready_metadata_IPs'][role]])
        # The actual unchanged CLI verifies the complete privately assembled package.
        argv=sys.argv[:]
        try:
            sys.argv=['public_glm_native','--config',str(S/'config-CPU.json'),'--config-sha',sha((S/'config-CPU.json').read_bytes()),
                '--source-sha',sha((P/'SOURCE_MANIFEST.json').read_bytes()),'--stage',str(role),'--run','CPU-ready-CLI',
                '--actual-step',q['actual_CPU'],'--gpu-start-epoch',str(q['CPU_fixture_epoch'])]
            runpy.run_module('public_glm_native',run_name='__main__')
        finally:sys.argv=argv
        q['actual_whole_source_CLI']=True;save()
        def cfg(name,stage,ips=None,age=0):
            c=dict(config,stage=stage,run='CPU-recording-'+name,runtime_directory=str(B/'runs'),
                source_manifest_SHA256=sha((P/'SOURCE_MANIFEST.json').read_bytes()),
                provider_config_SHA256=sha((S/'config-CPU.json').read_bytes()))
            if ips:c['ready_metadata_IPs']=ips
            if age:c['gpu_start_epoch']=time.time()-age
            return c
        def records(run):
            values=[]
            for name,raw,row in original:
                v=copy.deepcopy(row);v['original_ready_identity']=v['identity'];v['identity']=run.identity()
                v.update(recording_metadata_only=True,not_a_current_loaded_owner=True,original_ready_SHA256=sha(raw))
                values.append(v)
            return values
        def publish(run):
            for (name,_,_),row in zip(original,records(run)):
                data=dict(row);data.pop('identity');run.write(name,data)
            return run.all_ready()
        def join(thread):
            thread.join(timeout=4);assert not thread.is_alive(),'CPU recording thread did not exit'
        # Local tests are on node15, retaining original receiver fault role.
        if role==1:
            index=0
            def loop_ips():
                nonlocal index
                index+=1;return ['127.70.0.%d'%index,'127.71.0.%d'%index]
            def wire_case(kind):
                ips=loop_ips();c=cfg('wire-'+kind,1,ips,age=200 if kind=='expired'else 0)
                run=protocol.Run(c);values=records(run);errors=[]
                with wire.ReadyReceiver(run)as receiver:
                    def client():
                        try:
                            c0=protocol.Run(dict(c,stage=0));limit=time.monotonic()+4
                            with socket.socket()as sock:
                                sock.bind((ips[0],0));sock.connect((ips[1],43844))
                                hello=wire.receive_json(sock,c0,lambda:False,limit)
                                value=dict(kind='ready-records-v1',nonce=hello['nonce'],identity=c0.identity(),records=values,
                                    stage0_all_ready_return_unix=time.time()-150 if kind=='expired'else time.time())
                                if kind=='nonce':value['nonce']='0'*64
                                elif kind=='identity':value['identity']=dict(value['identity'],run='foreign')
                                elif kind=='missing':value['records']=values[:-1]
                                elif kind=='overlap':value['records'][1]['affinity']=copy.deepcopy(values[0]['affinity'])
                                elif kind in('loaded','postload_4GiB','service_agreement'):value['records'][0][kind]=False
                                if kind=='truncated':sock.sendall(struct.pack('!I',100)+b'{');sock.shutdown(socket.SHUT_WR)
                                elif kind=='oversized':sock.sendall(struct.pack('!I',wire.MAX_FRAME+1));sock.shutdown(socket.SHUT_WR)
                                elif kind=='duplicate_JSON':
                                    raw=b'{"kind":1,"kind":2}';sock.sendall(struct.pack('!I',len(raw))+raw)
                                elif kind=='nonfinite_JSON':
                                    raw=b'{"x":NaN}';sock.sendall(struct.pack('!I',len(raw))+raw)
                                else:wire.send_json(sock,value,c0,lambda:False,limit)
                                if kind=='positive':
                                    ack=wire.receive_json(sock,c0,lambda:False,limit)
                                    assert ack['records_SHA256']==sha(wire.canonical(values))
                        except BaseException as error:errors.append(repr(error))
                    thread=threading.Thread(target=client);thread.start()
                    try:
                        got=receiver.receive()
                        assert kind=='positive','Invalid proof accepted'
                        assert got==values
                        outcome='accepted'
                    except (ValueError,AssertionError,TimeoutError,EOFError)as error:
                        if kind=='positive':raise
                        assert str(error)!='Invalid proof accepted';outcome=repr(error)
                    finally:join(thread)
                assert receiver.listener is receiver.connection is None
                # Rejected peers may make the receiver the active TCP closer.
                # A closed FD with TIME_WAIT is not a leaked listener; endpoints
                # are unique per case, so no retry or SO_REUSEADDR hides it.
                bind_observation=capture(lambda:ports(ips))
                assert not errors,errors
                return dict(outcome=outcome,FDs_closed=True,post_bind_observation=bind_observation)
            for name in ['positive','nonce','identity','missing','overlap','loaded','postload_4GiB','service_agreement','expired','truncated','oversized','duplicate_JSON','nonfinite_JSON']:
                case('actual_socket_'+name,lambda n=name:wire_case(n))
            def provider_case(inject):
                ips=loop_ips();c=cfg('provider-'+inject,1,ips);stop=threading.Event();sender_errors=[];publications=[]
                class Engine:
                    def start(self):pass
                    def stop(self):pass
                    def is_healthy(self):return True
                obj=provider.GLMProvider(c,dict(engine=Engine(),identity={'recording_only':True}))
                group=obj.group
                c0=dict(c,stage=0);peer=actors.Actors(c0)
                peer.monitor_thread=threading.Thread(target=peer.finished.wait);peer.monitor_thread.start()
                def sender():
                    try:peer.wait_peer_node()
                    except BaseException as error:sender_errors.append(repr(error))
                thread=None
                # Inspect listener ordering without establishing a second connection.
                def start_recording():
                    nonlocal thread
                    occupied=False
                    try:
                        with socket.socket()as test:test.bind((ips[1],43844))
                    except OSError:occupied=True
                    assert occupied,'Listener must precede every owner-ready publication'
                    group.owner_thread=threading.Thread(target=stop.wait,name='recording-emitter-leaf');group.owner_thread.start()
                    publish(group.run)
                    for rank in range(8):group.run.write('closed-0-%d.json'%rank,dict(normal=True,recording_metadata_only=True))
                    thread=threading.Thread(target=sender);thread.start()
                group.start=start_recording;group.stop=lambda:None # Native owner/process leaves; actual changed wiring retained.
                real_write=group.run.write
                def spy(name,value):
                    if name=='MODEL_READY.json':
                        publications.append(value)
                        real_write('CPU_EXPECTED_READY_PUBLICATION.json',dict(recording_metadata_only=True,value=value))
                    else:real_write(name,value)
                group.run.write=spy
                exit_real=wire.ReadyReceiver.__exit__
                def exit_after_closed(receiver,kind,error,tb):
                    result=exit_real(receiver,kind,error,tb)
                    assert receiver.listener is receiver.connection is None
                    if kind is None:
                        if inject=='STOP':group.run.stop()
                        elif inject=='owner_failed':group.failure=RuntimeError('recording owner failure afterACK');group.failed.set()
                        elif inject=='owner_exited':stop.set();join(group.owner_thread)
                    return result
                wire.ReadyReceiver.__exit__=exit_after_closed
                try:
                    if inject=='positive':obj.start();assert obj._ready and len(publications)==1
                    else:
                        try:obj.start()
                        except (RuntimeError,TimeoutError):pass
                        else:raise AssertionError('Publication cancellation failed')
                        assert not obj._ready and publications==[]
                    assert thread is not None;join(thread);assert not sender_errors,sender_errors
                    assert peer.stage0_all_ready_return_unix is not None and peer.finished.is_set()
                finally:
                    wire.ReadyReceiver.__exit__=exit_real;stop.set()
                    if group.owner_thread:join(group.owner_thread)
                    peer.finished.set();join(peer.monitor_thread)
                    if thread:join(thread)
                ports(ips)
                return dict(publication_attempts=len(publications),actual_Actors_wait_peer_node=True,
                    actual_Provider_start=True,after_ACK_closed_injection=inject,Native_process_load_leaves='recording only')
            for name in ['positive','STOP','owner_failed','owner_exited']:
                case('actual_provider_'+name,lambda n=name:provider_case(n))
            def keeper_case():
                spec=importlib.util.spec_from_file_location('unchanged_keeper_gate',S/'startup_gate.py')
                mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
                c=cfg('keeper-late',1);run=protocol.Run(c);publish(run)
                gate=mod.StartupGate(run);start=time.time();assert gate.observe(start)is False
                try:gate.observe(start+120)
                except mod.StartupTimeout:pass
                else:raise AssertionError('Late READY should latch failure')
                # New late input is a quarantined CPU record, not service READY.
                run.write('CPU_LATE_READY_RECORDING.json',dict(Model_READY=True,actual_owner_count=16,recording_metadata_only=True))
                try:gate.observe(start+121)
                except mod.StartupTimeout:pass
                else:raise AssertionError('Latch released')
                assert gate.blocked and not gate.accepted
                return gate.state()
            case('unchanged_keeper_late_READY_latch',keeper_case)
        run=protocol.Run(cfg('cross-node-receipt',role));q['cross_node_Run_identity']=run.identity()
        if role==1:
            with wire.ReadyReceiver(run)as receiver:
                # CPU-only ordering barrier, never used for propagation timing.
                (B/'CPU_LISTENER_BOUND.json').write_text(json.dumps(dict(actual_CPU=q['actual_CPU'],recording_metadata_only=True)))
                event('cross.listener_bound')
                values=receiver.receive();event('cross.verified_after_all16_hardlink_receipt',records_SHA256=sha(wire.canonical(values)))
                samples=[]
                for name,_,_ in original:
                    path=run.path/name
                    def opened():
                        fd=os.open(path,os.O_RDONLY)
                        try:raw=os.read(fd,65537)
                        finally:os.close(fd)
                        return dict(bytes=len(raw),SHA256=sha(raw))
                    samples.append(dict(name=name,Path_exists=capture(path.exists),
                        os_stat=capture(lambda p=path:dict(bytes=os.stat(p).st_size,inode=os.stat(p).st_ino)),os_open=capture(opened)))
                q['causally_post_publication_local_samples']=samples;save()
            assert receiver.listener is receiver.connection is None
        else:
            until=time.monotonic()+75
            while not(B/'CPU_LISTENER_BOUND.json').exists():
                if time.monotonic()>=until:raise TimeoutError('CPU-only listener ordering barrier missing')
                time.sleep(.05)
            flag=json.loads((B/'CPU_LISTENER_BOUND.json').read_bytes());assert flag['actual_CPU']==q['actual_CPU']
            time.sleep(.3)
            for(name,_,_),row in zip(original,records(run)):
                value=dict(row);value.pop('identity');run.write(name,value)
                event('cross.original_hardlink.return',name=name,SHA256=sha((run.path/name).read_bytes()))
            values=run.all_ready();first=time.time();event('cross.all16_hardlink_and_original_validation.return')
            wire.send_verified_ready(run,values,first);event('cross.same_TCP_receipt_ACK.return',records_SHA256=sha(wire.canonical(values)))
        q['post']=ports([config['ready_metadata_IPs'][role]])
        q['forbidden_modules_present']=[name for name in sys.modules if name=='torch'or name.startswith('torch.')or name in('_infinilm',)]
        q['Native_SDK_maps']=[line for line in Path('/proc/self/maps').read_text().splitlines()if any(v in line for v in ['libGLM_','libinfini','_infinilm','libtorch','libcuda','libamdhip'])]
        assert not q['forbidden_modules_present']and not q['Native_SDK_maps']
        q.update(completed=True,passed=True)
    except BaseException as error:
        q['errors'].append(dict(error=repr(error),traceback=traceback.format_exc()));raise
    finally:save()


if __name__=='__main__':main()
