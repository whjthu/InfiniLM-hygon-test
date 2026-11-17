"""Derive only original LLaVA64 input fingerprints; no model or state restoration."""
import argparse
import gc
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import signal
import stat
import sys
import time
import zipfile
from common import D,P,Budget,require,sha,stamp,safe,small,unique,save
from input_contract import receipt_files,native_binding,input_schema,check_affinity


def resolve_all(budget,out):
    root=safe(P['reference_root']);rows=[];receipt_stamps={};total=0
    pin=P['original_native_binding'];raw,_=small(Path(pin['path']),256*1024,pin)
    native_binding(unique(raw))
    for rank in range(8):
        budget.remaining(work=True)
        pin=P['source_receipts'][str(rank)];raw,identity=small(Path(pin['path']),16384,pin);receipt=unique(raw)
        files=receipt_files(receipt,rank,P)
        receipt_stamps[str(rank)]=identity
        for ordinal in range(1,9):
            name='forward-%02d-input.pt'%ordinal;record=files[name]
            require(type(record['bytes']) is int and 0<record['bytes']<=P['per_input_file_limit_bytes']
                    and type(record['SHA256']) is str and len(record['SHA256'])==64
                    and not set(record['SHA256'])-set('0123456789abcdef'),'INPUT_PIN')
            path=safe(root/('capture/rank-%02d'%rank)/name);identity=path.lstat()
            require(stat.S_ISREG(identity.st_mode) and identity.st_size==record['bytes'],'INPUT_STAT')
            total+=record['bytes'];require(total<=P['all_input_file_limit_bytes'],'TOTAL_INPUT_BUDGET')
            rows.append(dict(rank=rank,micro_ordinal=ordinal,global_step=(ordinal-1)//4,path=str(path),
                             bytes=record['bytes'],file_SHA256=record['SHA256'],identity=stamp(identity)))
    require(len(rows)==64,'ALL_PINS_BEFORE_LOAD')
    save(out/'resolved-input-pins.json',dict(case='T04.sft',rows=rows,total_input_bytes=total,
         source_receipts=P['source_receipts'],receipt_stamps=receipt_stamps,all_pins_resolved_before_Torch=True))
    return rows,receipt_stamps,total


def load_original(row,budget,torch):
    budget.remaining(work=True);path=safe(row['path'])
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as stream:
        before=os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and stamp(before)==row['identity']==stamp(path.lstat()),'INPUT_IDENTITY')
        digest=hashlib.sha256()
        for block in iter(lambda:stream.read(4*1024**2),b''):
            budget.remaining(work=True);digest.update(block)
        require(digest.hexdigest()==row['file_SHA256'] and stamp(os.fstat(stream.fileno()))==row['identity'],'INPUT_SHA')
        stream.seek(0)
        with zipfile.ZipFile(stream) as archive:
            entries=archive.infolist()
            require(len(entries)<=30000 and all(x.compress_type==zipfile.ZIP_STORED for x in entries)
                    and sum(x.file_size for x in entries)<=row['bytes']
                    and all(x.file_size<=16*1024**2 for x in entries if x.filename.endswith('/data.pkl')),'ZIP_BOUNDS')
        stream.seek(0);value=torch.load(stream,weights_only=True,map_location='cpu')
        require(stamp(os.fstat(stream.fileno()))==row['identity']==stamp(path.lstat()),'LOAD_IDENTITY')
    budget.remaining(work=True);return value


def main(epoch,out):
    budget=Budget(epoch);completed=0;phase='receipt_and_pin_resolution'
    def expired(*unused):raise TimeoutError('WORK_DEADLINE')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,budget.remaining(work=True))
    try:
        require(str(out)==P['output_namespace'],'OUTPUT_NAMESPACE')
        require(all(os.environ.get(k)=='-1' for k in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES')),'CPU_VISIBILITY')
        actual_affinity=sorted(os.sched_getaffinity(0))
        check_affinity(actual_affinity,json.loads(os.environ['EXPECTED_CPU_AFFINITY']))
        save(out/'worker-cpu-ready.json',dict(CPU_affinity=actual_affinity,excluded_CPU_cores=[0,1],
             epoch=epoch,deadline=epoch+P['whole_seconds'],before_Torch_and_payload=True))
        rows,receipt_stamps,total=resolve_all(budget,out)
        phase='qualified_CPU_Torch_import'
        import torch
        from importlib.metadata import version
        require(version('torch')==P['distribution_torch_version'] and str(torch.__version__)==P['raw_torch_version'],'TORCH_VERSION')
        version_path=Path(torch.__file__).resolve().parent/'version.py';vraw,_=small(version_path,16384)
        require(sha(vraw)==P['torch_version_source_SHA256'] and torch.version.cuda=='12.9' and torch.version.hip is None,'TORCH_SOURCE')
        torch.set_num_threads(2);torch.set_num_interop_threads(1);require(not torch.cuda.is_initialized(),'CUDA_INITIALIZED')
        canonical=D/'canonical/first_update_audit.py';craw,_=small(canonical,6939,dict(bytes=6939,SHA256=P['fingerprint_source_SHA256']))
        spec=importlib.util.spec_from_file_location('canonical_input_fingerprint',canonical);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        require(Path(inspect.getsourcefile(module.fingerprint)).resolve()==canonical.resolve(),'CANONICAL_FUNCTION_PROVIDER')
        derived=[];phase='input_only_hash_load_fingerprint'
        for row in rows:
            value=load_original(row,budget,torch)
            input_schema(value,row)
            fingerprint=module.fingerprint(dict(args=value['args'],kwargs=value['kwargs']))
            require(fingerprint['finite'] is True,'INPUT_FINITE');budget.remaining(work=True)
            derived.append({k:v for k,v in row.items() if k!='identity'}|{'input_fingerprint_SHA256':fingerprint['SHA256']})
            completed+=1;del value,fingerprint;gc.collect()
        for rank,pin in P['source_receipts'].items():require(stamp(Path(pin['path']).lstat())==receipt_stamps[rank],'RECEIPT_CHANGED')
        require(completed==64 and not torch.cuda.is_initialized(),'FINAL_DENOMINATOR')
        doc=dict(schema='native_first_two_inputs_v1',case='T04.sft',reference_request_id=P['reference_request_id'],
            original_binding_SHA256=P['original_binding_SHA256'],original_value_audit_SHA256=P['original_value_audit_SHA256'],
            fingerprint_source_SHA256=P['fingerprint_source_SHA256'],projection=['args','kwargs'],world_size=8,
            accumulation=4,source_receipts=P['source_receipts'],rows=derived)
        output_pin=save(out/'input-fingerprints.json',doc,P['reference_limit_bytes'])
        result=dict(passed=True,completed_files=64,file_denominator=64,input_bytes=total,output=output_pin,
            input_payload_only=True,other136_payloads_read=0,models=0,device_compute_calls=0,driver_query_calls=0,RNG_equivalence=False)
        save(out/'worker-result.json',result);print(json.dumps(result),flush=True);return 0
    except BaseException as error:
        result=dict(passed=False,phase=phase,error_type=type(error).__name__,completed_files=completed,file_denominator=64,
                    models=0,device_compute_calls=0,driver_query_calls=0,replay_allowed=False)
        save(out/'worker-result.json',result)
        print(json.dumps(dict(error_type=type(error).__name__,phase=phase,completed_files=completed)),file=sys.stderr,flush=True)
        return 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--epoch',type=float,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();raise SystemExit(main(args.epoch,args.out))
