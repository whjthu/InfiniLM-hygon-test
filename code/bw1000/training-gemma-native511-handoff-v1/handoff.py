"""Byte-identical CPU511 metadata handoff; never read/authorize Gemma initial weights."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import time

BASE=PurePosixPath('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')
CPU_LABEL='training-gemma-usgs-render72-native-input-audit-v1'
TRAIN_LABEL='training-gemma-usgs-render72-exposure25-resume20-v1'
SOURCE=BASE/CPU_LABEL/'audit/result.json'
DESTINATION=BASE/TRAIN_LABEL/'native-input-cpu-result.json'
REPORT={'bytes':1218,'SHA256':'e176276ed6e821d1002d41782633b6b40bef56c7587966a036aa42d1f6db6591'}
WORKER={'bytes':344,'SHA256':'3832af1aef87b88149d955417ad39f665c50fe957a5f91caaf42f44bbc992e80'}
CAPTURE='ff448956bd5f03ba08ff3a072b88c3c83807c1360d2e2c7281ea5cab27b063e6'
REQUEST='8d6d573cc64b4ab3b082b17682043a11'
PUBLICATION='d62f910ec59c61cf2970308da16a047c33c29b30'
HANDOFF_REQUEST='6e40d7ab1eef45f293abc960b324cbf7'


def need(ok,why):
    if not ok:raise ValueError(why)


def pin(raw):return {'bytes':len(raw),'SHA256':hashlib.sha256(raw).hexdigest()}


def same_fields(value,expected,why):
    need(type(value) is dict and all(type(value.get(k)) is type(v) and value[k]==v for k,v in expected.items()),why)


def validated_report(raw):
    need(pin(raw)==REPORT,'Exact original511 independent report bytes/SHA differs')
    r=json.loads(raw)
    same_fields(r,{'state':'ORIGINAL_GEMMA_CPU72_INDEPENDENT_INPUT_VERIFICATION_PASS','request':REQUEST,
        'publication':PUBLICATION,'passed':True,'actual_verified_rows':72,'planned_rows':72,
        'observed_encode_calls':144,'observed_collator_calls':72,'optimizer_updates':0,'model_loaded':False,
        'training_READY':False,'original_replay_allowed':False,'private_capture_SHA256':CAPTURE},'Original native audit scope differs')
    need(r['worker_receipt']==WORKER and r['independent_accounting']['terminal'] is True,
         'Actual original producer/terminal binding differs')
    need(r['independent_accounting']['row']==['1103606.511',CPU_LABEL,'COMPLETED','0:0','36','f13r1n12','cpu=4,mem=8G,node=1'],
         'Original actual CPU511 row differs')
    return r


def plan_from_report(raw):
    r=validated_report(raw)
    return {'schema':'GEMMA_ORIGINAL_CPU511_METADATA_HANDOFF_PLAN_V1','handoff_request':HANDOFF_REQUEST,
        'original_CPU_request':REQUEST,'original_CPU_publication':PUBLICATION,'original_CPU_step':'1103606.511',
        'independent_report':REPORT,'source':{'path':str(SOURCE),**WORKER},
        'destination':{'path':str(DESTINATION),**WORKER},'native_cpu_receipt_SHA256_for_future_intent':WORKER['SHA256'],
        'canonical_producer_reserialized':False,'copies_capture_or_inputs':False,'initial_trainables_accessed_or_authorized':False,
        'actual_rank_loader_observed':False,'planned_eight_rank_loader_batches':100,'planned_accumulation':4,
        'planned_checkpoint20_skip_batches':80,'planned_tail_batches_per_rank':20,
        'source_unique_rows':72,'source_maximum_tokens':r['maximum_row_tokens'],'source_supervised_tokens_unique72':r['total_shifted_supervised_tokens'],
        'actual_800_exposure_supervised_tokens':None,'byte_handoff_executed':False,'training_READY':False,'GPU_GO':False,'training_GO':False}


def validate_worker(raw,report):
    # Also used by clearly synthetic local fixtures. Production always calls
    # validated_report() first, then only reads SOURCE and the fixed WORKER pin.
    need(pin(raw)==report['worker_receipt'],'Original producer raw bytes/SHA differs')
    r=json.loads(raw)
    same_fields(r,{'passed':True,'unique_rows':72,'encode_calls':144,'collator_calls':72,'optimizer_updates':0,
        'model_loaded':False,'actual_rank_loader_observed':False,'training_ready':False,
        'private_capture_SHA256':report['private_capture_SHA256']},'Native producer qualification fields differ')
    epoch,deadline=r['epoch'],r['deadline']
    need(type(epoch) in (int,float) and type(deadline) in (int,float) and math.isfinite(epoch)
         and math.isfinite(deadline) and epoch>0 and deadline==epoch+390,'Original producer CPU epoch differs')
    return r


def stable_read(path,expected):
    before=path.lstat()
    need(stat.S_ISREG(before.st_mode) and not path.is_symlink() and before.st_size==expected['bytes'],'Expected exact regular metadata file')
    with path.open('rb') as f:
        opened=os.fstat(f.fileno());raw=f.read(expected['bytes']+1);after=os.fstat(f.fileno())
    identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    need(identity(before)==identity(path.lstat()) and identity(opened)==identity(after),'Metadata changed during original read')
    if os.name!='nt':need(identity(before)==identity(opened),'Path/FD identity differs')
    need(pin(raw)==expected,'Exact metadata body pin differs')
    return raw


def exclusive_copy(raw,root,report):
    # Reversible metadata only. No run intent is created and no stale run may
    # be silently restaged. Root must call this before its new GPU intent.
    for name in ['intent.json','terminal-accounting.json','gpu-step-binding.json',
                 'continuous/native-intent.json','resumed/native-intent.json',
                 'native-input-cpu-result.json','native-input-cpu-handoff.json']:
        need(not (root/name).exists(),'Existing original run/receipt requires reconciliation: '+name)
    validate_worker(raw,report)
    output=root/'native-input-cpu-result.json'
    with output.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    need(pin(stable_read(output,report['worker_receipt']))==report['worker_receipt'],'Exclusive destination verification failed')
    receipt={'schema':'GEMMA_CPU511_ORIGINAL_BYTES_HANDOFF_RECEIPT_V1','handoff_request':HANDOFF_REQUEST,
        'original_CPU_request':REQUEST,'original_CPU_step':'1103606.511','worker':report['worker_receipt'],
        'destination':str(output),'source_producer_bytes_reserialized':False,
        'initial_trainables_read_or_authorized':False,'actual_new_encodes_or_model_updates':0,'training_READY':False}
    with (root/'native-input-cpu-handoff.json').open('x',encoding='utf8') as f:
        json.dump(receipt,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    return receipt


def main():
    start=time.monotonic();p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--independent-report',type=Path,required=True)
    p.add_argument('--install-target',action='store_true')
    p.add_argument('--plan-output',type=Path)
    args=p.parse_args();raw=stable_read(args.independent_report,REPORT);report=validated_report(raw)
    if args.install_target:
        need(os.name=='posix','Target installation is POSIX-only; no Windows/Linux path substitution')
        source=Path(str(SOURCE));root=Path(str(DESTINATION.parent))
        need(source.resolve(strict=True)==source and root.resolve(strict=True)==root and root.is_dir(),'Canonical existing target scope required')
        producer=stable_read(source,WORKER);need(time.monotonic()-start<10,'Metadata handoff budget elapsed')
        result=exclusive_copy(producer,root,report)
    else:
        result=plan_from_report(raw)
        if args.plan_output:
            with args.plan_output.open('x',encoding='utf8') as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
