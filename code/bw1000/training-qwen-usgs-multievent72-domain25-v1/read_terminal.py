"""Exact original metadata only; no Slurm refresh, data, PT body or model import."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import signal
import time

LABEL='training-qwen-usgs-multievent72-domain25-v1'
REQUEST='c933ba7359f84e8b9b80ac40d2d30964'
ROOT=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')/LABEL


def consume(context):
    total=0; pins={}; started=time.monotonic()
    def need(ok,reason):
        if not ok: raise ValueError(reason)
    def load(name,expected=None):
        nonlocal total
        p=ROOT/name
        need(p.resolve(strict=True)==p and p.is_file(),'unaliasable original metadata required')
        before=p.stat()
        need(0<before.st_size<=2<<20 and total+before.st_size<=16<<20,'metadata byte bound')
        raw=p.read_bytes();after=p.stat()
        need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==
             (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'metadata changed')
        pin=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())
        need(expected is None or pin['SHA256']==expected,'metadata SHA differs')
        pins[name]=pin;total+=len(raw)
        return json.loads(raw)
    intent=load('intent.json')
    need(intent['request_id']==context.request==REQUEST and intent['publication']==context.publication and
         intent['epoch']==context.epoch and intent['deadline']==context.epoch+2400 and
         intent['source_manifest_SHA256']==context.source_manifest_sha256 and
         intent['claim_SHA256']==context.claim_sha256,'original request/source/clock differs')
    source=load('src/SOURCE_FILES.json',context.source_manifest_sha256)
    claim=load('src/CLAIM.json',context.claim_sha256)
    need(claim['READ_GO'] is True,'exact original terminal read must be adopted')
    for name,pin in source['files'].items():
        p=ROOT/'src'/name;raw=p.read_bytes();total+=len(raw)
        need(total<=16<<20 and len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256'],
             'original frozen source member differs')
    receiver,controller=load('receiver-outcome.json'),load('controller-outcome.json')
    result=dict(state='ACTUAL_PARTIAL_OR_FAILED',passed=False,request=REQUEST,
        publication=context.publication,planned_updates=25,unique_train_QA=72,physical_exposures=800,
        actual_updates_by_rank={},GPU_cards=8,formal17=False,Model_GO=False,effect_qualified=False,
        source_manifest_SHA256=context.source_manifest_sha256,claim_SHA256=context.claim_sha256)
    failure=controller.get('failure')
    if isinstance(failure,str):
        result['private_failure_reason']=failure[:256]
        result['failure_reason_SHA256']=hashlib.sha256(failure.encode()).hexdigest()
    success=(receiver['passed'] is True and receiver['normal_held_controller_reaped'] is True and
        receiver['controller_rc']==0 and controller['passed'] is True and controller['failure'] is None)
    # Never manufacture zero updates on an interrupted native path.
    ranks=[]; supervised=0; padded=0; initial_names=set(); zero_B=[]
    for rank in range(8):
        prefix=f'continuous/train/treasury-witness/rank-{rank:02d}/'
        if not (ROOT/prefix/'result.json').is_file():
            success=False;continue
        row=load(prefix+'result.json')
        count=len(row.get('steps',[]));result['actual_updates_by_rank'][str(rank)]=count
        need(row['rank']==rank and row['world_size']==8 and row['completed_global_step']==25 and count==25 and
             len(row['inputs'])==len(row['outputs'])==100 and row['saved_checkpoints']==[10,20,25],
             'full all-rank update/micro/checkpoint denominator missing')
        need([x['global_step'] for x in row['steps']]==list(range(1,26)) and
             all(x['optimizer_slot_count']==504 and x['optimizer_step_min']==x['optimizer_step_max']==x['global_step']
                 for x in row['steps']) and any(x['adapter_changed'] for x in row['steps']),
             'native positive optimizer-counter and parameter change not proven')
        need(all(math.isfinite(x['loss_scalar']) for x in row['outputs']),'nonfinite actual loss')
        init=load(prefix+'initialization.json')
        need(init['parameter_tensors']==504 and init['parameter_numel']==43646976 and
             init['global_step']==0 and init['optimizer_state_entries']==0 and init['RNG_restored'] is False,
             'original permitted trainable initialization differs')
        initial_names.add(init['named_parameters_SHA256'])
        zero_B.append(init.get('lora_B_all_zero') is True and init.get('lora_B_tensors')==252)
        for item in row['inputs']:
            tokens=item['tokens']
            need(type(tokens['shifted_supervised_tokens']) is int and tokens['shifted_supervised_tokens']>0 and
                 type(tokens['padded_input_tokens']) is int and tokens['padded_input_tokens']>0,'actual token accounting missing')
            supervised+=tokens['shifted_supervised_tokens'];padded+=tokens['padded_input_tokens']
        ranks.append(dict(rank=rank,updates=25,microbatches=100,parameter_changing_updates=sum(x['adapter_changed'] for x in row['steps'])))
    result['actual_updates']=25 if len(ranks)==8 else None
    if success:
        need(len(ranks)==8 and len(initial_names)==1,'eight-rank original initial values differ')
        for phase in ('native72','pre','gpu','post'):
            step=controller['stages'][phase];row=step['row']
            need(step['passed'] is True and step['terminal_visible'] is True and step['rc']==0 and
                 len(row)==7 and re.fullmatch(r'1103606\.\d+',row[0]) and row[1]==LABEL+'-'+phase and
                 row[2:4]==['COMPLETED','0:0'] and row[5]=='f13r1n10','actual original phase terminal differs')
            tres=dict(x.split('=',1) for x in row[6].split(','))
            need({k:v for k,v in tres.items() if k.startswith('gres/')}==({'gres/dcu':'8'} if phase=='gpu' else {}),
                 'phase device allocation differs')
        cpu=load('native72-result.json')
        need(cpu['state']=='NATIVE_QWEN_MULTIEVENT72_PASS' and
             cpu['rows']==cpu['labels_and_EOS_valid']==cpu['full_sequence_retained']==cpu['collator_rows']==72,
             'new processor qualification incomplete')
        before,after=load('before.json'),load('after.json')
        need(before['pci']==after['pci'] and len(before['pci'])==8 and before['boot_id']==after['boot_id'] and
             after['kfd_holder_count']==0 and after['full_fd_inventory_performed'] is True and
             after['owned_remaining']==0 and after['port_available'] is True,'actual post cleanup not complete')
        native=load('continuous/native-outcome.json')
        need(native['passed'] is True and native['rc']==0 and len(native['finished'])==8 and
             native['cleanup']['owned_processes_remaining']==0 and native['cleanup']['termination_needed'] is False,
             'native tree/exit receipts missing')
        for entry in native['finished']:
            name=Path(entry['path']).relative_to(ROOT).as_posix()
            need(pins[name]==dict(bytes=entry['bytes'],SHA256=entry['SHA256']),'rank receipt body pin differs')
        inventory={}
        for n in (10,20,25):
            item=load(f'continuous-checkpoint-{n}-files.json')
            need(item['global_step']==n and item['member_count']==len(item['files'])==14,'saved checkpoint14 missing')
            inventory[str(n)]=item['files']
        effect=load('effect/native-outcome.json')
        generation=load('effect/generated/generation-finalized.json')
        need(effect['passed'] is True and effect['rc']==0 and effect['completed']==72 and
             effect['cleanup']['owned_processes_remaining']==0 and effect['cleanup']['termination_needed'] is False and
             generation['passed'] is generation['all72sealed_before_gold'] is True and
             generation['all_prediction_files_closed'] is True and generation['dev_gold_read'] is False and
             generation['blind_read'] is False and generation['planned_per_arm']==36 and
             set(generation['arms'])=={'base','domain25'} and all(zero_B),
             'complete sealed dev36 metric inputs/public-base initial equivalence not proven')
        prediction_pins={}
        for arm,entry in generation['arms'].items():
            need(entry['completed']==36 and entry['missing']==0 and entry['file']==arm+'-predictions-private.jsonl',
                 'full36 prediction denominator differs')
            path=ROOT/'effect/generated'/entry['file']
            before=path.stat();h=hashlib.sha256()
            need(0<before.st_size<=2<<20 and total+before.st_size<=16<<20,'bounded prediction hash only')
            with path.open('rb') as f:
                for block in iter(lambda:f.read(1<<20),b''):h.update(block)
            after=path.stat();total+=before.st_size
            need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==
                 (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns) and
                 dict(bytes=before.st_size,SHA256=h.hexdigest())=={k:entry[k] for k in ('bytes','SHA256')},
                 'actual sealed prediction file differs')
            prediction_pins[entry['file']]=dict(bytes=before.st_size,SHA256=h.hexdigest())
        prediction_pins['generation-finalized.json']=pins['effect/generated/generation-finalized.json']
        result.update(state='QWEN_MULTIEVENT72_CONTINUOUS25_INDEPENDENT_METADATA_PASS',passed=True,
            actual_updates=25,actual_global_microbatches=800,ranks=ranks,initial_named_SHA256=next(iter(initial_names)),
            actual_shifted_supervised_tokens=supervised,actual_padded_input_tokens=padded,
            actual_native72_qualified=True,checkpoint25_files=inventory['25'],post_clean=True,
            all72sealed_before_gold=True,sealed_predictions=prediction_pins,
            initial_LoRA_B_all_zero=True,metric_gold_read=False,allocated_DCU=8,visible_inference_DCU=1)
    result.update(actual_read_bytes=total,metadata_files=len(pins),wall_seconds=time.monotonic()-started,
        metadata_manifest_SHA256=hashlib.sha256(json.dumps(pins,sort_keys=True).encode()).hexdigest(),
        dev_or_blind_read=False,new_model_or_processor_calls=0)
    need(len(json.dumps(result))<=16384,'bounded public summary exceeded')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('request','publication','source-manifest-sha256','claim-sha256'):p.add_argument('--'+name,required=True)
    p.add_argument('--epoch',type=float,required=True)
    a=p.parse_args();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('read40s')));signal.alarm(40)
    print(json.dumps(consume(a),sort_keys=True,allow_nan=False),flush=True)
