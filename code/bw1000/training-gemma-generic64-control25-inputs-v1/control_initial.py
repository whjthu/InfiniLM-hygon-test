"""Restricted same-FD reuse of target-created step-zero values/RNG, never A100 state."""
import hashlib
import json
import os
from pathlib import Path

from control_contract import need,identity,FRESH_NAMED_SHA,TRAINING_REQUEST,DOMAIN


def bounded_load(torch,path,expected,check):
    path=Path(path);before=path.lstat()
    need(path.is_file() and not path.is_symlink() and 0<before.st_size<=128*1024**2 and
         before.st_size==expected['bytes'],'bounded target-created state file required')
    with path.open('rb') as f:
        need(identity(os.fstat(f.fileno()))==identity(before),'original state opened a different lifetime')
        h=hashlib.sha256();count=0
        while True:
            check();raw=f.read(min(1<<20,before.st_size-count+1))
            if not raw:break
            count+=len(raw);need(count<=before.st_size,'source state grew beyond frozen length');h.update(raw)
        need(count==before.st_size and h.hexdigest()==expected['SHA256'],'target-created state SHA differs')
        f.seek(0)
        value=torch.load(f,map_location='cpu',weights_only=True)
        need(identity(os.fstat(f.fileno()))==identity(before),'state changed during restricted deserialization')
    need(identity(path.lstat())==identity(before),'state pathname changed during load')
    return value


def restore_target_initial(config,trainables,local,witness):
    import random
    import numpy as np
    import torch
    from vlm.runtime.first_update_capture import cpu
    from vlm.runtime.first_update_audit import fingerprint
    from vlm.runtime.resume_witness import rng_fingerprint,rng_values
    need(config.get('initialization_mode')=='TARGET_CREATED_DOMAIN_INITIAL_REUSE' and
         config.get('seed')==config.get('data_seed')==42,'explicit target-created control initialization required')
    stage_path=Path(config['control_stage'])
    raw=stage_path.read_bytes()
    need(hashlib.sha256(raw).hexdigest()==config['control_stage_SHA256'],'actual control-stage pin differs')
    stage=json.loads(raw)
    need(stage['passed'] is True and stage['training_request']==TRAINING_REQUEST and
         stage['view']['SHA256']==config['dataset_SHA256'] and
         stage['reference_initial']['named_SHA256']==FRESH_NAMED_SHA and
         stage['original_A100_initial_read'] is False,'same target-created source stage required')
    need(not witness.trainer.optimizer.state_dict()['state'] and witness.trainer.state.global_step==0,
         'target initial reuse requires a genuinely fresh optimizer at zero')
    if witness.trainer.args.seed!=42 or witness.trainer.args.data_seed!=42:
        raise ValueError('control seed/data seed differs')
    root=DOMAIN/'continuous/train/continuous-witness'
    original=stage['reference_initial']
    values=bounded_load(torch,root/'rank-00/target-fresh-initial.pt',original['parameters'],witness.check_time)
    need(set(values)=={'schema','seed','named_SHA256','parameters'} and
         values['schema']=='GEMMA_TARGET_FRESH_INITIAL_V1' and values['seed']==42 and
         values['named_SHA256']==FRESH_NAMED_SHA,'original target-created schema differs')
    parameters=values['parameters'];actual=dict(trainables)
    need(len(actual)==len(parameters)==476 and set(actual)==set(parameters) and
         sum(p.numel() for p in parameters.values())==29802496,'exact target initial476 parameter population required')
    for name,destination in trainables:
        source=parameters[name]
        need(isinstance(source,torch.Tensor) and source.device.type=='cpu' and
             source.shape==destination.shape and source.dtype==destination.dtype and
             bool(torch.isfinite(source).all()),'target initial name/shape/dtype/finite mismatch')
    need(fingerprint(parameters)['SHA256']==FRESH_NAMED_SHA,'actual source named values differ')
    with torch.no_grad():
        for name,destination in trainables:destination.copy_(parameters[name])
    named=fingerprint(cpu({name:value for name,value in trainables}))
    need(named['SHA256']==FRESH_NAMED_SHA,'actual destination initial values differ')
    records=[None]*8
    torch.distributed.all_gather_object(records,dict(rank=local,named_SHA256=named['SHA256']))
    need([r['rank'] for r in records]==list(range(8)) and
         {r['named_SHA256'] for r in records}=={FRESH_NAMED_SHA},'actual initial8-rank values differ')
    rng=bounded_load(torch,root/f'rank-{local:02d}/target-fresh-rng.pt',original['rng'][str(local)]['file'],witness.check_time)
    need(set(rng)=={'python','cpu','cuda_local','local_rank','numpy'} and rng['local_rank']==local,
         'exact original per-rank RNG schema required')
    state=rng['numpy']
    need(set(state)=={'bit_generator','keys','position','has_gauss','cached_gaussian'},'safe numpy RNG schema differs')
    random.setstate(rng['python'])
    np.random.set_state((state['bit_generator'],np.asarray(state['keys'],dtype=np.uint32),
                        state['position'],state['has_gauss'],state['cached_gaussian']))
    torch.set_rng_state(rng['cpu'])
    torch.cuda.set_rng_state(rng['cuda_local'],local)
    observed=rng_fingerprint(rng_values(lambda:[torch.cuda.get_rng_state(local)]))
    need(observed==original['rng'][str(local)]['RNG_local'],'actual restored local RNG differs from domain source')
    return dict(mode='TARGET_CREATED_DOMAIN_INITIAL_REUSE',rank=local,seed=42,data_seed=42,
        named=named,target_reference_read=True,original_A100_initial_read=False,
        original_A100_value_equivalence=False,RNG_local=observed,RNG_equal_domain_reference=True,
        target_created=True,parameters_equal_domain_reference=True,optimizer_state_entries=0,
        checkpoint_resume=False,source_named_SHA256=FRESH_NAMED_SHA)
