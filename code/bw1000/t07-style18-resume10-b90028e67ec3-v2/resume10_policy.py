"""Frozen Style18 science identity and independent resume10 record review."""
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import re
import stat
import time


REQUEST='b90028e67ec347cd8c3beb72fc0c4aa1'
LABEL='t07-style18-resume10-'+REQUEST[:12]+'-v2'
OLD=Path('/fusion-run')
WORK=Path('/resume10-work')
PACKAGE=Path('/resume10-src')
SOURCE='811385e18f134998dc144ebc767e94b74937a76c12e7214152d32aa7aa1c002b'
CLAIM='0bc1cd6ff9d1dc364d3530e02b332d6b1d9533056fbb1ebdde71d74a38601a73'
ORIGINAL_FULL_READ='bf9140d28259cdbc8194826011bb283a6ecc15d25290f2ad00111e0a1d1f0b49'
ORIGINAL_OUT='3f7f1c7866e406997507d648f5edd888c39e28cf74da5a464395fe697a3d0152'
CP10_PIN=dict(bytes=60423969,SHA256='bf9defe7d44110c7c96b1746277509f22e123e4008071544ce3b67dc1688c43e')
ORIGINALS={
    'checkpoint-receipts.json':'38ac393ae954092fc80f1a41e5664297429e41299cb79b2bb4e7e09e3534d5f8',
    'optimizer-journal.jsonl':'17acfb499f7f2afc1e3e070360bff31aed05b91b7ce7f955f16e0198378682d4',
    'steps.jsonl':'8b7df504e769289ef9dd98e88e8ccda39f64e9a7ab9e362c6a5c50ada0e42f7d',
    'worker-result.json':ORIGINAL_OUT,
}
STEPS=tuple(range(11,26))
STATE_KEYS=('adapter_SHA256','state_SHA256','optimizer_SHA256','diffusion_scheduler_SHA256','rng_SHA256','integrity_SHA256')


def need(flag,code):
    if not flag:
        raise ValueError(code)


def pin(raw):return dict(bytes=len(raw),SHA256=sha256(raw).hexdigest())


def write(path,value):
    body=(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
    with path.open('xb') as stream:
        stream.write(body);stream.flush();os.fsync(stream.fileno())
    return pin(body)


def identity(st):return (st.st_dev,st.st_ino,st.st_size,st.st_mode,st.st_mtime_ns,st.st_ctime_ns)


def exact(path,cap,sha=None):
    need(path.resolve(strict=True)==path,'ORIGINAL_PATH_MUST_BE_CANONICAL')
    before=path.lstat()
    with path.open('rb') as stream:
        actual=os.fstat(stream.fileno())
        same=lambda a,b: identity(a)==identity(b) if os.name!='nt' else identity(a)[:-1]==identity(b)[:-1]
        need(same(before,actual) and stat.S_ISREG(before.st_mode) and 0<actual.st_size<=cap,'BOUNDED_PINNED_INPUT')
        raw=stream.read(cap+1)
        need(same(actual,os.fstat(stream.fileno())) and identity(before)==identity(path.lstat())
             and len(raw)==before.st_size and (sha is None or pin(raw)['SHA256']==sha),'EXACT_STABLE_ORIGINAL_INPUT')
    return raw


def journal(path,expected,sha=None):
    raw=exact(path,65536,sha)
    records=[json.loads(line) for line in raw.splitlines()]
    desired=[dict(event=event,global_step=step) for step in expected for event in ('optimizer_intent','optimizer_returned')]
    need(records==desired,'EXACT_OPTIMIZER_INTENT_RETURNED_SEQUENCE')
    return pin(raw)


def original_contract(original,recipe):
    original_raw=exact(original/'continuous/contract.json',131072)
    actual=json.loads(original_raw)
    need(actual==recipe,'ORIGINAL_1139_CHECKPOINT_RECIPE_OBJECT_DIFFERS')
    return dict(original_contract_pin=pin(original_raw),reconstructed_contract_pin=pin(
        (json.dumps(recipe,indent=2,sort_keys=True,allow_nan=False)).encode()))


def original_reference(original):
    exact(original/'SOURCE_FILES.json',16384,SOURCE)
    original_claim=json.loads(exact(original/'CLAIM.json',16384,CLAIM))
    need(original_claim['request']=='1d26c9a5ed994b06b6762359c2902606'
         and original_claim['parent']=='1087656' and original_claim['node']=='f11r4n13',
         'ACTUAL_1139_FROZEN_RECIPE_CLAIM')
    folder=original/'continuous'
    records={key:exact(folder/key,65536,v) for key,v in ORIGINALS.items()}
    worker=json.loads(records['worker-result.json'])
    need(worker['passed'] is True and worker['step']=='1087656.1139'
         and worker['optimizer_intended_steps']==worker['optimizer_returned_steps']==list(range(1,26)),
         'ACTUAL_ORIGINAL_1139_COMPLETED_STEPS')
    journal(folder/'optimizer-journal.jsonl',range(1,26),ORIGINALS['optimizer-journal.jsonl'])
    cps=json.loads(records['checkpoint-receipts.json'])
    steps=[json.loads(line) for line in records['steps.jsonl'].splitlines()]
    need([x['global_step'] for x in cps]==[10,20,25] and [x['next_data_cursor'] for x in cps]==[10,8,1]
         and [x['global_step'] for x in steps]==list(range(1,26))
         and all(x['ordinal']==(x['global_step']-1)%12 and isfinite(x['loss']) for x in steps)
         and cps[0]['bytes']==CP10_PIN['bytes'] and cps[0]['checkpoint_SHA256']==CP10_PIN['SHA256'],
         'ORIGINAL_25_RECORDS_AND_10_20_25_CHECKPOINTS')
    return original_claim,cps,steps


def prepare_checkpoint(original,recipe,deadline):
    _,checkpoints,_=original_reference(original)
    recipe_pins=original_contract(original,recipe)
    path=original/'continuous/checkpoint-10.private.pt'
    need(path.resolve(strict=True)==path,'EXACT_ORIGINAL_CP10_PATH')
    fd=os.open(path,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW|os.O_NONBLOCK)
    data=sha256()
    count=0
    try:
        before=os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and before.st_size==CP10_PIN['bytes']
             and os.fstatvfs(fd).f_flag & os.ST_RDONLY and identity(before)==identity(path.lstat()),
             'EXACT_CP10_EFF_RO_STABLE_BOUND_DESCRIPTOR')
        while chunk:=os.read(fd,4<<20):
            deadline()
            count+=len(chunk)
            need(count<=CP10_PIN['bytes'],'BOUNDED_CP10_EXTENT')
            data.update(chunk)
        need(dict(bytes=count,SHA256=data.hexdigest())==CP10_PIN
             and identity(before)==identity(os.fstat(fd))==identity(path.lstat()),'CURRENT_ACTUAL_CP10_FULL_SHA')
        # A pathname API must still open precisely this held, verified inode.
        os.lseek(fd,0,os.SEEK_SET)
        return fd,before,checkpoints[0],recipe_pins
    except BaseException:
        os.close(fd)
        raise


def confirm_checkpoint(fd,info,original):
    need(identity(info)==identity(os.fstat(fd))==identity((original/'continuous/checkpoint-10.private.pt').lstat()),
         'ORIGINAL_PINNED_CP10_CHANGED_DURING_PROVIDER_LOAD')


def mount_only():
    for location in (OLD,PACKAGE):
        need(location.resolve(strict=True)==location and os.statvfs(location).f_flag & os.ST_RDONLY,
             'ORIGINAL_AND_NEW_SOURCE_ARE_EFFECTIVELY_READONLY')
    need(WORK.resolve(strict=True)==WORK and not os.statvfs(WORK).f_flag & os.ST_RDONLY,
         'NEW_PRIVATE_WORK_IS_WRITABLE')


def rehash_output(path,expected,deadline):
    need(path.resolve(strict=True)==path,'EXACT_NEW_CHECKPOINT_OUTPUT')
    digest=sha256()
    count=0
    before=path.lstat()
    with path.open('rb') as stream:
        opened=os.fstat(stream.fileno())
        need(stat.S_ISREG(opened.st_mode) and opened.st_size==before.st_size==expected['bytes']
             and 0<opened.st_size<=256<<20,'BOUNDED_STORED_NATIVE_CHECKPOINT')
        while body:=stream.read(4<<20):
            count+=len(body)
            need(count<=opened.st_size and time.monotonic()<deadline,'INDEPENDENT_CP20_25_REHASH_DEADLINE')
            digest.update(body)
        need(identity(os.fstat(stream.fileno()))==identity(opened) and identity(path.lstat())==identity(before)
             and count==opened.st_size and digest.hexdigest()==expected['checkpoint_SHA256'],
             'EXACT_INDEPENDENT_CP20_CP25_STORED_FILE_DIGEST')
    return dict(bytes=count,SHA256=digest.hexdigest())


def compare(original,out):
    _,baseline,source_steps=original_reference(original)
    restored=json.loads(exact(out/'restore.json',65536))
    user_steps=[json.loads(line) for line in exact(out/'steps.jsonl',65536).splitlines()]
    new_cps=json.loads(exact(out/'checkpoint-receipts.json',65536))
    resumed_journal=journal(out/'optimizer-journal.jsonl',STEPS)
    need(restored['restored_boundary_exact'] is True and restored['metadata']['global_step']==10
         and restored['next']['global_step']==11 and restored['next']['sample_cursor']==10
         and restored['checkpoint_SHA256']==CP10_PIN['SHA256']
         and [x['global_step'] for x in user_steps]==list(STEPS)
         and [x['global_step'] for x in new_cps]==[20,25] and [x['next_data_cursor'] for x in new_cps]==[8,1],
         'EXACT_NEW_RESUME10_BOUNDARY_15_STEPS_AND_20_25_CKPTS')
    need(original_contract(original,json.loads(exact(out/'contract.json',131072)))==restored['recipe_pins'],
         'ORIGINAL_RECONSTRUCTED_CONTRACT_DID_NOT_SURVIVE')
    precisions=[]
    operands=[]
    for expected,actual in zip(source_steps[10:],user_steps,strict=True):
        need(expected['global_step']==actual['global_step'] and expected['ordinal']==actual['ordinal']
             and isfinite(actual['loss']) and isfinite(actual['preclip_norm']) and actual['preclip_norm']>0,
             'RESUMED_NATIVE_STEP_PHYSICAL_AND_INPUT_DENOMINATOR')
        precisions.append(abs(expected['loss']-actual['loss']))
        operands.append(expected['operands']==actual['operands'])
    states={str(left['global_step']):{name:left[name]==right[name] for name in STATE_KEYS}
            for left,right in zip(baseline[1:],new_cps,strict=True)}
    deadline=time.monotonic()+30
    storage={str(row['global_step']):rehash_output(out/f"checkpoint-{row['global_step']:02d}.private.pt",row,deadline)
             for row in new_cps}
    return dict(planned=15,verified=15,paired_steps=list(STEPS),first_cursor=10,
        paired_native_operands=sum(operands),max_native_loss_difference=max(precisions),
        native_state_matches=states,resumed_optimizer_journal=resumed_journal,actual_output_checkpoint_pins=storage,
        passed=all(operands) and max(precisions)<=1e-6 and all(all(x.values()) for x in states.values()),
        original_new_updates=0,heldout_used=False,formal_A100=False,formal_performance=False)
