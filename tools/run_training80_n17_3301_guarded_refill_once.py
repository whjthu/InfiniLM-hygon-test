"""Bind the proved .3301 natural return and prepare one guarded n17 successor."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import runpy
import shlex
import subprocess
import time
from types import FunctionType
from uuid import UUID, uuid5


ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/'local-only'
STEM='training80-n17-3301-guarded-bf16-v1'
WRAPPER=ROOT/'tools/run_training80_n17_3301_guarded_refill_once.py'
CLAIM=ROOT/'coordination/tasks'/f'{STEM}.json'
DRAFT=LOCAL/f'{STEM}.draft.json'
SPECIAL=ROOT/'tools/run_training80_r27_six_specialist_guarded_refill_once.py'
SPECIAL_SHA='9f941399241fb83d968217e0e0e13df0683a0464bd169bd3c8daa6f787210a18'
R25=LOCAL/'read-training80-r25-next-incumbents-once.py'
R25_SHA='0462a5247bb78a0b4f61095d3be8bf1f66c06d73339043654dbc135f68f4a789'
TERMINAL=LOCAL/'training80-r27-six-specialist-guarded-bf16-v1-f13r1n17-independent-e73ba41abd0d4b2cbeb2c6743398dab3.json'
TERMINAL_SHA='28effdb7f4f11f4fdf97b5a22a466ed7e03bc104ec56867c413d4fa20bdf89f8'
COUNTERS={
    'training80-r27-guarded-five-actual-numbered-and-counter-1e75b76dbe364dc08753eec81b246216.json':'ae0af8a21ff4b710646e949550a9f6b051d49813c63fb3666844e48ac0506caf',
    'training80-r27-guarded-five-actual-numbered-and-counter-3c9901eacb8a4f5dae04d0d6a4dc2d03.json':'3fdd14c1eee035e26f3e6b9f2a3e25ef10516f0f2c2e8374b9cbaee98091568f',
}
NODE,PARENT,PREVIOUS,CLEAR='f13r1n17','1103606','1103606.3301','1103606.3300'
OLD='aac14e185d1e46819f4dfce92cba8bd8'
REQUEST=uuid5(UUID('aa3a87cb-5615-42ba-9d4b-643d298276f6'),f'{TERMINAL_SHA}/{NODE}/{PREVIOUS}').hex
OLD_AUDIT=f'training80-r27-six-specialist-guarded-bf16-v1-{NODE}-{OLD}'
HOST_READ=r'''
def qualification(x):
    original=group(x)
    folder=base/('training80-dcu-standby-'+x['request'])/x['node']
    worker,worker_pin=read(folder/'worker.json')
    item,cpin=read(folder/'worker-cgroup.json')
    submitted,submission_pin=read(folder.parent/'submission-receipt.json')
    guard,guard_pin=read(base/ORIGINAL_SCOPE/'outcome.json')
    detail=[line.split('|') for line in query('sacct','-nP','-j',OLD_CPU,'-o',fmt)]
    clear=[r for r in detail if len(r)==9 and r[0]==OLD_CPU]
    cpu=clear[0] if len(clear)==1 else None
    cgroup_ok=bool(item and worker and re.fullmatch('[a-f0-9]{64}',cpin or '') and item['step']==worker['step']==x['previous']
                   and item['node']==worker['node']==x['node'] and item['same_UID'] is item['exact_job_and_step_cgroup'] is True
                   and item['worker_pid']==worker['worker_pid']==x['previous_worker_pid'] and item['worker_start_ticks']>0)
    good=bool(original['original_terminal_and_targets_empty'] is True and original['terminal_SHA256']==EXPECTED_RETURN
              and original['accounting'][:4]+original['accounting'][5:7]+original['accounting'][8:]==OLD_GPU_ROW
              and worker_pin==x['worker_SHA256'] and cgroup_ok and submission_pin==EXPECTED_SUBMISSION
              and submitted and submitted['request']==x['request'] and submitted['synthetic'] is True
              and guard_pin==EXPECTED_GUARD and guard==OLD_GUARD
              and cpu and cpu[:4]+cpu[5:7]+cpu[8:]==OLD_CPU_ROW)
    original.update(proved_terminal_with_cgroup=good,worker_cgroup_SHA256=cpin if cgroup_ok else None,
                    original_zero_GRES_accounting=cpu,submit_SHA256=submission_pin,old_guard_SHA256=guard_pin)
    return original
'''


def dig(raw):
    return sha256(raw).hexdigest()


def require(ok,reason):
    if not ok:
        raise ValueError(reason)


def pin(path,sha=None):
    require(path.resolve(strict=True)==path and path.is_relative_to(ROOT),'source or receipt path differs')
    raw=path.read_bytes()
    require(0<len(raw)<=262144 and (sha is None or dig(raw)==sha),'source or receipt bytes differ')
    return raw


def replace(text,old,new):
    require(text.count(old)==1,'frozen guard boundary differs')
    return text.replace(old,new,1)


def inputs():
    pin(SPECIAL,SPECIAL_SHA);pin(R25,R25_SHA)
    specialized=runpy.run_path(str(SPECIAL));read=runpy.run_path(str(R25))
    base,config,claim,_,prior,clearance,legacy=specialized['load']()
    wrapper=json.loads(pin(TERMINAL,TERMINAL_SHA));out=wrapper['observed']
    end,terminal_sha=out['worker_terminal'];guard,guard_sha=out['original_guard_outcome']
    require(wrapper['rc']==0 and wrapper['request']==out['request']==guard['request']==OLD and out['node']==guard['node']==NODE
            and out['state']=='EXACT_TERMINAL_ACCOUNTING' and out['step']==end['step']==PREVIOUS and end['state']=='WORKER_RETURNED'
            and out['accounting']==[PREVIOUS,'training80-dcu-standby-'+OLD[:12]+'-'+NODE,'COMPLETED','0:0',NODE,
                                   'cpu=8,gres/dcu=8,mem=16G,node=1','2026-11-25T09:04:39']
            and end['node']==NODE and not out['new_scheduling_operations'] and guard['GPU_srun_attempted'] is True
            and guard['fresh_zero_GRES_accounting'][:6]==[CLEAR,'training80-g4-fresh-'+OLD[:12],'COMPLETED','0:0',NODE,
                                                        'cpu=1,mem=256M,node=1']
            and guard['state']=='SRUN_PROCESS_STARTED_AWAIT_SCHEDULER_ACCOUNTING' and guard['receipt']['worker_seconds']==10603,
            'exact .3301/.3300 terminal evidence differs')
    views=[]
    for name,sha in COUNTERS.items():
        receipt=json.loads(pin(LOCAL/name,sha));v=receipt['result']['nodes'][NODE]
        require(receipt['rc']==0 and v['request']==OLD and v['previous']==PREVIOUS and v['parent']==PARENT
                and v['worker_seconds']==10603 and v['worker_PID']>0 and v['status']=='RUNNING_WITH_ORIGINAL_BUDGET'
                and all(re.fullmatch(r'[a-f0-9]{64}',v[key]) for key in ('worker_SHA256','expected_SHA256','submit_SHA256')),
                'old numbered worker proof differs')
        views.append(v)
    earlier,later=views
    pci=sorted(earlier['second'])
    require(len(set(pci))==8 and set(later['second'])==set(pci)
            and all(later['second'][dev]['calls']>earlier['second'][dev]['calls'] and later['second'][dev]['epoch']>earlier['second'][dev]['epoch'] for dev in pci)
            and all(earlier[k]==later[k] for k in ('worker_SHA256','expected_SHA256','submit_SHA256','worker_start_epoch','worker_PID'))
            and later['worker_start_epoch']+10603<=end['epoch'] and claim['targets'][NODE]['request']==OLD,
            'old n17 did not finish within matched worker identity')
    row=dict(node=NODE,parent=PARENT,previous=PREVIOUS,request=OLD,worker_SHA256=later['worker_SHA256'],expected_SHA256=later['expected_SHA256'],
             submit_SHA256=later['submit_SHA256'],previous_worker_pid=later['worker_PID'],worker_start_epoch=later['worker_start_epoch'],
             worker_seconds=10603,planned_end_epoch=later['worker_start_epoch']+10603,pci=pci,cgroup_SHA256=None)
    return specialized,read,base,config,claim,prior,clearance,legacy,row,out,terminal_sha,guard_sha


def expected_claim(material):
    row=material[8]
    return dict(scope=STEM,state='PREPARED_ONLY_LOCAL_CGROUP_PROOF_AND_FRESH_DEVICE_REQUIRED',node=NODE,parent=PARENT,previous_GPU=PREVIOUS,
                original_zero_GRES=CLEAR,previous_request=OLD,request=REQUEST,previous_worker_SHA256=row['worker_SHA256'],
                previous_expected_SHA256=row['expected_SHA256'],previous_submit_SHA256=row['submit_SHA256'],
                previous_observer_SHA256=TERMINAL_SHA,previous_terminal_SHA256=material[10],same_worker_counter_receipts=COUNTERS,
                GPU=8,Slurm_cap_seconds=11100,max_invocations=1,launcher=WRAPPER.relative_to(ROOT).as_posix(),
                replay_allowed=False,ledger_mutations=0)


def render_read(material,nonce):
    require(re.fullmatch(r'[a-f0-9]{32}',nonce) and int(nonce,16)>0 and nonce not in (OLD,REQUEST),'new exact read ID required')
    _,read,*_=material
    row,out,terminal_sha,guard_sha=material[8:]
    bind=dict(OLD_GPU_ROW=out['accounting'],OLD_CPU_ROW=out['original_guard_outcome'][0]['fresh_zero_GRES_accounting'],
              OLD_GUARD=out['original_guard_outcome'][0],OLD_CPU=CLEAR,EXPECTED_RETURN=terminal_sha,
              EXPECTED_SUBMISSION=row['submit_SHA256'],EXPECTED_GUARD=guard_sha,ORIGINAL_SCOPE=OLD_AUDIT)
    source="import os\nos.environ.update(TZ='UTC', SLURM_TIME_FORMAT='%Y-%m-%dT%H:%M:%S')\n"
    source+=''.join(f'{k}={v!r}\n' for k,v in bind.items())+read['render']([row],nonce)
    source=replace(source,'    assert path.resolve(strict=True)==path\n','    assert path.resolve(strict=True)==path and path.stat().st_uid==os.getuid()==154459\n')
    source=replace(source,'print(json.dumps(dict(read_id=',HOST_READ+'\nprint(json.dumps(dict(read_id=')
    source=replace(source,'targets=[group(item) for item in items]','targets=[qualification(item) for item in items]')
    compile(source,'<n17 read-only original worker cgroup evidence>','exec')
    return source


def read_path(nonce):
    require(re.fullmatch(r'[a-f0-9]{32}',nonce),'invalid exact read nonce')
    return LOCAL/f'{STEM}-read-{nonce}.json'


def read_original(material,nonce):
    source=render_read(material,nonce).encode()
    helper=runpy.run_path(str(material[3][0]['HELPER']))
    command=shlex.join(['/usr/local/python3.12/bin/python3','-I','-S','-B','-'])
    path=read_path(nonce)
    with path.open('x',encoding='utf8') as stream:
        t0=time.monotonic();anchor=time.time()-t0
        try:
            reply=subprocess.run([*helper['ssh'](ROOT.parent),command],input=source,capture_output=True,timeout=45)
            rc,out,err=reply.returncode,reply.stdout,reply.stderr
        except subprocess.TimeoutExpired as e:
            rc,out,err=None,e.stdout or b'',e.stderr or b''
        except (OSError,ValueError,AssertionError,StopIteration,subprocess.SubprocessError) as e:
            rc,out,err=None,b'',str(e).encode()
        at=time.monotonic(); found=None
        if rc==0 and not err and 0<len(out)<=16384:
            try:
                body=json.loads(out)
                [target]=body['targets']
                if body['read_id']==nonce and body['Slurm_mutations']==body['physical_admissions']==0 and (target['node'],target['step'],target['request'])==(NODE,PREVIOUS,OLD):found=target
            except (ValueError,TypeError,KeyError):
                pass
        json.dump(dict(rc=rc,observed=found,read_id=nonce,source_SHA256=dig(source),boot_epoch_anchor=anchor,received_monotonic=at,
                       RTT=at-t0,stdout_SHA256=dig(out),stderr_SHA256=dig(err),replay_allowed=False),stream,sort_keys=True,allow_nan=False)
    return dict(file=path.name,SHA256=dig(pin(path)),qualified=bool(found and found.get('proved_terminal_with_cgroup')),GPU=0)


def checked(material,nonce,proof,*,fresh):
    observed=proof['observed'];row,out,terminal_sha,guard_sha=material[8:]
    require(proof['rc']==0 and proof['read_id']==nonce and proof['replay_allowed'] is False and 0<=proof['RTT']<=45
            and proof['stderr_SHA256']==dig(b'') and isinstance(observed,dict)
            and proof['source_SHA256']==dig(render_read(material,nonce).encode()) and observed['proved_terminal_with_cgroup'] is True
            and observed['node']==NODE and observed['step']==PREVIOUS and observed['request']==OLD
            and observed['scontrol_target_steps']==observed['squeue_target_steps']==[]
            and observed['worker_SHA256']==row['worker_SHA256'] and observed['expected_SHA256']==row['expected_SHA256']
            and observed['terminal_SHA256']==terminal_sha and observed['submit_SHA256']==row['submit_SHA256']
            and observed['old_guard_SHA256']==guard_sha and re.fullmatch(r'[a-f0-9]{64}',observed['worker_cgroup_SHA256'] or ''),
            'exact original C0/worker cgroup and current two empty numeric views absent')
    gpu,cpu=observed['accounting'],observed['original_zero_GRES_accounting']
    require(gpu[:4]+gpu[5:7]+gpu[8:]==out['accounting'] and int(gpu[4])>=row['worker_seconds']
            and cpu[:4]+cpu[5:7]+cpu[8:]==out['original_guard_outcome'][0]['fresh_zero_GRES_accounting'],
            'original GPU or zero-GRES actual C0 and End differ')
    if fresh:
        now=time.monotonic()
        require(0<=now-proof['received_monotonic']<=45 and abs(time.time()-now-proof['boot_epoch_anchor'])<2,'old terminal read aged')
    return gpu,cpu,[row['worker_SHA256'],observed['worker_cgroup_SHA256'],row['expected_SHA256']]


def make_guard(material,evidence,revision):
    old,_,base,config,claim,prior,clearance,legacy,row,out,terminal_sha,guard_sha=material
    gpu,cpu,pins=evidence
    spec=dict(claim['targets'][NODE],request=REQUEST,original_GPU=PREVIOUS,original_zero_GRES=CLEAR,
              original_request=OLD,original_reply_SHA256=TERMINAL_SHA)
    bounded=FunctionType(old['bound'].__code__,dict(old['bound'].__globals__,STEM=STEM))
    render=FunctionType(old['render'].__code__,dict(old['render'].__globals__,bound=bounded))
    targets=dict(targets={NODE:spec})
    updated=dict(accounting=gpu,file_pins=pins,original_PCI=row['pci'],original_return_SHA256=terminal_sha)
    result=render(base,config,targets,{NODE:updated},{NODE:cpu},NODE,revision)
    source=result['launch']
    gate=(f"            for rel, expected_sha in {dict([(OLD_AUDIT+'/outcome.json',guard_sha),('training80-dcu-standby-'+OLD+'/submission-receipt.json',row['submit_SHA256'])])!r}.items():\n"
          "                p = RESULTS / rel\n"
          "                require(p.resolve(strict=True)==p and p.stat().st_uid==os.getuid(), 'original launch evidence scope changed')\n"
          "                with p.open('rb') as original_handle:\n"
          "                    original_bytes=original_handle.read(16385)\n"
          "                require(0<len(original_bytes)<=16384 and sha256(original_bytes).hexdigest()==expected_sha, 'original launch evidence changed')\n")
    result['launch']=replace(source,'            check_previous(account(PREVIOUS))',gate+'            check_previous(account(PREVIOUS))')
    compile(result['launch'],'<n17 strict .3301 genuine successor>', 'exec')
    return result,bounded,spec


def paths():
    return tuple(LOCAL/f'{STEM}-{NODE}-{REQUEST}-{name}.json' for name in ('intent','send','reply'))


def publication(material,revision):
    old,_,base,config=material[:4]
    require(Path(__file__).resolve()==WRAPPER and re.fullmatch(r'[a-f0-9]{40}',revision),'exact installed successor publication required')
    base['verify_publication'](revision,config)
    git=['git','--no-optional-locks','-c','safe.directory='+ROOT.as_posix()]
    for path in (WRAPPER,CLAIM,SPECIAL,old['CLAIM']):
        ref=revision+':'+path.relative_to(ROOT).as_posix()
        require(subprocess.check_output([*git,'cat-file','blob',ref],cwd=ROOT,stderr=subprocess.PIPE,timeout=8)==pin(path),
                'published n17 or inherited authority bytes differ')


def dispatch(material,revision,receipt,claim_sha):
    old,_,base,config,_,prior,_,legacy=material[:8]
    intent,token,reply=paths()
    require(not any(os.path.lexists(p) for p in (intent,token,reply)),'fixed n17 request already reserved')
    publication(material,revision)
    old['require_no_slots'](base['JOURNAL'],NODE,legacy[NODE],(1,2))
    old['require_no_slots'](base['JOURNAL'],NODE,prior[NODE]['accounting'][0],(0,1,2))
    old['require_no_slots'](base['JOURNAL'],NODE,PREVIOUS,(0,1,2))
    require(NODE not in base['live_authority'](config) and NODE not in base['local_holds'](),'existing real business/ordinary priority blocks n17')
    nonce,sha=receipt
    proof=json.loads(pin(read_path(nonce),sha))
    evidence=checked(material,nonce,proof,fresh=True)
    program,_,_=make_guard(material,evidence,revision)
    helper=runpy.run_path(str(config[0]['HELPER']));sender=runpy.run_path(str(config[0]['TRANSPORT']));argv=helper['ssh'](ROOT.parent)
    require(0<=time.monotonic()-proof['received_monotonic']<=45,'old terminal evidence aged before reservation')
    record=dict(node=NODE,request=REQUEST,publication=revision,claim_SHA256=claim_sha,program_SHA256=dig(program['launch'].encode()),
                terminal_read_id=nonce,terminal_read_SHA256=sha,GPU_requested=8,replay_allowed=False)
    config[1]['save'](intent,record)
    done=sender['send'](argv,request=REQUEST,program=program['launch'],data=b'',send_token=token,authority_seconds=18,outcome_seconds=82)
    config[1]['save'](reply,dict(record,transport=done))
    return dict(receipt=reply.name,SHA256=dig(pin(reply)),state=(done.get('response') or {}).get('state'),replay_allowed=False)


def observe(material):
    intent,token,_=paths(); evidence=json.loads(pin(intent))
    require(token.is_file() and evidence['request']==REQUEST and evidence['node']==NODE,'only exact already-sent original can be read')
    nonce=evidence['terminal_read_id']; proof=json.loads(pin(read_path(nonce),evidence['terminal_read_SHA256']))
    guard,bounded,spec=make_guard(material,checked(material,nonce,proof,fresh=False),evidence['publication'])
    require(dig(guard['launch'].encode())==evidence['program_SHA256'],'original n17 payload identity differs')
    return bounded(material[2],'observe',NODE,spec)(NODE,guard,material[3])


def fixtures(material):
    old,_,base,conf,*_=material
    row,out,term,gs=material[8:]
    nonce='1'*32
    gpu=out['accounting'];cpu=out['original_guard_outcome'][0]['fresh_zero_GRES_accounting']
    def sample(acct,elapsed):return acct[:4]+[str(elapsed)]+acct[4:6]+[acct[6],acct[6]]
    point=dict(node=NODE,request=OLD,step=PREVIOUS,proved_terminal_with_cgroup=True,scontrol_target_steps=[],squeue_target_steps=[],
               worker_SHA256=row['worker_SHA256'],expected_SHA256=row['expected_SHA256'],terminal_SHA256=term,submit_SHA256=row['submit_SHA256'],
               old_guard_SHA256=gs,worker_cgroup_SHA256='f'*64,accounting=sample(gpu,10603),original_zero_GRES_accounting=sample(cpu,1))
    proof=dict(rc=0,read_id=nonce,replay_allowed=False,RTT=1,stderr_SHA256=dig(b''),
               source_SHA256=dig(render_read(material,nonce).encode()),observed=point)
    evidence=checked(material,nonce,proof,fresh=False)
    binary,*_=make_guard(material,evidence,None)
    old['offline_staging_fixture'](binary['launch'],conf[6])
    for delta in (dict(worker_cgroup_SHA256=None),dict(scontrol_target_steps=['new']),dict(terminal_SHA256='f'*64)):
        try:checked(material,nonce,dict(proof,observed=dict(point,**delta)),fresh=False)
        except (ValueError,TypeError):continue
        raise ValueError('n17 unsafe receipt accepted locally')
    return dict(request=REQUEST,worker_seconds=binary['worker_seconds'],compiled_host_SHA256=dig(binary['launch'].encode()),
                state='PREPARED_ONLY_LIVE_CGROUP_PROOF_REQUIRED',staging_positive=1,staging_vetoes=2,
                receipt_positive=1,receipt_vetoes=3,GPU=0,SCNet=0,Git=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true')
    mode.add_argument('--observe-predecessor')
    mode.add_argument('--publication')
    mode.add_argument('--observe-original',action='store_true')
    parser.add_argument('--terminal-receipt',nargs=2)
    args=parser.parse_args()
    require(bool(args.terminal_receipt)==bool(args.publication),'actual cgroup-bound terminal receipt mandatory')
    data=inputs(); raw=pin(DRAFT if args.check and not CLAIM.exists() else CLAIM)
    require(json.loads(raw)==expected_claim(data),'exact single n17 claim differs')
    if args.check: result=fixtures(data)
    elif args.observe_predecessor: result=read_original(data,args.observe_predecessor)
    elif args.observe_original: result=observe(data)
    else: result=dispatch(data,args.publication,args.terminal_receipt,dig(raw))
    print(json.dumps(dict(result,claim_SHA256=dig(raw)),sort_keys=True,allow_nan=False))


if __name__=='__main__':
    main()
