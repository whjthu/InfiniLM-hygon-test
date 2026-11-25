"""Guard a distinct n15 successor after the rejected probe was reconciled."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import runpy
from uuid import UUID, uuid5


ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/'local-only'
STEM='training80-r41-n15-warning-guarded-bf16-v1'
WRAPPER=ROOT/'tools/run_training80_r41_n15_warning_guarded_successor_once.py'
CLAIM=ROOT/'coordination/tasks'/f'{STEM}.json'
DRAFT=LOCAL/f'{STEM}.draft.json'
R39=('tools/run_training80_r39_last40_twenty_three_guarded_successor_once.py',
     '2fa7e7ec704fb2665bf8ea9cb43a07decafe1fd828ee43b21da399bb3efc05d5')
WARNING=('tools/run_training80_r36_two_warning_guarded_successor_once.py',
         '801531127d7351ca6e57dfab16c62ac6c1e3cd94ccce363ca8245de196831dd5')
LAST40=('tools/run_training80_r25_last40_ordinary_guarded_refill_once.py',
        '575ecd2d9d08f300e287a77c0b25fdeb3a6a48f2ed2135273dbe390bdd16c88e')
TERMINAL=('training80-r41-n15-warning-original-read-724f689f6f3f4ef4a1a0be5acbfa84a3.json',
          '98d49e4f09dcea9299d47ac4993836ee0356f676dfa6347b66124a6b20a52bef')
REJECTED='6a38bab31c105e2498a3845407faa4bd'
REJECTED_FILES={'intent':'1f19c44d6d68bb585740db769c2fee1b12e28ed66287d97200180d7fc38b58bc',
                'send':'cf63f4eb951557d4de8b0981cf4b9ccbe7ff7b90a15fed2fbc8b7180fb2116e3',
                'reply':'dc49b8020e8bab30603fd5dc7fb129796682d7f2d3ad45ffb98e4949c27e96b8'}
NODE='f13r4n15'
PREVIOUS='1194900.1351'
CPU='1194900.1412'
REQUEST=uuid5(UUID('8100462c-3b7f-4824-9fcd-50f30be384e4'),f'{PREVIOUS}/{REJECTED}/{TERMINAL[1]}').hex


def require(ok,reason):
    if not ok:
        raise ValueError(reason)


def digest(data):
    return sha256(data).hexdigest()


def pin(path,expected):
    require(path.is_relative_to(ROOT) and path.resolve(strict=True)==path,'exact original path changed')
    with path.open('rb') as stream:
        raw=stream.read(262145)
    require(0<len(raw)<=262144 and digest(raw)==expected,'exact original bytes changed')
    return raw


def module(spec):
    file,sha=spec
    pin(ROOT/file,sha)
    return runpy.run_path(str(ROOT/file))


def load():
    old=module(R39)
    warning=module(WARNING)
    historical,_,_=old['entry']()
    budget,first,r34,early,n02,context,rows,ancestry,_=historical()
    base=rows[NODE]
    receipt=json.loads(pin(LOCAL/TERMINAL[0],TERMINAL[1]))
    view=receipt['observed']
    [r]=view['targets']
    [cpu]=r['warning_CPU_rows']
    rejected_scope=f"{old['STEM']}-{NODE}-{REJECTED}"
    intent,send,reply=(json.loads(pin(LOCAL/f'{rejected_scope}-{kind}.json',REJECTED_FILES[kind])) for kind in ('intent','send','reply'))
    transport=reply['transport']
    out=transport['response']
    require(receipt['rc']==0 and receipt['Slurm_mutations']==receipt['physical_admissions']==view['Slurm_mutations']==view['physical_admissions']==0
            and r['qualified_original_release'] is r['qualified_rejected_probe_accounting'] is r['durable_no_send_matches'] is True
            and r['physical_clear'] is r['rejected_GPU_namespace_exists'] is r['rejected_GPU_marker_exists'] is False
            and r['controller_rows']==r['queue_rows']==r['rejected_GPU_label_rows']==[]
            and r['node']==NODE and r['parent']==base['parent']=='1194900' and r['original_numbered']==base['previous']==PREVIOUS
            and r['request']==base['request'] and r['old_gpu_accounting']==base['pinned_GPU'] and r['rejected_request']==base['new_request']==REJECTED
            and cpu[:4]==[CPU,'training80-r25-fresh-'+REJECTED[:12],'COMPLETED','0:0']
            and cpu[5:7]==[NODE,'cpu=1,mem=256M,node=1'] and cpu[7]<cpu[8]
            and reply==dict(intent,transport=transport) and REJECTED==intent['request']==send['request']==transport['request']==out['request']
            and transport['payload_sha256']==send['payload_sha256'] and transport['rc']==0 and transport['remote_outcome']=='RESPONSE_OBSERVED'
            and transport['authority_verified'] is transport['remote_received'] is transport['payload_send_attempted'] is True
            and out['GPU_srun_attempted'] is False and out['zero_GRES_call_attempted'] is out['mutex_acquired'] is True
            and out['state']=='GPU_NOT_SENT' and out['probe_rc']==0 and out['probe_stderr_SHA256']==warning['OLD_WARNING']
            and r['files_SHA256']==ancestry['targets'][NODE]['sealed_original_files'],
            'actual recent old probe/CPU/no-GPU or original worker authority differs')
    controller=budget['controller'](context[1][6]['controller.py'])
    seconds=controller['node_seconds'](NODE,10800,REQUEST)
    row=dict(base,new_request=REQUEST,renewal_seconds=seconds,rejected_request=REJECTED,rejected_scope=rejected_scope,
             rejected_outcome=out,old_CPU=cpu,old_no_send_SHA256=r['durable_no_send_SHA256'])
    claim=dict(scope=STEM,state='PREPARED_ONLY_FRESH_DURABLE_AND_PHYSICAL_ADMISSION_REQUIRED',node=NODE,parent=base['parent'],
               launcher=WRAPPER.relative_to(ROOT).as_posix(),prior_GPU=PREVIOUS,prior_clearance_CPU=base['clearance'][0],
               rejected_CPU=CPU,rejected_GPU_request=REJECTED,rejected_scope_file_SHA256=REJECTED_FILES,request=REQUEST,
               independent_original_reconciliation_SHA256=TERMINAL[1],durable_no_send_SHA256=r['durable_no_send_SHA256'],
               old_worker_files=r['files_SHA256'],original_R39_launcher_SHA256=R39[1],controller_SHA256=budget['CONTROLLER'][1],
               worker_seconds=seconds,Slurm_time_cap_seconds=seconds+300,GPU=8,max_invocations=1,ledger_mutations=0,replay_allowed=False)
    return budget,first,r34,early,n02,context,row,claim,warning


def runtime(material,claim_sha):
    budget,first,r34,early,n02,context,row,claim,warning=material
    fn,rep=first['_fn'],early['unique']
    tools=dict(early)
    def read_program(ctx,item,nonce):
        source=early['read_program'](ctx,item,nonce)
        return rep(source,'    return dict(node=x["node"],step=step,request=x["request"],accounting=fresh,',
                   '    ready=bool(ready and fresh==x["pinned_GPU"] and tpin==x["pinned_terminal"])\n'
                   '    return dict(node=x["node"],step=step,request=x["request"],accounting=fresh,')
    tools['read_program']=read_program
    def with_no_send(ctx,item,acct,terminal,pub):
        launched=fn(early['render'],STEM=STEM)(ctx,item,acct,terminal,pub)
        source=launched['launch']
        label=f'training80-dcu-standby-{REJECTED[:12]}-{NODE}'
        source=rep(source,'([old_label], [gpu_label])',f'([old_label], [gpu_label], [{label!r}])')
        anchor='            check_previous(account(PREVIOUS))'
        new=(
            f"            no_send_scope=RESULTS/{item['rejected_scope']!r}\n"
            "            no_send_path=no_send_scope/'outcome.json'\n"
            "            require(no_send_path.resolve(strict=True)==no_send_path and no_send_path.stat().st_uid==os.getuid(), 'prior no-send owner or path changed')\n"
            "            with no_send_path.open('rb') as old_no_send_handle:\n"
            "                old_no_send_bytes=old_no_send_handle.read(8193)\n"
            f"            require(0<len(old_no_send_bytes)<=8192 and sha256(old_no_send_bytes).hexdigest()=={item['old_no_send_SHA256']!r}\n"
            f"                    and json.loads(old_no_send_bytes)=={item['rejected_outcome']!r}, 'immutable original no-send changed')\n"
            f"            require(not os.path.lexists(RESULTS/{('training80-dcu-standby-'+REJECTED)!r})\n"
            "                    and not os.path.lexists(no_send_scope/'gpu-send-intent.json'), 'original GPU namespace or send intent appeared')\n"
            f"            require(account({CPU!r})=={warning['seven'](item['old_CPU'])!r}, 'previous warning probe not exact C0')\n")
        source=rep(source,anchor,new+anchor)
        require(source.index('fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)')<source.index('no_send_scope=RESULTS')
                <source.index('outcome["zero_GRES_call_attempted"] = True'),'existing lock/no-send/fresh admission order changed')
        launched['launch']=source
        return launched
    tools['render']=with_no_send
    budget_api=fn(budget['api'],STEM=STEM,G34=LAST40,WRAPPER=WRAPPER,CLAIM=CLAIM,__file__=__file__)
    ctx,reader,read,render,dispatch,observe=budget_api((first,r34,tools,n02,context,{NODE:row},claim),claim_sha)
    earlier=dispatch.__globals__['publication']
    def publication(ctx,revision):
        earlier(ctx,revision)
        _,publisher=ctx[3][-2]['modules']()
        for path,sha in (R39,WARNING):
            require(publisher['git']('cat-file','blob',revision+':'+path)==pin(ROOT/path,sha),'published original no-send guard differs')
    return ctx,reader,read,render,fn(dispatch,publication=publication),observe


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true')
    mode.add_argument('--observe-predecessor')
    mode.add_argument('--publication')
    mode.add_argument('--observe-original',action='store_true')
    p.add_argument('--terminal-receipt',nargs=2)
    args=p.parse_args()
    require(bool(args.publication)==bool(args.terminal_receipt),'exact current worker read is required')
    material=load()
    budget,first,r34,_,_,_,row,claim,warning=material
    blob=r34['pinned'](DRAFT if args.check and not CLAIM.exists() else CLAIM)
    require(json.loads(blob)==claim,'exact n15 successor claim differs')
    sha=digest(blob)
    ctx,reader,read,render,dispatch,observe=runtime(material,sha)
    if args.check:
        check=reader(ctx,row,'a'*32)
        cgroup=json.dumps(dict(node=NODE,step=PREVIOUS,worker_pid=row['previous_worker_pid'],worker_start_ticks=1,
                               same_UID=True,exact_job_and_step_cgroup=True)).encode()
        synthetic=json.dumps(row['rejected_outcome'],sort_keys=True).encode()
        fixture=dict(row,old_no_send_SHA256=digest(synthetic))
        program=render(ctx,fixture,fixture['pinned_GPU'],(fixture['pinned_terminal'],digest(cgroup)),None)
        warning['no_send_fixture'](program['launch'],fixture,synthetic)
        first['cgroup_fixture'](program['launch'],fixture,cgroup,require)
        seconds=budget['spawn_fixture'](program['launch'],fixture,ctx[1][6]['controller.py'])
        result=dict(state='PREPARED_ONLY',request=REQUEST,worker_seconds=seconds,reader_SHA256=digest(check.encode()),
                    locked_no_send_positive=1,locked_no_send_vetoes=4,locked_cgroup_CPU_positive=1,locked_cgroup_CPU_vetoes=2,
                    actual_spawn_positive=1,actual_spawn_vetoes=2,Git=0,SSH=0,Slurm=0,GPU=0)
    elif args.observe_predecessor:
        result=read(ctx,row,args.observe_predecessor)
    elif args.observe_original:
        result=observe(ctx,row)
    else:
        result=dispatch(ctx,row,args.publication,args.terminal_receipt)
    print(json.dumps(dict(result,claim_SHA256=sha),sort_keys=True,allow_nan=False))


if __name__=='__main__':
    main()
