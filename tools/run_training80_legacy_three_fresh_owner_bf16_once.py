"""Create one guarded fresh-owner BF16 standby on each of three closed legacy targets."""

import argparse
import ast
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import runpy
from tempfile import TemporaryDirectory
from types import FunctionType, SimpleNamespace
from uuid import UUID, uuid5


ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT/'local-only'
WRAPPER = Path(__file__).resolve()
STEM = 'training80-legacy-three-fresh-owner-bf16-v1'
CLAIM = ROOT/'coordination/tasks'/f'{STEM}.json'
R25 = ('tools/run_training80_r25_early_ordinary_guarded_refill_once.py',
       'f64000177ec337dad0b097007440a0a727ebf4e99712f68bd4603d527922b9fe')
R35_READER = ('tools/run_training80_r35_first_four_guarded_successor_once.py',
              '50518c96ab1a752c0a9272482074b5d67def57bb2945a7ab7a98b21e6154f609')
R35_BUDGET = ('tools/run_training80_r35_next_three_guarded_successor_once.py',
              'a35f3359976f1bf6af0ea20f199091e83728e5d21e9cb1275828bea959d6b043')
CLOSURE = ('local-only/training80-three-legacy-four-original-bridge-writer-closure-readonly-35cb2118626d4cb6b1c5e49137e4c345/receipt.json',
           '95099336984dc308787dbcea75d59991cc8927e5162de485bb4795321fb79997')
DIAG = ('local-only/training80-0751-legacy-four-zero-gres-diagnostic-v1-18ad8a6e285440339ed16b50313aaf2c-reply.json',
        'c5e12a600d7aa3ec34aa62cb8ea453b0b85eb5449ee01184c89eeb60541c7e7d')
DIAG_TERMINAL = ('local-only/training80-four-zero-gres-exact-independent-terminal-18ad8a6e-f90e0b2aa83c466b8be5f437c35e325c/receipt.json',
                 'bc7693a90532ace97618bd9183724418a1aa2177f6cc462bd74d9af05340df98')
DIAG_MANIFEST = ('local-only/training80-0751-legacy-four-zero-gres-diagnostic-v1-manifest.json',
                 'e5bc42d229bc67e577982c50d12727fe1255bd20ea13b8a13f2e5dc72c8841f2')
GAPS = ('local-only/training80-0751-37-numbered-gaps-ledger-and-hold-audit-v1.json',
        '46a14209a43ababccdf1bc04bf00e02c5d55063a434af3d42e0c687c5cc4e30c')
NODES = ('f13r1n08','f12r3n17','f13r4n17')
PROTECTED = ('f13r1n15','f13r2n19','f13r2n07','f13r2n08','f13r3n13')
REVISION = re.compile('[0-9a-f]{40}').fullmatch


LOCKED_WRITERS = r'''
def old_writer_closure():
    for rel,pin in OLD_WRITER_MEMBERS.items():
        p=RESULTS/rel
        require(p.resolve(strict=True)==p and p.stat().st_uid==os.getuid(),'old bridge file owner changed')
        with p.open('rb') as stream:
            raw=stream.read(262145)
        require(0<len(raw)<=262144 and sha256(raw).hexdigest()==pin,'old bridge closure evidence changed')
    ids=set(OLD_ACCOUNTING)
    by_id={i:[] for i in ids}
    for line in query('sacct','-nP','-j',','.join(sorted(ids)),'-o',
                      'JobIDRaw,JobName%110,State,ExitCode,NodeList,AllocTRES%160,End'):
        row=line.split('|')
        if row[0] in ids:
            by_id[row[0]].append(row)
    require(all(by_id[ident]==[row] for ident,row in OLD_ACCOUNTING.items()),'old GPU/control/diagnostic terminal changed')
    for parent in sorted({i.split('.')[0] for i in ids}):
        controller={fields(x)['StepId'] for x in query('scontrol','show','step','-o',parent)}
        queue={line.split('|')[0] for line in query('squeue','--steps','-h','-j',parent,'-o','%i|%N')}
        require(not (controller|queue)&ids,'old writer or previous diagnostic again visible')
'''


def require(test, detail):
    if not test:
        raise ValueError(detail)


def pin(spec):
    path=ROOT/spec[0]
    require(path.is_relative_to(ROOT) and path.resolve(strict=True)==path,'noncanonical pinned custody')
    with path.open('rb') as stream:
        raw=stream.read(262145)
    require(0<len(raw)<=262144 and sha256(raw).hexdigest()==spec[1],'exact frozen bytes differ')
    return raw


def module(spec):
    pin(spec)
    return runpy.run_path(str(ROOT/spec[0]))


def observed(packet):
    require(packet['rc']==0 and packet['remote_outcome']=='RESPONSE_OBSERVED'
            and packet['authority_verified'] is packet['remote_received'] is True
            and packet['request']==packet['response']['request'],'independent response authority differs')
    return packet['response']


def seven(row):
    return row[:4]+row[5:7]+row[8:]


def load():
    base=module(R25); reader=module(R35_READER); budget=module(R35_BUDGET)
    original=base['load']()
    local=json.loads(pin(GAPS))['rows']
    diag_raw=json.loads(pin(DIAG))
    diag=observed(diag_raw['transport'])
    terminals=observed(json.loads(pin(DIAG_TERMINAL)))
    wires=observed(json.loads(pin(CLOSURE)))
    manifest=json.loads(pin(DIAG_MANIFEST))
    require(wires['global_error'] is None and len(wires['bridges'])==4 and wires['Slurm_mutations']==wires['GPU']==0
            and all(v['state']=='REMOTE_OLD_WRITER_TERMINAL' and v['old_handles_absent_from_both_current_views']
                    for v in wires['bridges'].values())
            and terminals['seen_terminal']==4 and terminals['scope']==diag_raw['request']
            and all(not view for record in terminals['current'].values() for view in record.values())
            and diag['GPU_requested']==0 and diag['old_release_proven'] is False,'old external closure is incomplete')
    nodes={}
    targets={}
    for node in NODES:
        frozen=manifest['targets'][node]
        sample=diag['nodes'][node]['sample']
        cpu=terminals['accounting'][sample['step']]
        require(local[node]['detail']['old_cgroup_SHA256'] is None
                and sample['node']==node and sample['parent']==frozen['parent']
                and sample['own_probe_cgroup'] is sample['global_driver_KFD_queried'] is True
                and sample['first']==sample['second']==dict(PCIs=frozen['pci'],kfd_process_count=0,
                    own_device_fd_process_count=0,occupied_requested_ports=[])
                and re.fullmatch('[a-f0-9]{64}',sample['boot_SHA256'])
                and cpu[:4]==[sample['step'],'training80-four-gap-zero-'+diag_raw['request']+'-'+node,'COMPLETED','0:0']
                and cpu[5:7]==[node,'cpu=1,mem=256M,node=1'],'separate 08:16 PCI/boot or zero-GRES accounting differs')
        records=local[node]['ledger_rows']
        require(len(records)==2 and records[1]['bridge']==frozen['old_request'],'known fixed SQLite ancestry differs')
        old_account={cpu[0]:seven(cpu)}
        members={}
        for claim in records:
            req=claim['bridge']
            bridge=wires['bridges'][req]
            target=[i for i,values in bridge['exact_accounting'].items() if values[0][4]==node]
            require(len(target)==1 and node in bridge['target_nodes'],'exact old bridge GPU target differs')
            for handle in (bridge['control_step'],target[0]):
                [row]=bridge['exact_accounting'][handle]
                require(row[2:4]==['COMPLETED','0:0'],'old writer control or target C0 absent')
                old_account[handle]=row
            for name in ('intent','compute','child_terminal'):
                members['training80-standby-compute-'+req+'/'+name.replace('_','-')+'.json']=bridge[name+'_SHA256']
            members['training80-dcu-standby-'+req+'/submission-receipt.json']=bridge['submission_SHA256']
        previous=frozen['gpu'][0]
        require(old_account[previous]==seven(frozen['gpu']) and len(old_account)==5,'old exact accounting identity differs')
        new=uuid5(UUID('de92bc62-2788-49a8-a7b3-9f04672feea6'),f'{node}/{previous}/{frozen["old_request"]}').hex
        seconds=original[1][7](node,10800,new)
        entry=records[-1]
        nodes[node]=dict(node=node,parent=frozen['parent'],previous=previous,request=frozen['old_request'],
                         new_request=new,renewal_seconds=seconds,worker_start_epoch=frozen['worker_start_epoch'],
                         prepare_epoch=frozen['old_worker_end_epoch']-900,planned_end_epoch=frozen['old_worker_end_epoch'],
                         worker_seconds=int(round(frozen['old_worker_end_epoch']-frozen['worker_start_epoch'])),
                         previous_worker_pid=frozen['old_worker_pid'],worker_SHA256=frozen['files']['worker'],
                         expected_SHA256=frozen['files']['expected'],submit_SHA256=frozen['files']['submission'],
                         pci=frozen['pci'],pinned_GPU=frozen['gpu'],pinned_terminal=frozen['files']['terminal'],
                         old_members=members,old_account=old_account,boot=sample['boot_SHA256'],
                         ledger_old=entry['ancestor'],ledger_request=entry['claim'],published_group=entry['group'],
                         journal_steps=[item['ancestor'] for item in records]+[previous],ledger_rows=records)
        targets[node]=dict(parent=frozen['parent'],old_GPU=previous,new_request=new,worker_seconds=seconds,
                           old_cgroup_historical_state='MISSING_NOT_USED',initial_probe=cpu[0],
                           observed_boot_SHA256=sample['boot_SHA256'],original_bridge_requests=[x['bridge'] for x in records])
    require(diag['nodes']['f13r1n15']['sample']['second']['occupied_requested_ports']==[43843]
            and set(nodes).isdisjoint(PROTECTED),'port-blocked or business node overlaps')
    claim=dict(scope=STEM,launcher=WRAPPER.relative_to(ROOT).as_posix(),targets=targets,
        old_writer_closure_SHA256=CLOSURE[1],diagnostic_SHA256=DIAG[1],diagnostic_C0_SHA256=DIAG_TERMINAL[1],
        initial_historical_owner_evidence_SHA256=DIAG_MANIFEST[1],source_pins=dict((R25,R35_READER,R35_BUDGET)),
        actual_physical_admissions=0,protected_nodes=list(PROTECTED),GPU_per_node=8,ledger_mutations=0,
        max_concurrent_send=1,max_invocations_per_node=1,replay_allowed=False)
    return base,reader,budget,original,nodes,claim


def local_booking(base,ordinary,material,row):
    exact=base['exact_no_other_claims'](ordinary,material,row)
    expected={(v['parent'],v['ancestor'],v['claim'],v['wave'],v['group']) for v in row['ledger_rows']}
    require(len(exact)==2 and set(exact)==expected,'exact booking history has an extra writer')
    journal=LOCAL/'training80-1738-ordinary-renewal-dispatch'
    require(journal.is_dir() and not journal.is_symlink(),'unverified old ordinary journal')
    for step in row['journal_steps']:
        require(not any(os.path.lexists(journal/f"{row['node']}-{step}-{slot}.json") for slot in range(3)),
                'old ordinary slot still has a ticket')
    selected=re.compile(r'-(intent|send|reply)\.json$').search
    require(not any(selected(p.name) for p in LOCAL.glob('training80-*-'+row['node']+'-*.json')),
            'a target singleton intent/send/receipt exists; reconcile original')


def clone(function,**change):
    return FunctionType(function.__code__,dict(function.__globals__,**change),function.__name__,
                        function.__defaults__,function.__closure__)


def runtime(base,reader,budget,context,revision):
    def read_program(ctx,row,nonce):
        return f'# fresh-owner-publication {revision}\n'+base['read_program'](ctx,row,nonce)

    def terminal_identity(row,nonce,proof):
        account,terminal=base['terminal_identity'](row,nonce,proof)
        require(account==row['pinned_GPU'] and terminal==row['pinned_terminal']
                and proof['stderr_SHA256']==sha256(b'').hexdigest(),'fresh old natural return drifted')
        return account,terminal

    def render(ctx,row,account,terminal,commit):
        out=clone(base['render'],STEM=STEM)(ctx,row,account,terminal,commit)
        source=out['launch']
        inject=base['unique']
        source=inject(source,'\ndef run_guarded():',
            f"\nOLD_WRITER_MEMBERS={row['old_members']!r}\nOLD_ACCOUNTING={row['old_account']!r}\n"
            f"OLD_PROBE_BOOT={row['boot']!r}\n"+LOCKED_WRITERS+'\n\ndef run_guarded():')
        source=inject(source,'            check_previous(account(PREVIOUS))',
            '            old_writer_closure()\n            check_previous(account(PREVIOUS))')
        source=inject(source,'            probe_sample(sample)',
            '            require(sample["boot_SHA256"]==OLD_PROBE_BOOT, "target boot changed since independent observation")\n'
            '            probe_sample(sample)')
        source=budget['host_budget_guard'](source,inject)
        compile(source,'<fresh-owner same-lock guarded standby>','exec')
        out['launch']=source
        return out

    def publication(ctx,commit):
        require(commit==revision and REVISION(commit),'fresh read and dispatch publication must match')
        clone(base['publication'],WRAPPER=WRAPPER,CLAIM=CLAIM,__file__=__file__)(ctx,commit)
        _,publisher=ctx[3][-2]['modules']()
        for spec in (R25,R35_READER,R35_BUDGET):
            require(publisher['git']('cat-file','blob',commit+':'+spec[0])==pin(spec),'published custody guard differs')

    path=clone(base['read_path'],STEM=STEM)
    slots=clone(base['slot_paths'],STEM=STEM)
    read=clone(base['observe_predecessor'],read_program=read_program,read_path=path,
                runpy=SimpleNamespace(run_path=reader['predecessor_reader_helper']))
    terminal=clone(base['terminal'],read_program=read_program,terminal_identity=terminal_identity)
    dispatch=clone(base['dispatch'],slot_paths=slots,read_path=path,terminal=terminal,render=render,
                   publication=publication,exact_no_other_claims=lambda ordinary,material,row:local_booking(base,ordinary,material,row))
    observe=clone(base['observe_original'],STEM=STEM,read_path=path,slot_paths=slots,
                  terminal_identity=terminal_identity,render=render)
    return read_program,read,render,dispatch,observe,publication


def check_locked_gates(launch,row):
    tree=ast.parse(launch)
    old_gate=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='old_writer_closure')
    old_code=compile(ast.Module(body=[old_gate],type_ignores=[]),'<locked old writer fixture>','exec')
    boot=next(x for x in ast.walk(tree) if isinstance(x,ast.Expr)
              and 'target boot changed since independent observation' in ast.unparse(x))
    boot_code=compile(ast.Module(body=[boot],type_ignores=[]),'<locked boot fixture>','exec')
    with TemporaryDirectory() as tmp:
        folder=Path(tmp).resolve()
        p=folder/'bound'
        p.write_bytes(b'known bytes')
        exact=row['old_account']
        all_rows=['|'.join(r) for r in exact.values()]
        active=next(iter(exact))
        for mode in ('clean','old-bytes','missing-accounting','late-writer'):
            def query(tool,*args):
                if tool=='sacct':
                    return all_rows[1:] if mode=='missing-accounting' else all_rows
                if mode=='late-writer' and tool=='squeue' and active.startswith(args[3]+'.'):
                    return [active+'|'+row['node']]
                return []
            env=dict(RESULTS=folder,OLD_WRITER_MEMBERS={p.name:sha256(
                     b'drift' if mode=='old-bytes' else b'known bytes').hexdigest()},OLD_ACCOUNTING=exact,
                     require=require,sha256=sha256,query=query,fields=lambda line:{'StepId':line},
                     os=SimpleNamespace(getuid=lambda:p.stat().st_uid))
            exec(old_code,env)
            try:
                env['old_writer_closure']()
            except ValueError:
                require(mode!='clean','a valid closed writer was rejected')
            else:
                require(mode=='clean','old writer integrity/accounting/active veto missing')
    exec(boot_code,dict(require=require,sample={'boot_SHA256':row['boot']},OLD_PROBE_BOOT=row['boot']))
    try:
        exec(boot_code,dict(require=require,sample={'boot_SHA256':'0'*64},OLD_PROBE_BOOT=row['boot']))
    except ValueError:
        pass
    else:
        raise ValueError('different boot accepted before GPU path')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true')
    mode.add_argument('--check-bookings',action='store_true')
    mode.add_argument('--observe-predecessor')
    mode.add_argument('--publication')
    mode.add_argument('--observe-original',action='store_true')
    parser.add_argument('--node',choices=NODES)
    parser.add_argument('--published-revision')
    parser.add_argument('--terminal-receipt',nargs=2)
    args=parser.parse_args()
    require(args.check or args.check_bookings or args.node,'need exact target')
    require(bool(args.observe_predecessor)==bool(args.published_revision)
            and bool(args.publication)==bool(args.terminal_receipt),'exact publication and receipt required')
    revision=args.publication or args.published_revision or '0'*40
    require(REVISION(revision),'invalid publication syntax')
    base,reader,budget,original,nodes,claim=load()
    raw=CLAIM.read_bytes()
    require(json.loads(raw)==claim,'frozen fresh-owner claim differs')
    ctx=(*original[:5],nodes,sha256(raw).hexdigest())
    source,read,render,send,observe,publication=runtime(base,reader,budget,ctx,revision)
    if args.check:
        result={}
        for node,row in nodes.items():
            reader_source=source(ctx,row,'1'*32)
            out=render(ctx,row,row['pinned_GPU'],row['pinned_terminal'],None)
            launch=out['launch']
            require(launch.index('old_writer_closure()\n')<launch.index('outcome["zero_GRES_call_attempted"] = True')
                    and launch.index('target boot changed since independent observation')<launch.index('gpu-send-intent.json')
                    and 'worker-cgroup.json' in launch and len(row['old_members'])==8,'guard ordering differs')
            check_locked_gates(launch,row)
            base['host_fixture'](clone(base['render'],STEM=STEM)(ctx,row,row['pinned_GPU'],row['pinned_terminal'],None)['launch'],
                                 row,row['pinned_GPU'],ctx[1][6])
            seconds=budget['spawn_fixture'](launch,row,ctx[1][6]['controller.py'])
            result[node]=dict(request=row['new_request'],seconds=seconds,reader_SHA256=sha256(reader_source.encode()).hexdigest())
        response=dict(state='PREPARATION_ONLY',targets=result,old_writer_and_boot_vetoes=12,SCNet=0,Git=0,GPU=0)
    elif args.check_bookings:
        for row in nodes.values():
            local_booking(base,ctx[2],ctx[3],row)
        require(not set(nodes)&ctx[0]['local_holds'](),'current project priority owns a target')
        response=dict(state='LOCAL_BOOKINGS_CLEAR_3',targets=list(nodes),SCNet=0,Git=0,GPU=0)
    elif args.observe_predecessor:
        row=nodes[args.node]
        publication(ctx,revision)
        local_booking(base,ctx[2],ctx[3],row)
        require(row['node'] not in ctx[0]['local_holds'](),'current project priority owns the node')
        response=read(ctx,row,args.observe_predecessor)
    elif args.observe_original:
        response=observe(ctx,nodes[args.node])
    else:
        response=send(ctx,nodes[args.node],revision,args.terminal_receipt)
    print(json.dumps(response,sort_keys=True,allow_nan=False))


if __name__=='__main__':
    main()
