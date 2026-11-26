"""Prepare bounded zero-GRES diagnostics for the two independently closed legacy scopes."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
from uuid import UUID, uuid5


ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT/'local-only'
STEM = 'training80-r86-two-legacy-fresh-zero-gres-v1'
MANIFEST = LOCAL/f'{STEM}.json'
CLOSURE = ('local-only/training80-r86-two-legacy-exact-warning-lineages-8fa213376d784148af32b5a3f3d63d0d/receipt.json',
           'd513f154e54f93bcd6e5462c0adc86b9ce55f0ba913ab56a38201aa2aae7a74f')
CLOSURE_INTENT = ('local-only/training80-r86-two-legacy-exact-warning-lineages-8fa213376d784148af32b5a3f3d63d0d/intent.json',
                  'ee411a88c3526b879b62bef7404ae4131ffd704cef52e905f5e852a3ccb6b278')
CLOSURE_SEND = ('local-only/training80-r86-two-legacy-exact-warning-lineages-8fa213376d784148af32b5a3f3d63d0d/send.json',
                'ade08b633c923f21ecb37748a37c9e586743a3c676a7730602a9a4b4ca5d5911')
SCOPE = ('local-only/render_training80_r86_two_old_warning_lineages_readonly.py',
         '904618c7a81d0ea2615857c834ab6c43e11d495a24c3ca0c4d11c5733ec3e540')
DIAGNOSTIC = ('local-only/run_training80_legacy_four_zero_gres_diagnostic_once.py',
              '6e36707730d98b1312434b6f769d1c5f9791e104f93d14c3dd120d6c3d874f4a')
R29 = ('tools/run_training80_r29_other11_warning_guarded_refill_once.py',
       'c3162682176376fcd84099c09372ec3e3010de54c106e04278ed3752e132159b')
R25 = ('tools/run_training80_r25_early_ordinary_guarded_refill_once.py',
       'f64000177ec337dad0b097007440a0a727ebf4e99712f68bd4603d527922b9fe')
AUDIT = ('local-only/training80-0751-37-numbered-gaps-ledger-and-hold-audit-v1.json',
         '46a14209a43ababccdf1bc04bf00e02c5d55063a434af3d42e0c687c5cc4e30c')
NODES = ('f13r2n01','f13r2n02')
MARKERS = {
    '589902a329a15fa6a668fa432bc1c820': ('1093517ae85def8e11a70296073f7d2c5e648e5c56b9479f221b5e483737af8e',
                                      '2f1a867c2b6a9ba39a948d0494a4a4ac346780c5e1b0424977ce7b13ecd25d9a'),
    'ec38f1aef7255793b884aa677022262c': ('8b709b58e90746923040bf8b043d94b46ac10ee2b6927853d8b2a37e3aef86f4',
                                      'f81f1709dc9133283d82fa01d02e86a0782add5bc895cc5722c17698e07ebd58'),
    '81c30c6076ab5adcb7ac16903fc13d08': ('204e20b991935e2367b0239a198a7aa4377a4bc860392aeb507b05082bf73b3f',
                                      '7044d71c906e0617ba2478dfa8b3afa5cbca6515dfc75325b0b63b00d4e0fe41'),
    '3a3aebb97363500bbd3092c01bf07ad3': ('cde5c5dc73c4ab31749380ba0a5bfa3adee7335d358c50894a09115a2e637646',
                                      'f01b1671a8f16a3f2e59134939866a2510cbcac3a563e9e24a5766b72cd0018c'),
}
SCHED_READS = {
    'f13r2n01': ('local-only/training80-r29-other11-warning-guarded-bf16-v1-f13r2n01-read-9d53e2ecbd0e43baa0ef13a6e05101cb.json',
                 'd7581e1e5a0d6df1b6f32124c6eab6b0a178c5fb41368bc486139f9305da511f'),
    'f13r2n02': ('local-only/training80-r29-other11-warning-guarded-bf16-v1-f13r2n02-read-d435801d28da4091a316d273e4559f87.json',
                 'd2e6956850b50762db0c6cd2b82c6c2b24f6fa2a3b64d050db5562e860319f38'),
}


def require(test,detail):
    if not test:
        raise ValueError(detail)


def raw(path):
    require(path.is_relative_to(ROOT) and path.resolve(strict=True)==path,'canonical exact evidence path required')
    with path.open('rb') as stream:
        data=stream.read(262145)
    require(0<len(data)<=262144,'bounded private evidence required')
    return data


def pin(spec):
    data=raw(ROOT/spec[0])
    require(sha256(data).hexdigest()==spec[1],'frozen evidence differs')
    return data


def module(spec):
    pin(spec)
    return runpy.run_path(str(ROOT/spec[0]))


def plan():
    scope=module(SCOPE)
    legacy,bridges,targets=scope['prepare']()
    packet=json.loads(pin(CLOSURE))
    reply=packet['response']
    require(packet['rc']==0 and packet['remote_outcome']=='RESPONSE_OBSERVED'
            and packet['authority_verified'] is packet['payload_flushed'] is packet['remote_received'] is True
            and packet['request']==reply['request'] and reply['global_error'] is None and not reply['GPU_admission_authorized']
            and reply['GPU']==reply['Slurm_mutations']==0 and set(reply['bridges'])==set(bridges)
            and set(reply['targets'])==set(targets)==set(NODES),'actual independent known writers result differs')
    source=scope['render'](legacy,bridges,targets,packet['request'])
    intent,send=(json.loads(pin(spec)) for spec in (CLOSURE_INTENT,CLOSURE_SEND))
    require(intent['request']==send['request']==packet['request'] and send['payload_sha256']==packet['payload_sha256']
            and sha256(source.replace('\n','\r\n').encode()).hexdigest()==intent['program_sha'],
            'actual old-writer framed payload or Windows source bytes differ')
    route=module(R29)
    *_,older,_=route['bound_api']()['load']()
    audit=json.loads(pin(AUDIT))['rows']
    diagnostic=module(DIAGNOSTIC)
    probe=diagnostic['module'](diagnostic['PROBE_CODE'])
    probe_claim=diagnostic['data'](diagnostic['PROBE_CLAIM'])
    fleet_pci=diagnostic['data']((probe_claim['pci_pins']['path'],probe_claim['pci_pins']['SHA256']))
    nodes={}
    for node in NODES:
        previous=older[node]
        history=audit[node]['ledger_rows']
        current=reply['targets'][node]
        original_read=json.loads(pin(SCHED_READS[node]))
        [identity]=original_read['observed']['targets']
        require(current['state']=='KNOWN_WRITERS_AND_WARNING_SCOPES_CLOSED' and not current['physical_admitted']
                and not any(current['current_target_steps'].values())
                and set(current['writer_states'])=={row['bridge'] for row in history}
                and set(current['warning_scopes'])==set(targets[node]['warnings'])
                and original_read['rc']==0 and original_read['replay_allowed'] is False
                and identity['original_terminal_and_targets_empty'] is True and identity['worker_cgroup_SHA256'] is None
                and identity['request']==previous['request'] and identity['step']==previous['previous']
                and identity['worker_SHA256']==previous['worker_SHA256']
                and identity['expected_SHA256']==previous['expected_SHA256']
                and sorted(previous['pci'])==sorted(fleet_pci[node]) and len(set(previous['pci']))==8,
                'known old owner, terminal or historical cgroup state differs')
        account,members,markers={}, {}, {}
        for bridge_id in targets[node]['writers']:
            bridge=reply['bridges'][bridge_id]
            require(bridge['state']==current['writer_states'][bridge_id]=='REMOTE_OLD_WRITER_TERMINAL'
                    and bridge['old_handles_absent_from_both_current_views'] is True
                    and bridge['target_nodes']==bridges[bridge_id]['targets'],'original writer closure differs')
            for step,records in bridge['exact_accounting'].items():
                [row]=records
                require(row[0]==step and row[2:4]==['COMPLETED','0:0'],'old writer accounting differs')
                account[step]=row
            prefix='training80-standby-compute-'+bridge_id
            for item in ('intent','compute','child_terminal','entry','bridge'):
                extension='.py' if item in {'entry','bridge'} else '.json'
                members[f'{prefix}/{item.replace("_","-")}{extension}']=bridge[item+'_SHA256']
            members[f'training80-dcu-standby-{bridge_id}/submission-receipt.json']=bridge['submission_SHA256']
        warnings={}
        for request,warn in targets[node]['warnings'].items():
            evidence=current['warning_scopes'][request]
            [cpu]=evidence['CPU']
            require(evidence['verified'] is True and evidence['GPU_labels']==[]
                    and evidence['namespace_or_intent_exists'] is False and cpu[2:4]==['COMPLETED','0:0']
                    and cpu[4:6]==[node,'cpu=1,mem=256M,node=1'],'original zero-GRES warning closing proof differs')
            account[cpu[0]]=cpu
            warnings[request]=warn['scope']
            for name,key in (('intent','intent'),('fresh-probe-intent','probe_intent'),('outcome','outcome')):
                members[f'{warn["scope"]}/{name}.json']=evidence[key+'_SHA256']
            old=next(v for v in scope['NODES'][node][-1] if v[1]==request)
            for key,digest in zip(('intent','send','reply'),(*MARKERS[request],old[2]),strict=True):
                name=f'{warn["scope"]}-{key}.json'
                data=json.loads(pin((f'local-only/{name}',digest)))
                require(data['request']==request,'old local transport artifacts changed role')
                markers[name]=digest
        newest=history[-1]
        request=uuid5(UUID('a6c148be-7ce0-428b-9722-33ff23c0c867'),f'{node}/{CLOSURE[1]}').hex
        nodes[node]=dict(parent='1103606',node=node,previous=identity['step'],request=previous['request'],new_request=request,
            ledger_request=newest['claim'],ledger_old=newest['ancestor'],published_group=newest['group'],ledger_rows=history,
            journal_steps=[h['ancestor'] for h in history]+[identity['step']],local_markers=markers,
            old_account=account,old_members=members,warnings=warnings,gpu=identity['accounting'],
            control=account[reply['bridges'][previous['request']]['control_step']],
            old_request=previous['request'],pci=sorted(previous['pci']),old_worker_pid=previous['previous_worker_pid'],
            worker_start_epoch=previous['worker_start_epoch'],old_worker_end_epoch=previous['planned_end_epoch'],
            files=dict(worker=identity['worker_SHA256'],expected=identity['expected_SHA256'],terminal=identity['terminal_SHA256'],
                       submission=previous['submit_SHA256']),historical_cgroup_SHA256=None)
    claim=dict(scope=STEM,actual_old_writer_read_SHA256=CLOSURE[1],
        source_pins=dict((SCOPE,DIAGNOSTIC,R29,R25)),ports=diagnostic['PORTS'],zero_GRES_only=True,GPU_requested=0,
        physical_admissions=0,automatic_standby=False,replay_allowed=False,targets={
            n:dict(parent=row['parent'],request=row['new_request'],historical_GPU=row['previous'],
                   original_accounting_steps=sorted(row['old_account']),historical_cgroup_SHA256=None)
            for n,row in nodes.items()})
    return diagnostic,probe,probe_claim,nodes,claim


def booking(nodes):
    ordinary=module(R25)
    base,_,authority,context,*_=ordinary['load']()
    held=base['local_holds']()
    journal=LOCAL/'training80-1738-ordinary-renewal-dispatch'
    require(journal.is_dir() and not journal.is_symlink(),'ordinary intent journal differs')
    for node,row in nodes.items():
        historical=ordinary['exact_no_other_claims'](authority,context,row)
        expected={(v['parent'],v['ancestor'],v['claim'],v['wave'],v['group']) for v in row['ledger_rows']}
        require(len(historical)==2 and set(historical)==expected and node not in held,'active booking or business hold')
        require(not any(os.path.lexists(journal/f'{node}-{h}-{slot}.json') for h in row['journal_steps'] for slot in range(3)),
                'existing ordinary reservation must be reconciled')
        seen={p.name for p in LOCAL.glob(f'training80-*-{node}-*.json') if re.search(r'-(intent|send|reply)\.json$',p.name)}
        require(seen==set(row['local_markers']),'old or independent singleton intent differs')
        for name,digest in row['local_markers'].items():
            pin((f'local-only/{name}',digest))
    tags=['run_training80_r25_last40_ordinary_guarded_refill_once.py',
          'run_training80_r29_other11_warning_guarded_refill_once.py',*MARKERS]
    quoted=','.join(f"'{v}'" for v in tags)
    text="$ErrorActionPreference='Stop'; $tags="+quoted+"; "+r'''
$running = @(Get-CimInstance Win32_Process | Where-Object {
    if ($_.ProcessId -eq $PID -or -not $_.CommandLine) { return $false }
    $entry = $_.CommandLine.Replace('\','/')
    @($tags | Where-Object { $entry.IndexOf($_,[StringComparison]::OrdinalIgnoreCase) -ge 0 }).Count -gt 0
})
@{count=$running.Count} | ConvertTo-Json -Compress
'''
    exe = Path(os.environ['SystemRoot'])/'System32/WindowsPowerShell/v1.0/powershell.exe'
    reply=subprocess.run([str(exe),'-NoProfile','-NonInteractive','-Command',text],capture_output=True,timeout=18)
    require(reply.returncode==0 and 0<len(reply.stdout)<=2048 and not reply.stderr
            and json.loads(reply.stdout)=={'count':0},'an old local warning sender may still run')
    return dict(exact_adopted_rows_per_node=2,legacy_slots=0,closed_singleton_receipts_per_node=2,old_senders=0,held=[])


REMOTE = r'''
def seven(row):
    return row[:4]+row[5:7]+row[8:]


def verify_old(node,row,rows,history):
    for handle,pin in row['old_account'].items():
        current=rows[handle]
        need(len(current)==1 and seven(current[0])==pin,'old terminal or its allocation changed')
    for request,scope in row['warnings'].items():
        gpu_label='training80-dcu-standby-'+request[:12]+'-'+node
        need(not any(name==gpu_label for _,name in history)
             and not os.path.lexists(RESULTS/scope/'gpu-send-intent.json')
             and not os.path.lexists(RESULTS/('training80-dcu-standby-'+request)),
             'old warning GPU evidence now exists')
    for name,pin in row['old_members'].items():
        path=RESULTS/name
        need(path.resolve(strict=True)==path and path.stat().st_uid==os.getuid(),'old object owner or canonical path differs')
        with path.open('rb') as stream:
            data=stream.read(262145)
        need(0<len(data)<=262144 and sha256(data).hexdigest()==pin,'old closed object changed')
'''


def render(diagnostic,probe,claim,node,row):
    host=diagnostic['HOST']
    def swap(old,new):
        nonlocal host
        require(host.count(old)==1,'exact diagnostic source boundary differs')
        host=host.replace(old,new,1)
    swap('def precheck(',REMOTE+'\n\ndef precheck(')
    swap("originals={n:{row['control'][0],row['gpu'][0]} for n,row in PLAN.items()}",
         "originals={n:set(row['old_account']) for n,row in PLAN.items()}")
    swap("    need(all(rows[record[0]]==[record] for record in (row['gpu'],row['control'])),'old GPU/control C0 differs')",
         "    need(rows[row['gpu'][0]]==[row['gpu']],'old GPU and full accounting differ')\n"
         "    verify_old(node,row,rows,history)")
    swap("old_ids={value[0] for row in PLAN.values() for value in (row['gpu'],row['control'])}",
         "old_ids={handle for row in PLAN.values() for handle in row['old_account']}")
    swap('ThreadPoolExecutor(max_workers=4)','ThreadPoolExecutor(max_workers=1)')
    selected={k:v for k,v in row.items() if k not in {'ledger_rows','local_markers'}}
    request=row['new_request']
    bound={node:dict(selected,label='training80-r86-legacy-zero-'+request[:12]+'-'+node)}
    constants=dict(PLAN=bound,REQUEST=request,SCOPE=STEM,PORTS=diagnostic['PORTS'],
                   STAGED=claim['staged_probe_SHA256'],NODE_SOURCE=probe['NODE_PROBE'],PREFLIGHT=CLOSURE[1])
    source='\n'.join(f'{key}={value!r}' for key,value in constants.items())+'\n'+host
    compile(source,'<fresh standalone zero-GRES only>', 'exec')
    return source


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--check',action='store_true')
    modes.add_argument('--check-bookings',action='store_true')
    modes.add_argument('--render',choices=NODES)
    modes.add_argument('--freeze',action='store_true')
    args=parser.parse_args()
    diagnostic,probe,probe_claim,nodes,claim=plan()
    if args.freeze:
        with MANIFEST.open('xb') as stream:
            stream.write((json.dumps(claim,sort_keys=True,indent=2)+'\n').encode())
        return
    require(json.loads(raw(MANIFEST))==claim,'frozen zero-GRES plan differs')
    if args.render:
        booking({args.render:nodes[args.render]})
        print(render(diagnostic,probe,probe_claim,args.render,nodes[args.render]),end='')
    elif args.check_bookings:
        print(json.dumps(booking(nodes),sort_keys=True))
    else:
        sources={n:sha256(render(diagnostic,probe,probe_claim,n,row).encode()).hexdigest() for n,row in nodes.items()}
        print(json.dumps(dict(state='PREPARATION_ONLY',nodes=claim['targets'],source_SHA256=sources,
                              physical_admissions=0,SSH=0,Git=0,Slurm=0,GPU=0),sort_keys=True))


if __name__=='__main__':
    main()
