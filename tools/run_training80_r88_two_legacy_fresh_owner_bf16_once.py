"""Prepare one independent fresh-owner standby for each measured legacy node."""

from datetime import datetime, timedelta
from hashlib import sha256
import inspect
import json
from pathlib import Path
import runpy
import textwrap
from types import FunctionType
from uuid import UUID, uuid5


ROOT=Path(__file__).resolve().parents[1]
WRAPPER=Path(__file__).resolve()
STEM='training80-r88-two-legacy-fresh-owner-bf16-v1'
CLAIM=ROOT/'coordination/tasks'/f'{STEM}.json'
NODES=('f13r2n01','f13r2n02')
R82=('tools/run_training80_legacy_three_fresh_owner_bf16_once.py',
     'de55ffda601984e9222fc315535cdd957207f862a82ba8610d5f9cd8ee40ced6')
V3=('tools/run_training80_legacy_n17_warning_closed_bf16_v3_once.py',
    '34554bd81c1abf2df17df8d21c134b24f6dd7d37883b4222d17ae9f55ff5d73a')
SCOPE=('local-only/render_training80_r86_two_legacy_fresh_zero_gres.py',
       '744177a82faf42b6bcf3ecb802db7b4b613e4e8835976e8f445ca0248f959cd8')
SCOPE_CLAIM=('local-only/training80-r86-two-legacy-fresh-zero-gres-v1.json',
             '0c0ebf1cfb266dec7d1b72d307f6384a8ffb113d73050d21bcc668001fd1305c')
ARCHIVE=('results/bw1000/2026-11-26/training80-r86-two-legacy-nodes-3713-3714-independent-zero-gres.json',
         'f4df270e6cf2d31b19dd16a099ecab59d6dfa48d1d0d5f00fcfb17ce9abc81f5')
TERMINAL=('local-only/training80-r86-legacy-3713-3714-exact-independent-terminal-v2-ca597344a9e04353990693cae428abfa/receipt.json',
          'cf2808df07920021b3f4dd85f37c0374a1664c26d41b132bd666adddbf52d88a')
RECEIPTS={
    'f13r2n01': ('1103606.3713','fd32427bb81b50c6b5df9998802eedf0',
                 '73e91e6ccbf878a89956438dd3b372aab7ccd18dadbfba9463244be8b9bcfb90',
                 '43737763e03e993569b904b1d8e25ea79ebd6e1056351da490f05e4c73eb24ef',
                 '7c0b6445f5951937df1adea2ce67e2e9dcf72444c3cfff2b33bad070bf9cc794'),
    'f13r2n02': ('1103606.3714','0531f805f5825075bc0c91c61474f0c4',
                 'afd81cbcc02994d31dee9ffd4a458c67858116e0825bcfcff79983241e1c6e6a',
                 'd2cc5b02eaad71831437d66eb35e0f310a25d3d38c7a3240005673b49eef2bad',
                 '5d3035925239e17cb7f0b3c9043e11ab240c494f69a861a8995d0c76563eb499'),
}
LOCKED_NO_SEND=r'''
def verify_old_gpu_names_absent():
    history=[line.split('|') for line in query('sacct','-nP','-j',PARENT,'-o','JobIDRaw,JobName%110')]
    require(all(len(row)==2 for row in history) and any(row[0]==PARENT for row in history),'original parent history unavailable')
    for request,scope in CLOSED_WARNINGS.items():
        name='training80-dcu-standby-'+request[:12]+'-'+NODE
        require(not any(label==name for _,label in history)
                and not os.path.lexists(RESULTS/scope/'gpu-send-intent.json')
                and not os.path.lexists(RESULTS/('training80-dcu-standby-'+request)),
                'old warning GPU label or effective intent appeared')
'''


def require(ok,detail):
    if not ok:
        raise ValueError(detail)


def pin(spec):
    path=ROOT/spec[0]
    require(path.is_relative_to(ROOT) and path.resolve(strict=True)==path,'exact private evidence path differs')
    with path.open('rb') as stream:
        raw=stream.read(262145)
    require(0<len(raw)<=262144 and sha256(raw).hexdigest()==spec[1],'frozen evidence bytes differ')
    return raw


def module(spec):
    pin(spec)
    return runpy.run_path(str(ROOT/spec[0]))


def clone(function,namespace=None,**changes):
    return FunctionType(function.__code__,dict(namespace or function.__globals__,**changes),function.__name__,
                        function.__defaults__,function.__closure__)


def observed(packet,nonce):
    body=packet['response']
    require(packet['rc']==0 and packet['remote_outcome']=='RESPONSE_OBSERVED'
            and packet['authority_verified'] is packet['remote_received'] is packet['payload_flushed'] is True
            and packet['request']==body['request']==nonce,'independent authorized receipt differs')
    return body


def beijing_end_as_utc(text):
    end=datetime.strptime(text,'%Y-%m-%dT%H:%M:%S')-timedelta(hours=8)
    return end.isoformat(timespec='seconds')


def entry():
    original=module(R82)
    diagnostics=module(SCOPE)
    v3=module(V3)
    audit=json.loads(pin(ARCHIVE))
    terminal_packet=json.loads(pin(TERMINAL))
    terminal=observed(terminal_packet,'ca597344a9e04353990693cae428abfa')
    require(audit['independent_accounting_receipt_SHA256']==TERMINAL[1]
            and audit['actual_GPU_steps']==terminal['slurm_mutations']==0
            and audit['GPU_dispatch_eligible'] is False and set(audit['nodes'])==set(NODES)
            and set(terminal['actual'])=={value[0] for value in RECEIPTS.values()},
            'published independent legacy zero-GRES archive differs')
    namespace=dict(original,STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,NODES=NODES,__file__=__file__,__doc__=__doc__)

    def load():
        diagnostic,probe,probe_claim,planned,claim86=diagnostics['plan']()
        require(json.loads(pin(SCOPE_CLAIM))==claim86,'frozen original admission scope differs')
        base=module(original['R25'])
        reader=module(original['R35_READER'])
        budget=module(original['R35_BUDGET'])
        ctx=base['load']()
        rows,targets={},{}
        for node in NODES:
            step,probe_id,intent_sha,send_sha,receipt_sha=RECEIPTS[node]
            old=planned[node]
            directory=f'local-only/training80-r86-{node}-fresh-zero-gres-once-{probe_id}'
            intent,send,receipt=(json.loads(pin((f'{directory}/{name}.json',digest)))
                for name,digest in zip(('intent','send','receipt'),(intent_sha,send_sha,receipt_sha),strict=True))
            data=observed(receipt,probe_id)
            device=data['nodes'][node]
            sample=device['sample']
            numeric=terminal['actual'][step]
            acct=numeric['row']
            filled=diagnostics['render'](diagnostic,probe,probe_claim,node,old)
            rendered={sha256(filled.encode()).hexdigest(),sha256(filled.replace('\n','\r\n').encode()).hexdigest()}
            require(intent['request']==send['request']==old['new_request']==probe_id
                    and intent['program_sha'] in rendered and send['payload_sha256']==receipt['payload_sha256']
                    and receipt_sha==audit['nodes'][node]['observer_receipt_SHA256']
                    and data['scope']==diagnostics['STEM'] and set(data['nodes'])=={node}
                    and data['GPU_requested']==data['existing_workload_mutations']==0 and data['zero_GRES_attempted']==1
                    and data['GPU_dispatch_eligible'] is data['automatic_standby'] is data['replay_allowed'] is False
                    and device['probe_invoked'] is True and device['probe_rc']==0
                    and device['probe_stderr_SHA256']==sha256(b'').hexdigest()
                    and sample['request']==probe_id and sample['step']==step and sample['node']==node
                    and sample['parent']==old['parent']=='1103606' and sample['observer_uid']==154459
                    and sample['own_probe_cgroup'] is sample['global_driver_KFD_queried'] is True
                    and sample['other_UID_device_FDs']=='UNOBSERVED' and sample['ports_checked']==diagnostic['PORTS']
                    and sample['first']==sample['second']==dict(PCIs=old['pci'],kfd_process_count=0,
                        own_device_fd_process_count=0,occupied_requested_ports=[])
                    and original['re'].fullmatch('[a-f0-9]{64}',sample['boot_SHA256']),
                    'physical diagnostic or exact original staged request differs')
            require(numeric['rc']==0 and numeric['exact_rows']==numeric['valid_rows']==1
                    and acct==audit['nodes'][node]['exact_accounting_columns'] and len(acct)==9
                    and acct[:4]==[step,'training80-r86-legacy-zero-'+probe_id[:12]+'-'+node,'COMPLETED','0:0']
                    and acct[4].isdecimal() and acct[5:7]==[node,'cpu=1,mem=256M,node=1'] and acct[7]<=acct[8],
                    'independent initial zero-GRES C0/label or exact End differs')
            # The accepted R82/v3 contract checks exact seven-column C0/End.
            # ElapsedRaw is preserved in the claim, including the zero-second row.
            probe_account=acct[:4]+acct[5:7]+[beijing_end_as_utc(acct[8])]
            if 'probe_accounting' in device:
                [same]=device['probe_accounting']
                require(original['seven'](same)==probe_account,'independent Beijing and original UTC accounting differ')
            previous=old['previous']
            new=uuid5(UUID('c0d56268-4209-42d3-ba5f-e99de20358f6'),f"{node}/{previous}/{probe_id}").hex
            seconds=ctx[1][7](node,10800,new)
            require(new not in {old['request'],probe_id,*old['warnings']} and 10500<=seconds<10800,
                    'new exact controlled worker request differs')
            start=old['worker_start_epoch']
            end=old['old_worker_end_epoch']
            rows[node]=dict(old,new_request=new,renewal_seconds=seconds,pinned_GPU=old['gpu'],
                pinned_terminal=old['files']['terminal'],worker_SHA256=old['files']['worker'],
                expected_SHA256=old['files']['expected'],submit_SHA256=old['files']['submission'],
                previous_worker_pid=old['old_worker_pid'],worker_seconds=int(round(end-start)),
                planned_end_epoch=end,prepare_epoch=end-900,cgroup_SHA256=None,boot=sample['boot_SHA256'],
                old_account=dict(old['old_account'],**{step:probe_account}))
            targets[node]=dict(parent='1103606',old_GPU=previous,old_GPU_request=old['request'],
                new_request=new,worker_seconds=seconds,old_cgroup_historical_state='MISSING_NOT_USED',
                initial_probe_step=step,initial_probe_request=probe_id,initial_elapsed_raw_seconds=int(acct[4]),
                initial_probe_receipt_SHA256=receipt_sha,initial_observed_boot_SHA256=sample['boot_SHA256'],
                initial_inline_post_read_complete='post_views' in device,closed_warning_requests=sorted(old['warnings']))
        require(len({r['new_request'] for r in rows.values()})==len(rows)==2,'exact two fresh scopes differ')
        claim=dict(scope=STEM,launcher=WRAPPER.relative_to(ROOT).as_posix(),targets=targets,
            exact_diagnostic_archive_SHA256=ARCHIVE[1],independent_zero_GRES_terminal_SHA256=TERMINAL[1],
            old_writer_closure_SHA256=claim86['actual_old_writer_read_SHA256'],source_pins=dict((R82,V3,ARCHIVE)),
            private_diagnostic_pins=dict((SCOPE,SCOPE_CLAIM)),actual_physical_admissions=0,
            fresh_stderr_rule='REJECT_ALL_NONEMPTY_STDERR',GPU_per_node=8,ledger_mutations=0,
            max_concurrent_send=1,max_invocations_per_node=1,replay_allowed=False)
        return base,reader,budget,ctx,rows,claim

    def booking(base,ordinary,material,row):
        diagnostics['booking']({row['node']:row})

    def runtime(base,reader,budget,ctx,revision):
        inherited=clone(original['runtime'],namespace,local_booking=booking)
        source,read,old_render,send,observe,old_publication=inherited(base,reader,budget,ctx,revision)

        def render(ctx,row,account,terminal,commit):
            program=old_render(ctx,row,account,terminal,commit)
            injection=base['unique']
            launch=injection(program['launch'],'\ndef run_guarded():',
               f"\nCLOSED_WARNINGS={row['warnings']!r}\n"+LOCKED_NO_SEND+'\n\ndef run_guarded():')
            launch=injection(launch,'            old_writer_closure()\n',
                             '            old_writer_closure()\n            verify_old_gpu_names_absent()\n')
            program['launch']=v3['with_private_stderr'](launch,injection)
            return program

        def publication(ctx,commit):
            old_publication(ctx,commit)
            _,publisher=ctx[3][-2]['modules']()
            for spec in (R82,V3,ARCHIVE):
                require(publisher['git']('cat-file','blob',commit+':'+spec[0])==pin(spec),
                        'production fresh-owner source or independent archive differs')

        return source,read,render,clone(send,render=render,publication=publication),clone(observe,render=render),publication

    def check_additional(launch,row):
        v3['stderr_fixture'](launch)
        require(launch.index('            verify_old_gpu_names_absent()')
                < launch.index('outcome["zero_GRES_call_attempted"] = True')
                and len(row['warnings'])==2,'warning namespace comparison moved after shared physical gate')

    namespace.update(load=load,runtime=runtime,local_booking=booking,check_additional=check_additional)
    cli=textwrap.dedent(inspect.getsource(original['main']))
    anchor='            check_locked_gates(launch,row)\n'
    require(cli.count(anchor)==1,'inherited local fixture differs')
    cli=cli.replace(anchor,anchor+'            check_additional(launch,row)\n',1)
    cli=cli.replace("launch.index('gpu-send-intent.json')","launch.index('save(audit / \"gpu-send-intent.json\"')")
    cli=cli.replace("len(row['old_members'])==8","len(row['old_members'])==18")
    cli=cli.replace('LOCAL_BOOKINGS_CLEAR_3','LOCAL_BOOKINGS_CLEAR_2')
    cli=cli.replace('old_writer_and_boot_vetoes=12','verified_nodes=len(nodes),private_stderr_fixtures_per_node=4')
    exec(compile(cli,'<exact two legacy standby CLI>','exec'),namespace)
    return load,namespace['main']


if __name__=='__main__':
    _,main=entry()
    main()
