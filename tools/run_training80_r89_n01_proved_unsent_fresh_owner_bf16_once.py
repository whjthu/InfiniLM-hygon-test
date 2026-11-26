"""Prepare one new n01 standby after proving the prior attempt never sent a payload."""

from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
import runpy
import textwrap
from types import FunctionType
from uuid import UUID, uuid5


ROOT=Path(__file__).resolve().parents[1]
WRAPPER=Path(__file__).resolve()
STEM='training80-r89-n01-proved-unsent-fresh-owner-bf16-v1'
CLAIM=ROOT/'coordination/tasks'/f'{STEM}.json'
NODE='f13r2n01'
R88=('tools/run_training80_r88_two_legacy_fresh_owner_bf16_once.py',
     'f0351905d3f41540632a6f1273d0fd7314ca5353b4185953f822214e8b029f50')
R88_CLAIM=('coordination/tasks/training80-r88-two-legacy-fresh-owner-bf16-v1.json',
           'a215db9547151c6b6383a6d7a81b98d4b925ccb886295035b710ed81b6505b44')
R45=('tools/run_training80_r45_six_2117_guarded_successor_once.py',
     '067934fedb0e6b89a59ac8f5f3c65ac763ed55784680e880bcf6b1dc74644786')
FLEET=('local-only/training80-after-r88-new-two-original-receipts-7c5cbfdcb7f2412f983a679e62215d77/receipt.json',
       '06bedc822642c2df374708b42d20d3066b0b64c676908fe493e88a8d22b75b72')
PREVIOUS='d055b6c61e9b5b918dd48116bc45fb09'
PREVIOUS_PREFIX=f'training80-r88-two-legacy-fresh-owner-bf16-v1-{NODE}-{PREVIOUS}'
OLD_MARKERS={
    PREVIOUS_PREFIX+'-intent.json':'e2b5aba999c08535151d2a4a1593f660e30b075419415deeb1074fda08d73d6e',
    PREVIOUS_PREFIX+'-reply.json':'1c8c4c4873f34911dafb44f0e169cbeefdb28af7a14f0399d2e3a8ca0b3bace3',
}
OLD_READ=('local-only/training80-r88-two-legacy-fresh-owner-bf16-v1-f13r2n01-read-ebf5f364e0214aab916fe8023c673c15.json',
          '204964551eebaab0d2645e774ca65973c8f7c9a6af342672b80797b1c5aa99f9')


def clone(function,namespace=None,**changes):
    return FunctionType(function.__code__,dict(namespace or function.__globals__,**changes),function.__name__,
                        function.__defaults__,function.__closure__)


def entry():
    if sha256((ROOT/R88[0]).read_bytes()).hexdigest()!=R88[1]:
        raise ValueError('frozen R88 guard differs')
    prior=runpy.run_path(str(ROOT/R88[0]))
    require,pin,module=(prior[k] for k in ('require','pin','module'))
    original=module(prior['R82'])
    v3=module(prior['V3'])
    shadow=module(R45)
    diagnostic=module(prior['SCOPE'])
    load88,main88=prior['entry']()
    scope=dict(original,STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,NODES=(NODE,),__file__=__file__,__doc__=__doc__)

    def no_send_token(exists=os.path.lexists):
        path=ROOT/'local-only'/f'{PREVIOUS_PREFIX}-send.json'
        require(not exists(path),'prior R88 send token now exists; reconcile original')

    def exact_failed():
        intent,reply=(json.loads(pin((f'local-only/{path}',digest))) for path,digest in OLD_MARKERS.items())
        wire=reply['transport']
        require({k:v for k,v in reply.items() if k!='transport'}==intent and intent['node']==NODE
                and intent['request']==wire['request']==PREVIOUS and intent['replay_allowed'] is False
                and intent['GPU_requested']==8 and intent['claim_SHA256']==R88_CLAIM[1]
                and wire['remote_outcome']=='PAYLOAD_NOT_SENT' and wire['authority_verified'] is False
                and wire['payload_send_attempted'] is wire['payload_flushed'] is wire['remote_received'] is False
                and wire['rc'] is wire['response'] is None and wire['stdout_sha256']==sha256(b'').hexdigest(),
                'prior R88 positive no-send proof differs')
        no_send_token()
        return intent

    def load():
        base,reader,budget,ctx,older,prior_claim=load88()
        old=older[NODE]
        require(old['new_request']==PREVIOUS and old['cgroup_SHA256'] is None
                and json.loads(pin(R88_CLAIM))==prior_claim,'published R88 predecessor material differs')
        failed=exact_failed()
        receipt=json.loads(pin(OLD_READ))
        nonce=failed['terminal_read_id']
        require(failed['terminal_read_SHA256']==OLD_READ[1] and OLD_READ[0].endswith(f'-read-{nonce}.json')
                and receipt['rc']==0 and receipt['replay_allowed'] is False,'older actual predecessor read differs')
        actual=(*ctx[:5],older,R88_CLAIM[1])
        source,_,render,_,observe,_=main88.__globals__['runtime'](base,reader,budget,actual,failed['publication'])
        require(sha256(source(actual,old,nonce).encode()).hexdigest()==receipt['source_SHA256'],
                'older actual predecessor source differs')
        account,terminal=observe.__globals__['terminal_identity'](old,nonce,receipt)
        require(sha256(render(actual,old,account,terminal,failed['publication'])['launch'].encode()).hexdigest()
                ==failed['program_SHA256'],'exact unsent R88 launch bytes differ')
        fleet=json.loads(pin(FLEET))
        snapshot=fleet['response']
        require(fleet['rc']==0 and fleet['authority_verified'] is fleet['remote_received'] is True
                and snapshot['running_numbered_card_bounds']==[344,344]
                and snapshot['queries_read_only'] is True and NODE in snapshot['parents']['1103606']['nodes_without_running_DCU_step']
                and 'f13r2n02' not in snapshot['parents']['1103606']['nodes_without_running_DCU_step'],
                'fresh scheduler root scope differs')
        new=uuid5(UUID('516b9265-bb13-4bf2-a00c-31c1f997ac09'),f'{NODE}/{PREVIOUS}').hex
        seconds=ctx[1][7](NODE,10800,new)
        require(new not in {PREVIOUS,old['old_request']} and 10500<=seconds<10800,'different fresh request or actual budget required')
        row=dict(old,new_request=new,renewal_seconds=seconds,
                 local_markers=dict(old['local_markers'],**OLD_MARKERS),prior_no_send=dict(request=PREVIOUS,scope=PREVIOUS_PREFIX))
        target=dict(prior_claim['targets'][NODE],new_request=new,worker_seconds=seconds,
                    prior_R88_request=PREVIOUS,prior_R88_reply_SHA256=OLD_MARKERS[PREVIOUS_PREFIX+'-reply.json'],
                    prior_payload_send_attempted=False)
        claim=dict(prior_claim,scope=STEM,launcher=WRAPPER.relative_to(ROOT).as_posix(),targets={NODE:target},
                   source_pins=dict(prior_claim['source_pins'],**dict((R88,R88_CLAIM,R45))),
                   prior_R88_replay_allowed=False,prior_namespace_absence_required_under_lock=True,
                   excluded_sibling='f13r2n02')
        return base,reader,budget,ctx,{NODE:row},claim

    def booking(base,ordinary,material,row):
        exact_failed()
        diagnostic['booking']({NODE:row})

    def runtime(base,reader,budget,ctx,revision):
        inherited=clone(original['runtime'],scope,local_booking=booking)
        source,read,old_render,send,observe,old_publication=inherited(base,reader,budget,ctx,revision)

        def new_render(context,row,account,terminal,commit):
            program=old_render(context,row,account,terminal,commit)
            injection=base['unique']
            launch=injection(program['launch'],'\ndef run_guarded():',
               f"\nCLOSED_WARNINGS={row['warnings']!r}\n"+prior['LOCKED_NO_SEND']+'\n\ndef run_guarded():')
            launch=injection(launch,'            old_writer_closure()\n',
                             '            old_writer_closure()\n            verify_old_gpu_names_absent()\n')
            launch=shadow['guarded_shadow'](launch,row['prior_no_send'],NODE,injection)
            program['launch']=v3['with_private_stderr'](launch,injection)
            return program

        def publication(context,commit):
            old_publication(context,commit)
            _,publisher=context[3][-2]['modules']()
            for spec in (prior['R82'],prior['V3'],prior['ARCHIVE'],R88,R88_CLAIM,R45):
                require(publisher['git']('cat-file','blob',commit+':'+spec[0])==pin(spec),
                        'published old no-send or fresh-owner evidence differs')

        return (source,read,new_render,clone(send,render=new_render,publication=publication),
                clone(observe,render=new_render),publication)

    def check_shadow(launch,row):
        shadow['shadow_fixture'](launch,row,row['prior_no_send'])
        v3['stderr_fixture'](launch)
        no_send_token(lambda _:False)
        try:
            no_send_token(lambda _:True)
        except ValueError:
            pass
        else:
            raise ValueError('prior local send marker failed to veto')

    scope.update(load=load,runtime=runtime,local_booking=booking,check_shadow=check_shadow)
    cli=textwrap.dedent(inspect.getsource(original['main']))
    anchor='            check_locked_gates(launch,row)\n'
    require(cli.count(anchor)==1,'inherited check boundary differs')
    cli=cli.replace(anchor,anchor+'            check_shadow(launch,row)\n',1)
    cli=cli.replace("launch.index('gpu-send-intent.json')","launch.index('save(audit / \"gpu-send-intent.json\"')")
    cli=cli.replace("len(row['old_members'])==8","len(row['old_members'])==18")
    cli=cli.replace('LOCAL_BOOKINGS_CLEAR_3','LOCAL_BOOKINGS_CLEAR_N01')
    cli=cli.replace('old_writer_and_boot_vetoes=12','verified_nodes=len(nodes),private_stderr_fixtures_per_node=4')
    exec(compile(cli,'<exact n01 no-send successor check>','exec'),scope)
    return load,scope['main']


if __name__=='__main__':
    _,main=entry()
    main()
