"""Use a new scope for the three R82 reads that failed before any send."""

import ast
from hashlib import sha256
import json
import os
from pathlib import Path
import runpy
from types import FunctionType
from uuid import UUID, uuid5


ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/'local-only'
WRAPPER=Path(__file__).resolve()
STEM='training80-legacy-three-fresh-owner-bf16-v2'
CLAIM=ROOT/'coordination/tasks'/f'{STEM}.json'
V1=('tools/run_training80_legacy_three_fresh_owner_bf16_once.py',
    'de55ffda601984e9222fc315535cdd957207f862a82ba8610d5f9cd8ee40ced6')
R45=('tools/run_training80_r45_six_2117_guarded_successor_once.py',
     '067934fedb0e6b89a59ac8f5f3c65ac763ed55784680e880bcf6b1dc74644786')
REVISION_V1='91c2ca1b6f93a87ba9c84ba23f21629653ac6ea6'
CLAIM_V1_SHA='72de200c15349c50bd5a4c1573fcbf6f30ca4e12bb5aa544f95d25e14f122d40'
STDERR='ffbe619701dd12801e215e390af30a223800a39847494e6d44caefef04482214'
READS={
    'f13r1n08':('cfb7080b8b524c72a775013cfad1b04a','f721a1fb9f4c6651ee82ef83d06b8ac6172fa407c21c12145daeeca2ce4963f6'),
    'f12r3n17':('e65b2bd71dae4efba408a79b3988a4e5','7d73254d81a8ecdaf04c114e7aa26b439ea3a87a1fb35ba772b898ae7e065d6e'),
    'f13r4n17':('13b0eb42e11b4c8dbe77b39873ffd1bc','b9c12c66096041ed5c084a071b9c2d2a39cb4edd97f03f231574c88034b99ea3'),
}


def require(test,detail):
    if not test:
        raise ValueError(detail)


def pin(path,expected):
    require(path.is_relative_to(ROOT) and path.resolve(strict=True)==path,'noncanonical evidence path')
    with path.open('rb') as stream:
        raw=stream.read(262145)
    require(0<len(raw)<=262144 and sha256(raw).hexdigest()==expected,'frozen bytes differ')
    return raw


def clone(function,namespace=None,**changes):
    return FunctionType(function.__code__,dict(namespace or function.__globals__,**changes),function.__name__,
                        function.__defaults__,function.__closure__)


def no_v1_marker(scope,exists=os.path.lexists):
    prefix=LOCAL/scope
    require(not any(exists(prefix.with_name(prefix.name+'-'+kind+'.json')) for kind in ('intent','send','reply')),
            'original v1 has a launch marker; reconcile and stop')


def entry():
    raw=pin(ROOT/V1[0],V1[1])
    original=runpy.run_path(str(ROOT/V1[0]))
    shadow=runpy.run_path(str(ROOT/R45[0])) if pin(ROOT/R45[0],R45[1]) else None
    namespace=dict(original,ROOT=ROOT,LOCAL=LOCAL,WRAPPER=WRAPPER,STEM=STEM,CLAIM=CLAIM,
                   __file__=__file__,__doc__=__doc__)

    def load():
        base,reader,budget,ctx,old,prior_claim=original['load']()
        require(set(old)==set(READS) and json.loads(pin(original['CLAIM'],CLAIM_V1_SHA))==prior_claim,
                'exact v1 claim differs')
        original_ctx=(*ctx[:5],old,CLAIM_V1_SHA)
        old_source=original['runtime'](base,reader,budget,original_ctx,REVISION_V1)[0]
        new={}; targets={}
        expected_traceback=("Traceback (most recent call last):\n  File \"<stdin>\", line 95, in <module>\n"
                            "  File \"<stdin>\", line 69, in group\nKeyError: 'cgroup_SHA256'\n")
        require(sha256(expected_traceback.encode()).hexdigest()==STDERR,'frozen original traceback differs')
        for node,row in old.items():
            nonce,receipt_sha=READS[node]
            p=LOCAL/f"{original['STEM']}-{node}-read-{nonce}.json"
            receipt=json.loads(pin(p,receipt_sha))
            rendered=old_source(original_ctx,row,nonce)
            require(receipt['rc']==1 and receipt['observed'] is None and receipt['replay_allowed'] is False
                    and receipt['stdout_SHA256']==sha256(b'').hexdigest() and receipt['stderr_SHA256']==STDERR
                    and receipt['source_SHA256']==sha256(rendered.encode()).hexdigest()
                    and 'cgroup_SHA256' not in row and 'x["cgroup_SHA256"]' in rendered.splitlines()[68],
                    'original covered read did not fail at the exact known boundary')
            scope=f"{original['STEM']}-{node}-{row['new_request']}"
            no_v1_marker(scope)
            request=uuid5(UUID('a34f212e-50d7-4aa2-a1cc-3e5ddfe82ca0'),f"{node}/{row['new_request']}").hex
            seconds=ctx[1][7](node,10800,request)
            require(request not in {row['request'],row['new_request']} and 10500<=seconds<=10800,'new request differs')
            new[node]=dict(row,new_request=request,renewal_seconds=seconds,cgroup_SHA256=None,
                           previous_v1=dict(request=row['new_request'],scope=scope))
            targets[node]=dict(prior_claim['targets'][node],new_request=request,worker_seconds=seconds,
                                prior_v1_request=row['new_request'],prior_read_only_failure_SHA256=receipt_sha,
                                prior_v1_send_marker_present=False,historical_cgroup_null=True)
        claim=dict(prior_claim,scope=STEM,launcher=WRAPPER.relative_to(ROOT).as_posix(),targets=targets,
                   original_failed_claim_SHA256=CLAIM_V1_SHA,original_failed_source_SHA256=V1[1],
                   source_pins=dict(prior_claim['source_pins'],**dict((V1,R45))),v1_replay_allowed=False)
        return base,reader,budget,ctx,new,claim

    def booking(base,ordinary,material,row):
        no_v1_marker(row['previous_v1']['scope'])
        original['local_booking'](base,ordinary,material,row)

    def runtime(base,reader,budget,context,revision):
        inherited=clone(original['runtime'],namespace,local_booking=booking)
        source,read,old_render,dispatch,observe,old_publication=inherited(base,reader,budget,context,revision)
        def render(ctx,row,account,terminal,commit):
            program=old_render(ctx,row,account,terminal,commit)
            program['launch']=shadow['guarded_shadow'](program['launch'],row['previous_v1'],row['node'],base['unique'])
            return program
        def publication(ctx,commit):
            old_publication(ctx,commit)
            _,publisher=ctx[3][-2]['modules']()
            for spec in (V1,R45):
                require(publisher['git']('cat-file','blob',commit+':'+spec[0])==pin(ROOT/spec[0],spec[1]),
                        'published original scope or no-send guard differs')
        return source,read,render,clone(dispatch,publication=publication,render=render),clone(observe,render=render),publication

    def check_compatibility(reader_source,launch,row):
        tree=ast.parse(reader_source)
        inputs=next(n for n in tree.body if isinstance(n,ast.Assign)
                    and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='items')
        [wire]=ast.literal_eval(inputs.value)
        require('cgroup_SHA256' in wire and wire['cgroup_SHA256'] is None,'unattested historical field must be null on the wire')
        group=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='group')
        decoded=clone_fixture(group,wire)
        require(decoded['worker_cgroup_SHA256'] is None and decoded['original_terminal_and_targets_empty'] is False,
                'compatibility field must permit reading without granting terminal readiness')
        shadow['shadow_fixture'](launch,row,row['previous_v1'])
        no_v1_marker(row['previous_v1']['scope'],lambda _:False)
        try:
            no_v1_marker(row['previous_v1']['scope'],lambda _:True)
        except ValueError:
            pass
        else:
            raise ValueError('original launch marker not rejected')

    def clone_fixture(group,wire):
        import re
        env={'base':Path('.'),'read':lambda p:(None,None),'index':{wire['previous']:[]},
             'views':{wire['node']:{'scontrol':[],'squeue':[]}},'re':re}
        exec(compile(ast.Module(body=[group],type_ignores=[]),'<compatibility null predicate fixture>','exec'),env)
        return env['group'](wire)

    namespace.update(load=load,runtime=runtime,local_booking=booking,check_compatibility=check_compatibility)
    module=ast.parse(raw)
    original_main=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    text=ast.get_source_segment(raw.decode(),original_main)
    anchor='            check_locked_gates(launch,row)\n'
    require(text.count(anchor)==1,'existing fresh-owner fixture boundary differs')
    text=text.replace(anchor,anchor+'            check_compatibility(reader_source,launch,row)\n',1)
    exec(compile(text,'<R82 v2 compatibility CLI>','exec'),namespace)
    return load,namespace['main']


if __name__=='__main__':
    _,main=entry()
    main()
