"""Bind the exact R33 n18 natural return to one guarded successor."""

import ast
from hashlib import sha256
from pathlib import Path
import runpy
from types import FunctionType


ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/'local-only'
STEM='training80-r42-n18-guarded-bf16-v1'
WRAPPER=ROOT/'tools/run_training80_r42_n18_guarded_successor_once.py'
CLAIM=ROOT/'coordination/tasks'/f'{STEM}.json'
DRAFT=LOCAL/f'{STEM}.draft.json'
N13=('tools/run_training80_r41_n13_guarded_successor_once.py',
     'caf690dd2763779483d8ad693a62327dc3724d43d568a3aea4c0309294f9678c')
R33=('tools/run_training80_r33_n18_postdiagnostic_guarded_refill_once.py',
     '75b77e3107b1ec28008b2cb87ec879dba8619b139e2631c3457d0b9310d1d152')
TERMINAL=('training80-r41-n18-next-original-read-70b02981537f4ddeb893858992436e10.json',
          '139dd8a356b5816c89d88cde844c6013f1e0794ff0b5e1682dd32caa585f3081')
COUNTER=('training80-r33-n18-exact-paired-counters-224e9420abe34266804d0122633e1795.json',
         '600b60adf59c1e7b0443ff01a8433047d7bdc55a81acc779dedb2bfbe8bbf78b')
OLD_REPLY='cf72f0bb4130f9bd1dcbdc83ffdc5e3b237e8d4f375b36f5d3bc04144f30167b'
NODE='f12r3n18'
STEP='1194900.1388'
CPU='1194900.1387'
NODES=(NODE,)


def entry():
    path=ROOT/N13[0]
    raw=path.read_bytes()
    if path.resolve(strict=True)!=path or sha256(raw).hexdigest()!=N13[1]:
        raise ValueError('exact singleton scope implementation differs')
    prior=runpy.run_path(str(path))
    require=prior['require']
    env=dict(prior,__file__=__file__,STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,DRAFT=DRAFT,R33=R33,
             TERMINAL=TERMINAL,COUNTER=COUNTER,OLD_REPLY=OLD_REPLY,NODE=NODE,STEP=STEP,CPU=CPU,NODES=NODES)
    tree=ast.parse(raw)
    orig=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='qualified')
    text=ast.get_source_segment(raw.decode(),orig)
    require(text.count("terminal['scope']=='n13-next'")==1,'exact original terminal selector changed')
    text=text.replace("terminal['scope']=='n13-next'","terminal['scope']=='n18-next'",1)
    exec(compile(text,'<exact n18 returned-worker proof>', 'exec'),env)
    function=prior['entry']
    inherited=FunctionType(function.__code__,env,function.__name__,function.__defaults__,function.__closure__)
    load,main=inherited()
    inner=main.__globals__
    earlier_api=inner['api']
    def api(material,claim_sha):
        ctx,reader,read,render,dispatch,observe=earlier_api(material,claim_sha)
        older=dispatch.__globals__['publication']
        def publication(c,revision):
            older(c,revision)
            _,publisher=c[3][-2]['modules']()
            require(publisher['git']('cat-file','blob',revision+':'+N13[0])==raw,
                    'published exact guarded implementation differs')
        fn=material[1]['_fn']
        return ctx,reader,read,render,fn(dispatch,publication=publication),observe
    inner['api']=api
    return load,main


if __name__=='__main__':
    _,main=entry()
    main()
