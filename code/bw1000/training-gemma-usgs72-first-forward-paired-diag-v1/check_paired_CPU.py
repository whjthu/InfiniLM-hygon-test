"""Narrow source, control/tracer wiring and pair-counter fixtures; no Torch/Slurm."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from contract import D,ASSET,REQUEST,NODE,PORTS,validate_lease
from common_rng_tap import CommonRNGTap
from light_trace import install_if_enabled
from native_entry import derive_plan
from pair_compare import compare_pair


def write(root,relative,data):
    p=root/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data),encoding='utf8')


def synthetic_pair(root):
    state=lambda n:{kind:dict(SHA256=kind+str(n),finite=True) for kind in ('adapter','optimizer','scheduler')}
    for rank in range(8):
        for arm,offset in [('untraced',0),('traced',100)]:
            folder=Path(arm)/'train/resume-witness'/('rank-%02d'%rank)
            result=dict(rank=rank,world_size=8,completed_global_step=25,native_resume_window_complete=True,
                saved_checkpoints=[25],RNG_loads=[dict(saved_equal=True,global_step=20,actual={'SHA256':'load'+str(rank)})],
                inputs=[dict(ordinal=i+1,global_step=20+i//4,full_input={'SHA256':'in'+str(rank)+'-'+str(i),'finite':True}) for i in range(20)],
                outputs=[dict(ordinal=i+1,loss={'SHA256':'loss'+str(rank)+'-'+str(i),'finite':True},loss_scalar=float(i)/100) for i in range(20)],
                steps=[dict(global_step=i,state=state(i)) for i in range(21,26)])
            write(root,folder/'result.json',result)
            write(root,folder/'restored-before-RNG.json',dict(restored_parameter_optimizer_scheduler_equal=True,actual=state(20)))
            write(root,folder/'native-data-skip.json',dict(rank=rank,before=100,skipped=80,after=20))
            write(root,folder/'owner.json',dict(rank=rank,local_rank=rank,PID=100+rank+offset,startticks=1000+rank+offset))
            write(root,folder/'checkpoint-25-rng.json',dict(RNG={'SHA256':'final'+str(rank)}))
            tap=[dict(rank=rank,ordinal=i,event=event,global_step=20,RNG={'SHA256':'rng'+str(rank)+'-'+str(i)+event})
                 for i in range(1,5) for event in ('entry','exit')]
            write(root,Path(arm)/'common-rng'/('rank-%02d'%rank)/'result.json',dict(complete=True,rows=tap))


class Handle:
    def __init__(self,rows,item):self.rows,self.item=rows,item
    def remove(self):self.rows.remove(self.item)


class Model:
    def __init__(self):self.pre=[];self.post=[];self.counter=0;self.output=object()
    def register_forward_pre_hook(self,hook,**kwargs):self.pre.append(hook);return Handle(self.pre,hook)
    def register_forward_hook(self,hook,**kwargs):self.post.append(hook);return Handle(self.post,hook)
    def forward(self):
        for hook in self.pre:assert hook(self,(),{}) is None
        self.counter+=1
        for hook in self.post:assert hook(self,(),{},self.output) is None
        return self.output


class Paired(unittest.TestCase):
    def test_base_recipe_only_output_observer_port_change(self):
        base=json.loads((D/'BASE_RESUMED_PLAN.json').read_bytes())
        for arm in ('untraced','traced'):
            root=Path('/new-scope')/arm
            plan=derive_plan(base,root,Path('/new-scope/src/diagnostic_plugin.py'),PORTS[arm])
            original=base['command'];actual=plan['command']
            allowed={original.index('--output_dir')+1,original.index('--external_plugins')+1}
            self.assertEqual(len(original),len(actual))
            self.assertTrue(all(a==b for i,(a,b) in enumerate(zip(original,actual)) if i not in allowed))
            self.assertEqual({k:v for k,v in plan['environment_overrides'].items() if k not in ('OUTPUT_DIR','MASTER_PORT')},
                             {k:v for k,v in base['environment_overrides'].items() if k not in ('OUTPUT_DIR','MASTER_PORT')})
            self.assertEqual(actual[actual.index('--resume_from_checkpoint')+1],(ASSET/'continuous/train/checkpoint-20').as_posix())

    def test_real_additive_subclass_body_control_and_trace(self):
        tree=ast.parse((D/'diagnostic_plugin.py').read_bytes())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
        for arm in ('untraced','traced'):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                for name in ('common-rng','light-trace','diagnostic-hooks'):(root/name).mkdir()
                events=[];model=Model();trainer=SimpleNamespace(state=SimpleNamespace(global_step=20),model_wrapped=None,model=model)
                class Base:
                    def on_train_begin(self,args,state,control,**kwargs):
                        events.append('original_begin');self.witness=SimpleNamespace(check=lambda:None);return control
                    def on_train_end(self,args,state,control,**kwargs):events.append('original_end');return control
                class Backend:
                    def __init__(self,c,m,t,r,check):self.model=m;self.trainer=t;self.check=check
                    def rng_snapshot(self):return {'SHA256':hashlib.sha256(str(self.model.counter).encode()).hexdigest()}
                    def metadata(self,*_):return dict(context={'mode':'synthetic_train'},input_layouts={})
                cfg=dict(arm=arm,trace=dict(enabled=arm=='traced',schema='VLM_FIRST_FORWARD_LIGHT_TRACE_V1',
                    root_GO=True,trace_GO=True,diagnostic_only=True,request_id='a'*32,ranks=[0,1],phase='resumed',
                    start_ordinal=1,count=4,scope_root=str(root),output=str(root/'light-trace')))
                scope=dict(original=SimpleNamespace(GemmaDomainResume=Base),ROOT=root,ARM=arm,
                    stable_json=lambda _: (cfg,{'SHA256':'b'*64}),os=os,json=json,need=lambda ok,msg:None if ok else (_ for _ in ()).throw(ValueError(msg)),
                    CommonRNGTap=CommonRNGTap,NativeBackend=Backend,install_if_enabled=install_if_enabled)
                exec(compile(ast.Module(body=[cls],type_ignores=[]),'actual-additive-class','exec'),scope)
                obj=scope['GemmaPairedDiagnostic']();obj.trainer=trainer;control=object()
                with patch.dict(os.environ,{'RANK':'0','GEMMA_DIAGNOSTIC_CONFIG_SHA256':'b'*64}):
                    self.assertIs(obj.on_train_begin(None,None,control,model=model),control)
                    for _ in range(20):self.assertIs(model.forward(),model.output)
                    self.assertIs(obj.on_train_end(None,None,control),control)
                self.assertEqual(events,['original_begin','original_end'])
                self.assertEqual(model.counter,20)
                tap=json.loads((root/'common-rng/rank-00/result.json').read_bytes());self.assertTrue(tap['complete'])
                self.assertEqual(len(tap['rows']),8)
                self.assertEqual((root/'light-trace/rank-00').exists(),arm=='traced')

    def test_pair_exact_loss_mismatch_and_rng_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);synthetic_pair(root);good=compare_pair(root)
            self.assertTrue(good['observed_noninterference_exact'])
            self.assertTrue(good['same_independent_input_rng_qualified'])
            self.assertEqual((good['compared_input_loss_pairs'],good['compared_state_pairs'],good['compared_first4_RNG_pairs']),(160,40,64))
            p=root/'traced/train/resume-witness/rank-00/result.json';data=json.loads(p.read_bytes())
            data['outputs'][0]['loss_scalar']+=.125;p.write_text(json.dumps(data),encoding='utf8')
            bad=compare_pair(root);self.assertFalse(bad['observed_noninterference_exact'])
            self.assertTrue(bad['same_independent_input_rng_qualified']);self.assertEqual(bad['compared_input_loss_pairs'],160)
            p=root/'traced/common-rng/rank-00/result.json';data=json.loads(p.read_bytes())
            data['rows'][0]['RNG']={'SHA256':'changed'};p.write_text(json.dumps(data),encoding='utf8')
            self.assertFalse(compare_pair(root)['same_independent_input_rng_qualified'])

    def test_fresh_full_HOST_required_for_each_arm(self):
        external=dict(pci=list(range(8)),boot_id='boot')
        for arm in ('untraced','traced'):
            gate=dict(node=NODE,step='1103606.999',request_id=REQUEST,master_port=PORTS[arm],master_port_available=True,
                kfd_holder_count=0,full_fd_inventory_performed=True,pci=list(range(8)),boot_id='boot',monotonic_ns=100)
            validate_lease(gate,external,arm,101,'1103606.999')
            for change in ({'full_fd_inventory_performed':False},{'kfd_holder_count':1},{'master_port':1}):
                with self.assertRaises(ValueError):validate_lease(dict(gate,**change),external,arm,101,'1103606.999')
            with self.assertRaises(ValueError):validate_lease(gate,external,arm,45_000_000_101,'1103606.999')

    def test_clean_I_S_arm_process_resolution(self):
        for arm in ('untraced','traced'):
            code='import sys,runpy;sys.path.insert(0,'+repr(str(D))+');'
            code+=';'.join('runpy.run_path('+repr(str(D/n))+',run_name="source_only")' for n in
                          ('supervisor.py','native_entry.py','controller.py','read_terminal.py'))
            code+=';import contract;assert contract.ROOT.name=='+repr(arm)+';assert "torch" not in sys.modules;print("ARM_IMPORT_PASS")'
            x=subprocess.run([sys.executable,'-I','-S','-B','-c',code],env=dict(os.environ,GEMMA_LIGHT_DIAG_ARM=arm),capture_output=True,timeout=10)
            self.assertEqual(x.returncode,0,x.stderr.decode());self.assertEqual(x.stdout.strip(),b'ARM_IMPORT_PASS')


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Paired))
    print(json.dumps(dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        fixture_only=True,Torch_imported='torch' in sys.modules,Slurm_calls=0,GPU_calls=0,actual_updates=0,
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in D.glob('*.py')}),sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)
