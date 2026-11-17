"""One meaningful stdlib-only wrapper identity fixture; no actual Torch/Swift or native processor."""
import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import tempfile

from wrapper_binding import bind_encode

OUTER='''import functools
class Context:
    def __init__(self):self.enters=0
    def clone(self):return self
    def __enter__(self):self.enters+=1
    def __exit__(self,*args):pass
def context_decorator(ctx,func):
    ctx_factory=ctx
    @functools.wraps(func)
    def decorate_context(*args,**kwargs):
        with ctx_factory():return func(*args,**kwargs)
    return decorate_context
'''
RETRY='''from functools import wraps
def retry_decorator(retry=3):
    def _retry(func):
        @wraps(func)
        def new_func(*args,**kwargs):
            for i in range(retry):
                try:return func(*args,**kwargs)
                except Exception:
                    if i==retry-1:raise
        return new_func
    return _retry
'''
LEAF='''class Template:
    def encode(self,value):return value+1
'''


def main(output):
    if output.exists():raise ValueError('Original CPU boundary result exists')
    rejected=[];failures=[]
    def bad(name,action):
        try:action()
        except ValueError:rejected.append(name)
        else:failures.append(name)
    with tempfile.TemporaryDirectory(dir=output.parent,prefix='wrapper-only-') as td:
        root=Path(td)
        def load(name,text):
            path=root/(name+'.py');path.write_text(text,encoding='utf8')
            spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            return module,path
        outer,op=load('fixture_outer',OUTER);retry,rp=load('fixture_retry',RETRY);leaf,lp=load('fixture_leaf',LEAF)
        original=leaf.Template.encode;context=outer.Context()
        middle=retry.retry_decorator(3)(original);wrapped=outer.context_decorator(context.clone,middle);leaf.Template.encode=wrapped
        template=leaf.Template()
        policy=dict(leaf_module='fixture_leaf',chain=[])
        for role,path,qualname in [('outer',op,'context_decorator.<locals>.decorate_context'),('retry',rp,'retry_decorator.<locals>._retry.<locals>.new_func'),('implementation',lp,'Template.encode')]:
            raw=path.read_bytes();policy['chain'].append(dict(role=role,path=str(path),bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),code_qualname=qualname))
        observed=bind_encode(template,policy,outer.Context)
        assert len(observed['wrapper_chain'])==3 and observed['closure_targets_match_wrapped'] is True
        assert observed['wrapper_chain'][0]['file_SHA256']!=observed['unwrap_implementation']['file_SHA256']
        assert observed['method_source_SHA256']==hashlib.sha256(inspect.getsource(original).encode()).hexdigest()
        assert template.encode(7)==8 and context.enters==1 and template.encode.__func__ is wrapped
        valid=1
        altered=copy.deepcopy(policy);altered['chain'][0]['SHA256']='0'*64
        bad('unknown_outer_fullfile_SHA',lambda:bind_encode(template,altered,outer.Context))
        altered=copy.deepcopy(policy);altered['chain'][1]['path']=str(op)
        bad('wrong_wrapper_sourcefile',lambda:bind_encode(template,altered,outer.Context))
        altered=copy.deepcopy(policy);altered['leaf_module']='unbound_leaf'
        bad('wrong_implementation_module',lambda:bind_encode(template,altered,outer.Context))
        altered=copy.deepcopy(policy);altered['chain'][0]['code_qualname']='unexpected_wrapper'
        bad('unknown_wrapper_code_qualname',lambda:bind_encode(template,altered,outer.Context))
        class OtherContext:pass
        bad('wrong_context_factory_type',lambda:bind_encode(template,policy,OtherContext))
        wrapped.__wrapped__=original
        bad('wrapped_attribute_not_actual_func_closure',lambda:bind_encode(template,policy,outer.Context));wrapped.__wrapped__=middle
        cells=dict(zip(middle.__code__.co_freevars,middle.__closure__));old=cells['retry'].cell_contents;cells['retry'].cell_contents=4
        bad('native_retry_count_changed',lambda:bind_encode(template,policy,outer.Context));cells['retry'].cell_contents=old
        leaf.Template.encode=outer.context_decorator(context.clone,wrapped)
        bad('extra_unregistered_wrapper',lambda:bind_encode(template,policy,outer.Context));leaf.Template.encode=wrapped
        leaf.Template.encode=outer.context_decorator(context.clone,original)
        bad('missing_retry_wrapper',lambda:bind_encode(template,policy,outer.Context));leaf.Template.encode=wrapped
        original_cell=dict(zip(wrapped.__code__.co_freevars,wrapped.__closure__))['func'];old=original_cell.cell_contents
        original_cell.cell_contents=wrapped;wrapped.__wrapped__=wrapped
        bad('cyclic_wrapped_chain',lambda:bind_encode(template,policy,outer.Context));original_cell.cell_contents=old;wrapped.__wrapped__=middle
        class NotBound:encode=staticmethod(wrapped)
        bad('not_instance_bound_method',lambda:bind_encode(NotBound(),policy,outer.Context))
        assert not failures
    result=dict(passed=True,valid_cases=valid,rejected_cases=len(rejected),rejected=rejected,failures=failures,
        actual_native_processor_runs=0,actual_Torch_or_Swift_imports=0,actual_original_source_execution=False,
        fixture_is_stdlib_synthetic_only=True,actual_model_or_media_or_HTTP_calls=0)
    with output.open('x',encoding='utf8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({k:result[k] for k in ('passed','valid_cases','rejected_cases','actual_native_processor_runs','actual_Torch_or_Swift_imports')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
