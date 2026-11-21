"""Measure actual Hygon providers, including every decorator; no version/shadow query."""
import ast
import hashlib
import inspect
from pathlib import Path
import types
import sys
from common import need,pin,save,RUN

def definition(function,raw):
    code=function.__code__;found=[]
    def walk(nodes,prefix=''):
        for node in nodes:
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                name=prefix+node.name
                if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
                    start=min([node.lineno]+[d.lineno for d in node.decorator_list])
                    if start==code.co_firstlineno and node.name==code.co_name:found.append((name,node,start))
                walk(node.body,name+('.' if isinstance(node,ast.ClassDef) else '.<locals>.'))
    walk(ast.parse(raw).body);need(len(found)==1,'actual code AST owner missing/ambiguous')
    name,node,start=found[0]
    if hasattr(code,'co_qualname'):need(code.co_qualname==name,'code AST qualified name differs')
    lines=raw.decode('utf8').splitlines(keepends=True)
    return dict(code_qualname=name,first_line=start,end_line=node.end_lineno,
        definition_source_SHA256=hashlib.sha256(''.join(lines[start-1:node.end_lineno]).encode()).hexdigest())

def method_identity(method):
    function=method.__func__ if inspect.ismethod(method) else method
    need(isinstance(function,types.FunctionType),'native Python provider required')
    code=function.__code__;path=Path(code.co_filename).resolve()
    need(path.suffix=='.py' and path.name!='version.py' and path.stat().st_size<=262144,'bounded actual method source')
    need(Path(inspect.getsourcefile(function)).resolve()==path and Path(function.__globals__['__file__']).resolve()==path,'actual source/code/global provider disagree')
    identity=pin(path);raw=path.read_bytes();need(hashlib.sha256(raw).hexdigest()==identity['SHA256'],'provider changed')
    return dict(path=str(path),code_filename=code.co_filename,inspect_sourcefile=inspect.getsourcefile(function),
        module=function.__module__,public_qualname=function.__qualname__,**identity,**definition(function,raw))

def closure(function):return {k:c.cell_contents for k,c in zip(function.__code__.co_freevars,function.__closure__ or ())}

def bind_encode(template,expected_context_type,observation_name='encode-chain-observation.json'):
    need(observation_name in ('encode-chain-observation.json','full-encode-chain-observation.json'),'exact observation namespace')
    method=template.encode;need(inspect.ismethod(method) and method.__self__ is template,'actual bound encode')
    chain=[];function=method.__func__;seen=set()
    while True:
        need(id(function) not in seen and len(chain)<4,'extra/cyclic native encode wrapper')
        seen.add(id(function));chain.append(function)
        if not hasattr(function,'__wrapped__'):break
        nxt=function.__wrapped__;need(closure(function).get('func') is nxt,'wrapper actual closure target changed');function=nxt
    need(len(chain)==3,'native context/retry/Template.encode chain required')
    records=[method_identity(f) for f in chain]
    save(RUN/observation_name,dict(chain=records,qualification_pending=True))
    expected=['context_decorator.<locals>.decorate_context','retry_decorator.<locals>._retry.<locals>.new_func','Template.encode']
    need([r['code_qualname'] for r in records]==expected,'native wrapper structure changed')
    need(records[0]['path'].endswith('/torch/utils/_contextlib.py') and records[1]['path'].endswith('/swift/utils/utils.py')
         and records[2]['path'].endswith('/swift/template/base.py'),'actual wrapper provider roles changed')
    need(all(r['path'].startswith('/swift-runtime/site/swift/') for r in records[1:]),'Swift wrappers must originate in qualified offline runtime')
    factory=closure(chain[0]).get('ctx_factory')
    need(inspect.ismethod(factory) and factory.__name__=='clone' and type(factory.__self__) is expected_context_type,'original inference context clone required')
    need(type(closure(chain[1]).get('retry')) is int and closure(chain[1])['retry']==3,'native retry semantic changed')
    need(chain[2].__module__=='swift.template.base' and chain[2].__qualname__=='Template.encode','native implementation owner')
    for r,role in zip(records,['context','retry','implementation']):r['role']=role
    return dict(wrapper_chain=records,wrapper_order=['context','retry','implementation'],retry_policy=3,actual_retry_count='UNKNOWN',
        invocation_preserves_decorators=True,unwrapped_leaf_called=False,actual_source_qualification_inherited=False)

def model_source_identities(model, expected, persist):
    """Observe all encountered mathematical classes before checking fixed target pins."""
    records=[];errors=[];seen=set();enumeration_complete=False
    observation=dict(schema='ACTUAL_TARGET_MATH_PROVIDER_OBSERVATION_V1',
        planned_prefixes=sorted(expected),records=records,errors=errors,
        enumeration_complete=False,validation_attempted=False,numerical_equivalence_proven=False)
    try:
        for module_name,module in model.named_modules():
            cls=type(module);name=cls.__module__
            for prefix in expected:
                key=(prefix,name,cls.__qualname__)
                if not name.startswith(prefix) or key in seen:continue
                seen.add(key)
                record=dict(prefix=prefix,model_module_name=module_name,module=name,class_qualname=cls.__qualname__)
                records.append(record)
                try:
                    loaded=sys.modules.get(name)
                    module_file=getattr(loaded,'__file__',None)
                    inspect_file=inspect.getsourcefile(cls)
                    record.update(module_file=module_file,inspect_sourcefile=inspect_file)
                    need(isinstance(module_file,str) and isinstance(inspect_file,str),'actual mathematical source path missing')
                    path=Path(inspect_file).resolve(strict=True)
                    record.update(resolved_path=str(path),resolved_module_path=str(Path(module_file).resolve(strict=True)))
                    need(path.suffix=='.py' and path.stat().st_size<=262144,'bounded mathematical Python source')
                    record.update(pin(path))
                except Exception as error:
                    errors.append(dict(prefix=prefix,module=name,class_qualname=cls.__qualname__,
                        error_type=type(error).__name__,reason_SHA256=hashlib.sha256(str(error).encode()).hexdigest()))
        enumeration_complete=True
    finally:
        observation['enumeration_complete']=enumeration_complete
        persist(observation)
    need(not errors and enumeration_complete,'actual mathematical source observations incomplete')
    found={}
    for prefix,expected_pin in expected.items():
        rows=[r for r in records if r['prefix']==prefix]
        need(bool(rows),'required mathematical module prefix absent')
        need(all(r['resolved_path']==r['resolved_module_path'] for r in rows),'actual module and inspected class source disagree')
        identities={(r['resolved_path'],r['bytes'],r['SHA256']) for r in rows}
        need(len(identities)==1,'mixed mathematical source identities within one prefix')
        path,size,digest=next(iter(identities))
        need(dict(bytes=size,SHA256=digest)==expected_pin,'reviewed target mathematical source contract differs')
        found[prefix]=dict(path=path,bytes=size,SHA256=digest)
    need(set(found)==set(expected),'complete target InternVL/Qwen3 source closure')
    return found
