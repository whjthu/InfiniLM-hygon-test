"""Measure actual Hygon providers, including every decorator; no version/shadow query."""
import ast
import hashlib
import inspect
from pathlib import Path
import types
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

def bind_encode(template,expected_context_type):
    method=template.encode;need(inspect.ismethod(method) and method.__self__ is template,'actual bound encode')
    chain=[];function=method.__func__;seen=set()
    while True:
        need(id(function) not in seen and len(chain)<4,'extra/cyclic native encode wrapper')
        seen.add(id(function));chain.append(function)
        if not hasattr(function,'__wrapped__'):break
        nxt=function.__wrapped__;need(closure(function).get('func') is nxt,'wrapper actual closure target changed');function=nxt
    need(len(chain)==3,'native context/retry/Template.encode chain required')
    records=[method_identity(f) for f in chain]
    save(RUN/'encode-chain-observation.json',dict(chain=records,qualification_pending=True))
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

def model_source_identities(model,expected):
    found={}
    for _,module in model.named_modules():
        name=type(module).__module__
        for prefix,digest in expected.items():
            if name.startswith(prefix) and prefix not in found:
                path=Path(inspect.getsourcefile(type(module))).resolve()
                row=pin(path);need(row['SHA256']==digest,'original HF model mathematical provider source differs')
                found[prefix]=dict(path=str(path),**row)
    need(set(found)==set(expected),'complete original InternVL/Qwen3 mathematical source closure')
    return found
