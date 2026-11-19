"""Pure-stdlib identity audit; preserve and call the actual decorated method, never its leaf."""
import ast
import hashlib
import inspect
from pathlib import Path
import types


def need(ok,why):
    if not ok:raise ValueError('Wrapper binding: '+why)


def _closure(function):
    return {name:cell.cell_contents for name,cell in zip(function.__code__.co_freevars,function.__closure__ or ())}


def _definition(raw,qualname):
    """Find exact static code owner without inspect.getsource's implicit unwrap."""
    found=[]
    def walk(nodes,prefix=''):
        for node in nodes:
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                name=prefix+node.name
                if name==qualname:found.append(node)
                child=name+('.<locals>.' if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) else '.')
                walk(node.body,child)
    walk(ast.parse(raw).body)
    need(len(found)==1,'static code definition missing or ambiguous')
    node=found[0];start=min([node.lineno]+[d.lineno for d in getattr(node,'decorator_list',[])])
    lines=raw.decode('utf8').splitlines(keepends=True)
    return dict(first_line=start,definition_line=node.lineno,end_line=node.end_lineno,
        definition_source_SHA256=hashlib.sha256(''.join(lines[start-1:node.end_lineno]).encode()).hexdigest())


def _identity(function,pin,qualname):
    need(isinstance(function,types.FunctionType),'chain element is not Python function')
    expected=Path(pin['path']).resolve();code=function.__code__
    need(Path(code.co_filename).resolve()==expected and Path(inspect.getsourcefile(function)).resolve()==expected,
        'actual code filename/sourcefile outside exact provider')
    need(Path(function.__globals__['__file__']).resolve()==expected,'function globals provider differs')
    before=expected.stat();need(expected.is_file() and not expected.is_symlink() and before.st_size==pin['bytes']<=131072,'provider file path/size differs')
    raw=expected.read_bytes();after=expected.stat()
    need((before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
        and hashlib.sha256(raw).hexdigest()==pin['SHA256'],'provider full SHA changed')
    need(code.co_qualname==qualname,'actual code qualname differs')
    definition=_definition(raw,qualname)
    need(code.co_firstlineno==definition['first_line'],'actual code first line differs')
    return dict(path=expected.as_posix(),bytes=len(raw),file_SHA256=pin['SHA256'],code_filename=code.co_filename,
        inspect_sourcefile=str(inspect.getsourcefile(function)),code_name=code.co_name,code_qualname=code.co_qualname,
        code_bytecode_SHA256=hashlib.sha256(code.co_code).hexdigest(),**definition)


def bind_encode(template,policy,expected_context_type):
    method=template.encode
    need(inspect.ismethod(method) and method.__self__ is template,'encode is not the native instance bound method')
    outer=method.__func__;chain=[];seen=set();function=outer
    while True:
        need(id(function) not in seen and len(chain)<4,'cycle or extra wrapper in encode chain')
        seen.add(id(function));chain.append(function)
        if not hasattr(function,'__wrapped__'):break
        next_function=function.__wrapped__
        need(_closure(function).get('func') is next_function,'actual closure call target differs from __wrapped__')
        function=next_function
    need(len(chain)==3 and len(policy['chain'])==3,'exact outer/retry/implementation chain required')
    closure0=_closure(chain[0]);factory=closure0.get('ctx_factory')
    need(inspect.ismethod(factory) and factory.__name__=='clone' and type(factory.__self__) is expected_context_type,
        'outer context factory is not original inference-mode clone')
    closure1=_closure(chain[1]);need(type(closure1.get('retry')) is int and closure1['retry']==3,'native retry count changed')
    leaf=chain[-1]
    unwrapped=inspect.unwrap(method)
    if inspect.ismethod(unwrapped):unwrapped=unwrapped.__func__
    need(unwrapped is leaf,'unwrap implementation does not match actual closure leaf')
    need(leaf.__module__==policy['leaf_module'] and leaf.__qualname__=='Template.encode','native implementation owner differs')
    identities=[]
    for function,spec in zip(chain,policy['chain']):
        value=_identity(function,spec,spec['code_qualname']);value['role']=spec['role'];identities.append(value)
    return dict(binding_kind='EXACT_DECORATED_NATIVE_ENCODE_CHAIN',wrapper_chain=identities,
        wrapper_order=[v['role'] for v in policy['chain']],closure_targets_match_wrapped=True,
        context_factory_class=type(factory.__self__).__module__+'.'+type(factory.__self__).__qualname__,
        context_factory_method=factory.__qualname__,retry_count=3,unwrap_implementation=identities[-1],
        method_source_SHA256=hashlib.sha256(inspect.getsource(method).encode()).hexdigest(),
        native_bound_owner=type(template).__module__+'.'+type(template).__qualname__,
        invocation_preserves_decorators=True,unwrapped_leaf_called=False)
