"""Scope the pinned native ModelScope cache path to this one request."""

import ast
from contextlib import contextmanager
from hashlib import sha256
import inspect
import os
from pathlib import Path
import sys


SWIFT_SHA="82ef4b3c78a2ae3be8dcc0e8caee2dde7215f8f877bfd40a99d39f3e9586d252"
MODELSCOPE_SHA="b895b5dfe401aeec99b60684d0b69ffb4084f858a14933310e665a22b7f20d45"
KEY="MODELSCOPE_CACHE"
FUNCTION="get_temporary_cache_files_directory"
GETTER="get_cache_dir"


def need(ok,gate):
    if not ok:
        raise ValueError(gate)


def pin_path(value,root,category):
    if not isinstance(value,(str,bytes,os.PathLike)):
        return {"category":"UNAVAILABLE_OR_ERROR","path_SHA256":None,"within_request_scratch":False}
    path=Path(os.fsdecode(value)).absolute()
    return {"category":category(path),"path_SHA256":sha256(os.fsencode(str(path))).hexdigest(),
            "within_request_scratch":path.is_relative_to(root)}


def source(module,expected,read):
    filename=Path(module.__file__).resolve(strict=True)
    raw=read(filename,filename.stat().st_size,expected,1024*1024)
    return ast.parse(raw)


def definition(tree,name):
    values=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name==name]
    need(len(values)==1,"R31_EXACT_NATIVE_GETTER_DEFINITION")
    return values[0]


def swift_getters(tree,swift,module):
    function=definition(tree,FUNCTION)
    callees={node.func.id for node in ast.walk(function) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)}
    attribute={node.func.attr for node in ast.walk(function) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
               and isinstance(node.func.value,ast.Name) and node.func.value.id in {"utils","cache_utils","modelscope_utils"}}
    choices={actual for item in ast.walk(tree) if isinstance(item,ast.ImportFrom) and item.module=="modelscope.hub.utils.utils"
             for actual in (alias.name for alias in item.names if (alias.asname or alias.name) in callees)}
    choices.update(name for name in attribute if getattr(module,name,None) is not None)
    for name in callees:
        val=getattr(swift,name,None)
        if callable(val) and getattr(val,"__module__",None)==module.__name__:
            choices.add(val.__name__)
    return choices


def getenv_dependency(tree,name):
    aliases={target.id for node in tree.body if isinstance(node,ast.Assign) and
             isinstance(node.value,ast.Constant) and node.value.value==KEY
             for target in node.targets if isinstance(target,ast.Name)}
    wanted={name}
    visited=set()
    while wanted and len(visited)<6:
        current=wanted.pop()
        if current in visited:
            continue
        visited.add(current)
        fn=definition(tree,current)
        for node in ast.walk(fn):
            if isinstance(node,ast.Call) and node.args:
                arg,func=node.args[0],node.func
                if (isinstance(arg,ast.Constant) and arg.value==KEY or isinstance(arg,ast.Name) and arg.id in aliases) and (
                        isinstance(func,ast.Attribute) and func.attr=="getenv" and isinstance(func.value,ast.Name)
                        and func.value.id=="os"):
                    return current
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):
                if any(isinstance(func,ast.FunctionDef) and func.name==node.func.id for func in tree.body):
                    wanted.add(node.func.id)
    raise ValueError("R31_ACTUAL_MODELSCOPE_GETENV_DEPENDENCY")


def setup(temp,report):
    need(not any(name in sys.modules for name in ("swift","modelscope","datasets")),
         "R31_BEFORE_NATIVE_PROVIDER_IMPORT")
    local=temp/"modelscope"
    local.mkdir(mode=0o700)
    need(local.resolve(strict=True)==local and local.is_relative_to(temp),"R31_EXACT_REQUEST_MODELSCOPE_CACHE")
    os.environ[KEY]=str(local)
    report.update(configured_before_import=True,new_child_directory=True)
    return local


def verify(swift_utils,ms,cache,temp,category,read,report):
    swift_tree=source(swift_utils,SWIFT_SHA,read)
    ms_tree=source(ms,MODELSCOPE_SHA,read)
    candidates=swift_getters(swift_tree,swift_utils,ms)
    need(candidates=={GETTER},"R31_EXACT_SWIFT_LINKED_MODELSCOPE_GETTER")
    chosen=GETTER
    dependency=getenv_dependency(ms_tree,chosen)
    getter=getattr(ms,chosen)
    inspect.signature(getter).bind()
    actual=getter()
    projected=pin_path(actual,temp,category)
    outcome=Path(os.fsdecode(actual)).resolve(strict=False)
    need(projected["within_request_scratch"] and outcome.is_relative_to(cache),
         "R31_NATIVE_MODELSCOPE_GETTER_OUTSIDE_REQUEST")
    report.update(swift_utils_SHA256=SWIFT_SHA,modelscope_utils_SHA256=MODELSCOPE_SHA,
                  native_getter=chosen,actual_getenv_provider=dependency,path=projected)
    return getattr(swift_utils,FUNCTION)


@contextmanager
def observe(function,temp,cache,category,report,lane):
    """Observe actual Swift returns in one loader call, without calling the helper early."""
    previous=sys.getprofile()
    events=[]
    def trace(frame,event,arg):
        if frame.f_code is function.__code__ and event=="return" and len(events)<8:
            value=pin_path(arg,temp,category)
            if isinstance(arg,(str,bytes,os.PathLike)):
                resolved=Path(os.fsdecode(arg)).resolve(strict=False)
                value["within_modelscope_override"]=resolved.is_relative_to(cache)
            events.append(value)
    sys.setprofile(trace)
    try:
        yield
    finally:
        sys.setprofile(previous)
        report["swift_tmp_returns"][lane]=events
    need(all(item["within_request_scratch"] and item.get("within_modelscope_override") for item in events),
         "R31_NATIVE_SWIFT_TMP_OUTSIDE_MODELSCOPE_REQUEST_CACHE")
