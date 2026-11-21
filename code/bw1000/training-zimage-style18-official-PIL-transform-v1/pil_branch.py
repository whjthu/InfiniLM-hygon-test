"""Audited PIL-only source projection, not a fake installed Torch/TorchVision."""
import ast
import copy
from enum import Enum
import hashlib
import numbers
from pathlib import Path
from types import SimpleNamespace
import warnings

SOURCE_PINS={
    'functional.py':(67861,'0b2fdf8e6764f4be0113acd95b58d3b14158c13fa2374ed93fbbe1f4b8730ded'),
    '_functional_pil.py':(12118,'a30262323eefa9351540515b1ff78f218ecff2ce3f6ed957b271948cc8c9a276'),
    'LICENSE':(1517,'6502f676851cfe25f8af75531dfb32375b7325b73c37e7b43741fa422893e71d')}
OPS_PIN=(7735,'c02a73ebfc8bef63f175fe54882d0faf4224d7552ea852df75b27823a8f235a4')
LOG_TEST=ast.dump(ast.parse('not torch.jit.is_scripting() and not torch.jit.is_tracing()',mode='eval').body,include_attributes=False)
TENSOR_TEST=ast.dump(ast.parse('isinstance(img, torch.Tensor)',mode='eval').body,include_attributes=False)


def need(ok,why):
    if not ok:raise ValueError(why)


def raw(path,pin):
    b=Path(path).read_bytes()
    need((len(b),hashlib.sha256(b).hexdigest())==pin,'Exact source identity differs')
    return b


def digest(node):return hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()


class SpecializePIL(ast.NodeTransformer):
    def __init__(self):self.logs_removed=0;self.tensor_predicates_specialized=0
    def visit_If(self,node):
        if ast.dump(node.test,include_attributes=False)==LOG_TEST:
            need(len(node.body)==1 and not node.orelse and isinstance(node.body[0],ast.Expr)
                 and isinstance(node.body[0].value,ast.Call) and isinstance(node.body[0].value.func,ast.Name)
                 and node.body[0].value.func.id=='_log_api_usage_once','Unexpected nonlogging JIT block')
            self.logs_removed+=1;return None
        return self.generic_visit(node)
    def visit_Call(self,node):
        if ast.dump(node,include_attributes=False)==TENSOR_TEST:
            self.tensor_predicates_specialized+=1
            return ast.copy_location(ast.Constant(value=False),node)
        return self.generic_visit(node)


def compile_nodes(nodes,namespace,label):
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future]+nodes,type_ignores=[])),label,'exec'),namespace)


def build(source_root,operators):
    import numpy as np
    from PIL import Image,ImageOps
    sources={k:raw(Path(source_root)/k,v) for k,v in SOURCE_PINS.items()}
    trees={k:ast.parse(v) for k,v in sources.items() if k.endswith('.py')}
    proof={'official_source_tag':'v0.24.0','target_installed_version_or_source_binding_qualified':False,
           'PIL_only_namespace_projection':True,'Torch_or_TorchVision_package_imported':False,'functions':{}}
    pil_names={'_is_pil_image','get_dimensions','get_image_size','get_image_num_channels','crop','resize','pad','_parse_fill'}
    pil=[]
    for original in trees['_functional_pil.py'].body:
        if not isinstance(original,ast.FunctionDef) or original.name not in pil_names:continue
        n=copy.deepcopy(original)
        need([ast.unparse(x) for x in n.decorator_list]==['torch.jit.unused'],'Only original TorchScript registration decorator may be removed')
        n.decorator_list=[];pil.append(n)
        proof['functions']['PIL.'+n.name]={'line':original.lineno,'end_line':original.end_lineno,
             'original_AST_SHA256':digest(original),'executed_AST_SHA256':digest(n),'body_AST_SHA256':digest(ast.Module(body=n.body,type_ignores=[])),
             'only_decorator_removed':'torch.jit.unused','body_unchanged':ast.dump(ast.Module(body=original.body,type_ignores=[]),include_attributes=False)==ast.dump(ast.Module(body=n.body,type_ignores=[]),include_attributes=False)}
    need({n.name for n in pil}==pil_names,'PIL native dependency closure missing')
    pns={'np':np,'numbers':numbers,'Image':Image,'ImageOps':ImageOps,'accimage':None}
    compile_nodes(pil,pns,'official-v0.24.0-PIL-functions-no-JIT-registration')
    pil_api=SimpleNamespace(**{name:pns[name] for name in pil_names})
    outer_names={'get_dimensions','_compute_resized_output_size','_interpolation_modes_from_int','resize','center_crop','crop','pad'}
    outer=[];special=SpecializePIL()
    for original in trees['functional.py'].body:
        keep=(isinstance(original,ast.FunctionDef) and original.name in outer_names) or (isinstance(original,ast.ClassDef) and original.name=='InterpolationMode')
        keep=keep or (isinstance(original,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='pil_modes_mapping' for x in original.targets))
        if not keep:continue
        n=special.visit(copy.deepcopy(original));outer.append(n)
        name=getattr(n,'name','pil_modes_mapping')
        proof['functions']['functional.'+name]={'line':original.lineno,'end_line':original.end_lineno,
             'original_AST_SHA256':digest(original),'executed_AST_SHA256':digest(n)}
    need({n.name for n in outer if isinstance(n,ast.FunctionDef)}==outer_names,'Outer PIL route dependency closure missing')
    need(special.logs_removed==5 and special.tensor_predicates_specialized==4,'Closed JIT/Tensor specialization shape differs')
    # No torch object/module/stub exists. Inputs are strictly real PIL images;
    # only the now-unreachable original F_t return statements remain in source.
    need(not any(isinstance(n,ast.Name) and n.id=='torch' for x in outer for n in ast.walk(x)),
         'Unresolved Torch operation in specialized PIL-only code')
    ons={'np':np,'numbers':numbers,'warnings':warnings,'Enum':Enum,'F_pil':pil_api}
    compile_nodes(outer,ons,'official-v0.24.0-explicit-PIL-only-control-flow')
    trace=[]
    def invoke(name,*args,**kwargs):
        need(args and isinstance(args[0],Image.Image),'PIL source projection refuses every non-PIL input')
        result=ons[name](*args,**kwargs)
        trace.append({'name':name,'input_size':list(args[0].size),'output_size':list(result.size)})
        return result
    resize=lambda *a,**k:invoke('resize',*a,**k)
    crop=lambda *a,**k:invoke('center_crop',*a,**k)
    # Explicit data-method namespace projection, not sys.modules torchvision.
    api=SimpleNamespace(transforms=SimpleNamespace(functional=SimpleNamespace(resize=resize,center_crop=crop),InterpolationMode=ons['InterpolationMode']))
    ops=raw(operators,OPS_PIN);tree=ast.parse(ops)
    nodes=[copy.deepcopy(n) for n in tree.body if isinstance(n,ast.ClassDef) and n.name in {'DataProcessingOperator','ImageCropAndResize'}]
    need(len(nodes)==2,'Original DiffSynth transform classes missing')
    ns={'torchvision':api,'Image':Image}
    compile_nodes(nodes,ns,'original-DiffSynth-transform-on-explicit-official-PIL-route')
    original_transform=ns['ImageCropAndResize'](height=256,width=256,height_division_factor=16,width_division_factor=16)
    def transform(image):
        need(isinstance(image,Image.Image) and image.mode=='RGB' and min(image.size)>=2,'Real full RGB PIL input required')
        return original_transform(image)
    proof.update(logging_JIT_blocks_removed=special.logs_removed,Tensor_predicates_specialized_false=special.tensor_predicates_specialized,
        native_DiffSynth_classes_AST_SHA256=digest(ast.Module(body=nodes,type_ignores=[])),
        original_DiffSynth_class_bodies_changed=False,Tensor_or_padding_branch_runtime_claim=False)
    return transform,trace,proof,invoke
