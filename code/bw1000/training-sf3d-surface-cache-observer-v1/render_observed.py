"""Instrument the exact existing surface route; add no train/model/cache loop."""
import ast
import copy
import hashlib
import importlib.util
from pathlib import Path

D=Path(__file__).resolve().parent;G=D.parents[2]
ROOT_RENDER=G/'code/bw1000/training-sf3d-public-surface-cohort-v1/render_native.py'
BACKEND='generation/src/gencuda/backends/sf3d_surface_training.py'
HELPER='generation/src/gencuda/backends/surface_cache_observer.py'
PIN={'bytes':58679,'SHA256':'c81d32e0103dc7cee323bd434af099cb7267273501dc8334530c621f0dc5300b'}


def digest(raw):return hashlib.sha256(raw).hexdigest()


def render():
    # Check the published caller before importing any of its top-level code.
    if digest(ROOT_RENDER.read_bytes())!='d647811cfab3d835ad28884cf8c306b699806b9d834fefe59cb38d9fb006560d':
        raise ValueError('Published root surface renderer source differs')
    # That renderer additionally checks all its original source dependencies.
    spec=importlib.util.spec_from_file_location('published_sf3d_surface_renderer',ROOT_RENDER)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    overlay=module.render();original=overlay[BACKEND]
    if {'bytes':len(original),'SHA256':digest(original)}!=PIN:raise ValueError('Published surface backend differs')
    tree=ast.parse(original);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_sf3d_training')
    before=copy.deepcopy(fn);instrumented=[]
    def phase(name,statement,index=None):
        args=[ast.Constant(value=name)]
        if index is not None:args.append(ast.parse(index,mode='eval').body)
        return ast.With(items=[ast.withitem(context_expr=ast.Call(func=ast.Attribute(value=ast.Name(id='surface_cache_observer',ctx=ast.Load()),attr='phase',ctx=ast.Load()),args=args,keywords=[]))],body=[statement],type_comment=None)
    class Observe(ast.NodeTransformer):
        def visit_Assign(self,node):
            targets=[x.id for x in node.targets if isinstance(x,ast.Name)]
            if targets==['torch'] and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='require_cuda':
                instrumented.append('device_gate');return phase('device_gate',node)
            if targets==['model'] and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='from_pretrained':
                instrumented.append('model_init');return phase('model_init',node)
            key={'batch':('prepare_batch','_prepare_batch'),'triplane_features':('query_triplane',None),'image_features':('image_features','_image_features')}.get(targets[0] if len(targets)==1 else '')
            call=node.value.value if isinstance(node.value,ast.Subscript) else node.value
            match=key and isinstance(call,ast.Call) and (key[1] is None and isinstance(call.func,ast.Attribute) and call.func.attr=='query_triplane'
                or isinstance(call.func,ast.Name) and call.func.id==key[1])
            if match:
                instrumented.append(key[0]);return phase(key[0],node,'len(cached_batches)')
            if any(isinstance(t,ast.Tuple) and any(isinstance(v,ast.Name) and v.id=='scene_codes' for v in t.elts) for t in node.targets):
                instrumented.append('scene_codes');return phase('scene_codes',node,'len(cached_batches)')
            return self.generic_visit(node)
        def visit_Expr(self,node):
            if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and isinstance(node.value.func.value,ast.Name) and node.value.func.value.id=='cached_batches' and node.value.func.attr=='append':
                instrumented.append('material_fields');return phase('material_fields',node,'len(cached_batches)')
            return self.generic_visit(node)
    Observe().visit(fn)
    model_transfer=ast.dump(ast.parse('model.to("cuda").eval()').body[0],include_attributes=False)
    model_with=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.With)
        and len(n.items)==1 and isinstance(n.items[0].context_expr,ast.Call)
        and n.items[0].context_expr.args and isinstance(n.items[0].context_expr.args[0],ast.Constant)
        and n.items[0].context_expr.args[0].value=='model_init')
    if ast.dump(fn.body[model_with+1],include_attributes=False)!=model_transfer:
        raise ValueError('Exact original model transfer/eval operation missing')
    fn.body[model_with].body.append(fn.body.pop(model_with+1))
    setup=ast.parse('from .surface_cache_observer import SurfaceCacheObserver\nsurface_cache_observer = SurfaceCacheObserver(output_dir, labels_sha256=atlas["labels_sha256"], input_sha256=atlas["input_sha256"], planned_objects=len(atlas["images"]))').body
    pos=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='atlas' for t in n.targets))
    fn.body[pos+1:pos+1]=setup
    validation=ast.parse('''with surface_cache_observer.phase("validate_cache"):
    surface_cache_cpu = _sf3d_capture_cached_batches(cached_batches)
    _sf3d_validate_cached_batches(surface_cache_cpu, len(input_paths), cache_feature_spec, torch)
    surface_cache_observer.validate_actual_cache(surface_cache_cpu, atlas["targets"], cache_feature_spec, torch, expected_points=point_count)
    del surface_cache_cpu
''').body
    pos=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='adapter_sha_before' for t in n.targets))
    fn.body[pos+1:pos+1]=validation
    if sorted(instrumented)!=sorted(['device_gate','model_init','prepare_batch','scene_codes','query_triplane','image_features','material_fields']):
        raise ValueError('Exact seven existing native call sites required')
    # Strip ONLY inserted constructs and prove every original function AST node
    # survives unchanged, including the original optimizer/training loop.
    cleaned=copy.deepcopy(fn)
    class Strip(ast.NodeTransformer):
        def visit_With(self,n):
            if len(n.items)==1 and isinstance(n.items[0].context_expr,ast.Call) and isinstance(n.items[0].context_expr.func,ast.Attribute) and isinstance(n.items[0].context_expr.func.value,ast.Name) and n.items[0].context_expr.func.value.id=='surface_cache_observer':
                if ast.literal_eval(n.items[0].context_expr.args[0])=='validate_cache':return None
                return n.body
            return self.generic_visit(n)
        def visit_ImportFrom(self,n):return None if n.module=='surface_cache_observer' else n
        def visit_Assign(self,n):
            if any(isinstance(t,ast.Name) and t.id=='surface_cache_observer' for t in n.targets):return None
            return self.generic_visit(n)
    Strip().visit(cleaned)
    if ast.dump(cleaned,include_attributes=False)!=ast.dump(before,include_attributes=False):raise ValueError('Original native math/control-flow AST changed')
    raw=(ast.unparse(ast.fix_missing_locations(tree))+'\n').encode()
    ast.parse(raw,feature_version=(3,10))
    overlay[BACKEND]=raw;overlay[HELPER]=(D/'cache_observer.py').read_bytes()
    return overlay,{'original':PIN,'generated_backend':{'bytes':len(raw),'SHA256':digest(raw)},
                    'original_native_AST_after_stripping_instrumentation_identical':True,'instrumented_existing_call_sites':instrumented,
                    'new_training_or_cache_producer_loop':False,'actual_target_execution':False}
