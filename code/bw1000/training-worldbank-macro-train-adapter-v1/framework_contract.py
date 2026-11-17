"""Inspect exact existing common-framework API without importing it or native providers."""
import ast
import hashlib

COMMON_SHA='b218b9a6c1b5b80aaf6df5a73006b06c1529a30a77b1e4ed7b6a8a1d7a9b4f96'
VLM_SHA='71417958a659380bfeea4edace29e9948b2ce549ef3e1dd20e5b9dc17fc402c2'


def inspect_sources(common,vlm):
    if hashlib.sha256(common).hexdigest()!=COMMON_SHA or hashlib.sha256(vlm).hexdigest()!=VLM_SHA:
        raise ValueError('Exact common-framework source not available; do not guess provider API')
    c=ast.parse(common);v=ast.parse(vlm)
    calls=[n for n in ast.walk(c) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    if not any(isinstance(n.func.value,ast.Name) and n.func.value.id=='vlm' and n.func.attr=='main' for n in calls):
        raise ValueError('Unified T01 delegation changed')
    build=next(n for n in v.body if isinstance(n,ast.FunctionDef) and n.name=='build_launch')
    expected=['case','original','swift','models','dataset','plugins','output','master_port']
    if [a.arg for a in build.args.args]!=expected:raise ValueError('Common build_launch interface changed')
    return dict(common_source_SHA256=COMMON_SHA,vlm_source_SHA256=VLM_SHA,unified_entry='tools/hygon_training_recipe.py',
        VLM_dataset_entry='tools/hygon_vlm_recipe.py:build_launch',dataset_argument='dataset',
        data_format='Swift messages/images JSONL',source_api_verified_only=True,current_target_installation_verified=False,
        training_loop_added=False,formal_T01_registry_modified=False)
