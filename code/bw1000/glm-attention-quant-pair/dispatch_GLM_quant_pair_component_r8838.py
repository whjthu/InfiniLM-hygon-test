import pathlib,hashlib,json,ast
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';i=r/'incoming';m=i/'GLM-quant-pair-component-manifest-r8838.json';assert hashlib.sha256(m.read_bytes()).hexdigest()=='170be398d96d0a9fa62d2476c19c87bcba3609a56a4db15f4bb0d16d1ddabbc8'
s=(i/'launch_GLM_quant_pair_component_r8838.py').read_text();ast.parse(s);exec(s)
