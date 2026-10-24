import pathlib,hashlib,json,ast
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';i=r/'incoming';m=i/'GLM-tile2-component-manifest-r8836.json';assert hashlib.sha256(m.read_bytes()).hexdigest()=='d34dbb3381331ba9d6357118bfdf2745e67b3d090448f9837cc1e8f142a16560'
s=(i/'launch_GLM_tile2_component_r8836.py').read_text();ast.parse(s);exec(s)
