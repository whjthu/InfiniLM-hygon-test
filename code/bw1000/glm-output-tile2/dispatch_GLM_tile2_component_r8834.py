import pathlib,hashlib,json,ast
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';i=r/'incoming';m=i/'GLM-tile2-component-manifest-r8834.json';assert hashlib.sha256(m.read_bytes()).hexdigest()=='82374beed6f6a0421b7ca5b5f525a89e8a12e9e7b5fa019a4b6dfb8c7923ac3f'
s=(i/'launch_GLM_tile2_component_r8834.py').read_text();ast.parse(s);exec(s)
