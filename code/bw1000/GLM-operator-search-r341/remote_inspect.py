import pathlib, subprocess, json
r=pathlib.Path.home()/'hygon-inference-refactor-20261014'
paths=[r/'candidates/glm-hybrid-decode-r5386',r/'candidates/GLM-irregular-vector-r9270b']
q={}
for p in paths:
 q[str(p)]={str(f):'\n'.join(str(i)+':'+s for i,s in enumerate(f.read_text(errors='replace').splitlines()) if any(k in s for k in ['silu','Silu','quant','activation']))[:10000] for f in p.iterdir() if f.suffix in ['.cu','.cpp']}
print(json.dumps(q))
