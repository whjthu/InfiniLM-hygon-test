import pathlib,json
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');q={}
for tag in ['r9042','r9042a']:
 p=r/f'candidates/Kimi-rowreduce-group-{tag}/command0.log';ls=p.read_text().splitlines();indices=[i for i,s in enumerate(ls)if any(x in s for x in ['error:','fatal error:'])];q[tag]={'path':str(p),'errors':[{'line':i+1,'text':'\n'.join(ls[max(0,i-3):i+4])}for i in indices]}
print(json.dumps(q))
