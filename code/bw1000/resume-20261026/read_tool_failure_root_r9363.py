import pathlib,json,re,hashlib,datetime
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');out={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'read_only':True,'logs':[]}
for name in ['toolcall-Qwen06-CPU-r136-910374.1135.log','toolcall-Qwen06-CPU-r136-client.log']:
 p=r/'logs'/name;q={'path':str(p),'exists':p.exists()}
 if p.exists():
  s=p.read_text(errors='replace');lines=s.splitlines();q.update(SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),line_count=len(lines),error_lines=[x[:1400]for x in lines if re.search('error|assert|abort|terminate|Traceback|ImportError|device|No such|not found|ModuleNotFound',x,re.I)][-30:])
 out['logs'].append(q)
out['candidate_exists']=(r/'candidates/toolcall-Qwen06-r136').exists();print(json.dumps(out))
