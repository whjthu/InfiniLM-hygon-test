from pathlib import Path
import json,hashlib,ast,base64,subprocess,sys
r=Path.cwd();b=r/'Deep-eos-compaction-private-r9130b';c=r/'Deep-eos-compaction-private-r9130c';c.mkdir(exist_ok=False)
for p in b.iterdir():
 if p.is_file():
  text=p.read_text(encoding='utf8').replace('r9130b','r9130c')
  if p.name=='Deep_eos_compaction_r9130b.py':text=text.replace("sum(x['correct']for x in score[compact])","sum(x['flexible_correct']for x in score[compact])")
  (c/p.name.replace('r9130b','r9130c')).write_bytes(text.encode('utf8'))
p=c/'Deep-eos-compaction-manifest-r9130c.json';m=json.loads(p.read_bytes());m['incoming_source_SHA256']={q.name:hashlib.sha256(q.read_bytes()).hexdigest()for q in c.iterdir()if q.suffix in ['.py','.sh']};m['CPU_fixture_predecessor_error']='b schedulerfixture caught scorer field correct vs flexible_correct; noGPUlaunched';p.write_bytes((json.dumps(m,indent=2)+'\n').encode())
for p in c.iterdir():
 assert b'\r'not in p.read_bytes()
 if p.suffix=='.py':ast.parse(p.read_bytes(),str(p))
fixture=r/'verify_Deep_eos_CPU_contract_r9130c.py';fixture.write_bytes((r/'verify_Deep_eos_CPU_contract_r9130b.py').read_text(encoding='utf-8-sig').replace('r9130b','r9130c').encode());subprocess.run([sys.executable,'-B','-X','utf8',str(fixture)],check=True)
files={p.name:{'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'b64':base64.b64encode(p.read_bytes()).decode()}for p in c.iterdir()if p.is_file()}
code="import pathlib,json,hashlib,base64,subprocess\nr=pathlib.Path.home()/'hygon-inference-refactor-20261014';files="+repr(files)+"\nfor name,e in files.items():\n p=r/'incoming'/name;b=base64.b64decode(e['b64']);assert hashlib.sha256(b).hexdigest()==e['SHA256'];assert b'\\r'not in b\n if p.exists():assert p.read_bytes()==b,(name,'nooverwrite')\n else:p.write_bytes(b)\nfor name in files:\n if name.endswith('.sh'):subprocess.run(['bash','-n',str(r/'incoming'/name)],check=True)\np=r/'incoming/launch_Deep_eos_compaction_CPU_r9130c.py';x=subprocess.run(['python3',str(p)],capture_output=True,text=True,timeout=70);assert x.returncode==0,(x.stdout,x.stderr);print(x.stdout)\n"
(r/'stage_Deep_eos_CPU_r9130c.py').write_bytes(code.encode());(r/'read_Deep_eos_CPU_r9130c.py').write_bytes((r/'read_Deep_eos_CPU_r9130b.py').read_text().replace('r9130b','r9130c').encode());print(json.dumps({'fixture_PASS':True,'files_LF':True,'noGPU':True}))
