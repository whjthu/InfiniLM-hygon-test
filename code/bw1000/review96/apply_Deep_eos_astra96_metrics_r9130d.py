from pathlib import Path
import json,hashlib,ast,base64,subprocess,sys
r=Path.cwd();c=r/'Deep-eos-compaction-private-r9130c';d=r/'Deep-eos-compaction-private-r9130d';d.mkdir(exist_ok=False)
for p in c.iterdir():
 if p.is_file():
  text=p.read_text(encoding='utf8').replace('r9130c','r9130d')
  if p.name=='Deep_eos_compaction_r9130c.py':
   old='select(1);check(lib.modern_native_engine_reset(ctx));began=time.perf_counter();B=len(items)';new='select(1);began=time.perf_counter();check(lib.modern_native_engine_reset(ctx));B=len(items)';assert text.count(old)==1;text=text.replace(old,new)
   old='native_prefill_input_tok_s=sum(map(len,prompts))/pre if pre else None';new="prefill_completed_input_tokens=sum(sum(len(prompts[i])for i in w['rows'])for w in windows if w['prefill']),native_prefill_input_tok_s=sum(sum(len(prompts[i])for i in w['rows'])for w in windows if w['prefill'])/pre if pre else None";assert text.count(old)==1;text=text.replace(old,new)
  (d/p.name.replace('r9130c','r9130d')).write_bytes(text.encode())
p=d/'Deep-eos-compaction-manifest-r9130d.json';m=json.loads(p.read_bytes());m['incoming_source_SHA256']={q.name:hashlib.sha256(q.read_bytes()).hexdigest()for q in d.iterdir()if q.suffix in ['.py','.sh']};m['astra96_measurement_repairs']=['Cohortbeganbeforereset','Partialprefilldenominatoronlyactualexecutedprompttokens'];p.write_bytes((json.dumps(m,indent=2)+'\n').encode())
for p in d.iterdir():
 assert b'\r'not in p.read_bytes()
 if p.suffix=='.py':ast.parse(p.read_bytes(),str(p))
fixture=r/'verify_Deep_eos_CPU_contract_r9130d.py';fixture.write_bytes((r/'verify_Deep_eos_CPU_contract_r9130c.py').read_text(encoding='utf-8-sig').replace('r9130c','r9130d').encode());subprocess.run([sys.executable,'-B','-X','utf8',str(fixture)],check=True)
files={p.name:{'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'b64':base64.b64encode(p.read_bytes()).decode()}for p in d.iterdir()if p.is_file()}
code="import pathlib,json,hashlib,base64,subprocess\nr=pathlib.Path.home()/'hygon-inference-refactor-20261014';files="+repr(files)+"\nfor name,e in files.items():\n p=r/'incoming'/name;b=base64.b64decode(e['b64']);assert hashlib.sha256(b).hexdigest()==e['SHA256'];assert b'\\r'not in b\n if p.exists():assert p.read_bytes()==b,(name,'nooverwrite')\n else:p.write_bytes(b)\nfor name in files:\n if name.endswith('.sh'):subprocess.run(['bash','-n',str(r/'incoming'/name)],check=True)\np=r/'incoming/launch_Deep_eos_compaction_CPU_r9130d.py';x=subprocess.run(['python3',str(p)],capture_output=True,text=True,timeout=70);assert x.returncode==0,(x.stdout,x.stderr);print(x.stdout)\n"
(r/'stage_Deep_eos_CPU_r9130d.py').write_bytes(code.encode());(r/'read_Deep_eos_CPU_r9130d.py').write_bytes((r/'read_Deep_eos_CPU_r9130c.py').read_text().replace('r9130c','r9130d').encode());print('LF_AST_AND_192ROW_CONTRACT_PASS')
