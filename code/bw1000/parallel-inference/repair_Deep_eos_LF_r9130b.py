from pathlib import Path
import json,hashlib,ast,base64
r=Path.cwd();a=r/'Deep-eos-compaction-private-r9130';b=r/'Deep-eos-compaction-private-r9130b';b.mkdir(exist_ok=False)
for p in a.iterdir():
 if p.is_file():
  text=p.read_text(encoding='utf8').replace('r9130','r9130b');(b/p.name.replace('r9130','r9130b')).write_bytes(text.replace('\r\n','\n').encode('utf8'))
p=b/'Deep-eos-compaction-manifest-r9130b.json';d=json.loads(p.read_bytes());d['incoming_source_SHA256']={q.name:hashlib.sha256(q.read_bytes()).hexdigest()for q in b.iterdir()if q.suffix in ['.py','.sh']};d['supersedes_failed_CPU_variant']={'source':'r9130','handle':'910406.245','failure':'ShellCRLFpipefailbeforePython/noGPU','no_old_source_overwrite':True};p.write_bytes((json.dumps(d,indent=2)+'\n').encode())
for p in b.iterdir():
 assert b'\r'not in p.read_bytes()
 if p.suffix=='.py':ast.parse(p.read_bytes(),str(p))
files={p.name:{'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'b64':base64.b64encode(p.read_bytes()).decode()}for p in b.iterdir()if p.is_file()}
code="import pathlib,json,hashlib,base64,subprocess\nr=pathlib.Path.home()/'hygon-inference-refactor-20261014';files="+repr(files)+"\nfor name,e in files.items():\n p=r/'incoming'/name;b=base64.b64decode(e['b64']);assert hashlib.sha256(b).hexdigest()==e['SHA256'];assert b'\\r'not in b\n if p.exists():assert p.read_bytes()==b,(name,'nooverwrite')\n else:p.write_bytes(b)\nfor name in files:\n if name.endswith('.sh'):subprocess.run(['bash','-n',str(r/'incoming'/name)],check=True)\np=r/'incoming/launch_Deep_eos_compaction_CPU_r9130b.py';x=subprocess.run(['python3',str(p)],capture_output=True,text=True,timeout=70);assert x.returncode==0,(x.stdout,x.stderr);print(x.stdout)\n"
(r/'stage_Deep_eos_CPU_r9130b.py').write_bytes(code.encode('utf8'));read=(r/'read_Deep_eos_CPU_r9130.py').read_text(encoding='utf8-sig').replace('r9130','r9130b');(r/'read_Deep_eos_CPU_r9130b.py').write_bytes(read.encode('utf8'));print(json.dumps({'LF_allfiles':True,'AST_allPY':True,'nooverwrite_r9130':True}))
