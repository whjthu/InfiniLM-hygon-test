"""Resolve remaining called methods and actual Runtime binary sync/copy callees."""
from pathlib import Path
import json,hashlib,subprocess,re,datetime
R=Path.home()/'hygon-inference-refactor-20261014';host=lambda p:Path(str(p).replace('/refactor',str(R),1));sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
native=R/'candidates/GLM-PP4-registry-repair-r184b/libGLM_PP4_registry_r184b.so';assert sha(native)=='038d4ff2e7aee2e6db80f691dd885222f3e3160e25ae9d0e269296daf9596c00'
a=json.loads((native.parent/'CPU-build.json').read_bytes());objects=[host(p)for cmd in a['commands']if isinstance(cmd,list)and'-shared'in cmd for p in cmd if p.endswith('.o')]
q=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),CPU_only=True,GPU=False,dlopen=False,weights_opened=False,Native_SHA256=sha(native),extra_symbols=[],sources={},Runtime_binary_proof=[])
text=subprocess.check_output(['nm','-A','-C','-g','--defined-only',*[str(p)for p in objects]],text=True,timeout=45);selected=set()
for line in text.splitlines():
 if re.search(r' [TW] infinilm::models::glm_moe_dsa::[^ ]*(?:Prepare|Output|Projection|Indexer|Selection)[^ ]*::(?:forward|expand_cached)',line):
  o=Path(line.split(':',1)[0]);selected.add(o);q['extra_symbols'].append(dict(object=str(o),SHA256=sha(o),symbol=line.split(' ',2)[-1]))
for o in selected:
 md=o.with_suffix('.d')
 if not md.exists():continue
 for name in set(re.findall(r'[^\s\\]+\.cpp',md.read_text())):
  s=host(name)
  if s.exists():q['sources'][str(s)]=dict(SHA256=sha(s),MD_path=str(md),MD_SHA256=sha(md),text=s.read_text())
nm=subprocess.check_output(['nm','-D','--defined-only',str(native)],text=True,timeout=25)
q['existing_component_C_exports']=[line for line in nm.splitlines()if re.search(r' [TW] (?:glm|modern_glm)[^ ]*(?:layer|prefix|window)[^ ]*$',line)and any(s in line for s in ['create','capacity','resize','forward'])]
runtime=R/'candidates/rebased-5f91ad7-r3668/Common/GLM/InfiniLM/python/infinilm/_lib/libinfinilm_runtime.so';q['Runtime_DSO_SHA256']=sha(runtime)
symbols=subprocess.check_output(['nm','-D','-S','--defined-only',str(runtime)],text=True,timeout=25)
for line in symbols.splitlines():
 if any(s in line for s in ['Runtime10syncStreamEv','Runtime9memcpyD2H','Runtime9memcpyH2D']):
  fields=line.split();symbol=fields[-1];start=int(fields[0],16);end=start+int(fields[1],16);raw=subprocess.check_output(['objdump','-d','-C','--start-address='+hex(start),'--stop-address='+hex(end),str(runtime)],text=True,timeout=25);q['Runtime_binary_proof'].append(dict(symbol=symbol,address_start=hex(start),address_end=hex(end),disassembly_SHA256=hashlib.sha256(raw.encode()).hexdigest(),disassembly=raw,external_calls=[x.strip()for x in raw.splitlines()if'call'in x or'jmp'in x]))
print(json.dumps(q))
