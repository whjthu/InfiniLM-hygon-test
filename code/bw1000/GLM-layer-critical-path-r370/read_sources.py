"""Read exact current038d object/MD-selected call sites; no Native/weights/GPU."""
from pathlib import Path
import hashlib,json,subprocess,re,datetime
R=Path.home()/'hygon-inference-refactor-20261014';host=lambda p:Path(str(p).replace('/refactor',str(R),1));sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=R/'candidates/GLM-PP4-registry-repair-r184b/CPU-build.json';assert sha(p)=='6aa62f788aad6080027d665f30d7c9824c07cb1866de905397fef15c56b26529';a=json.loads(p.read_bytes());assert a['completed']and sha(host(a['library']))==a['library_SHA256']=='038d4ff2e7aee2e6db80f691dd885222f3e3160e25ae9d0e269296daf9596c00'
link=next(cmd for cmd in a['commands']if isinstance(cmd,list)and'-shared'in cmd);objects=[host(p)for p in link if p.endswith('.o')]
text=subprocess.check_output(['nm','-A','-C','-g','--defined-only',*[str(p)for p in objects]],text=True,timeout=45)
q=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),CPU_only=True,GPU=False,Native_dlopen=False,weights_opened=False,Native_SHA256=a['library_SHA256'],Native_CPU_receipt_SHA256=sha(p),symbols=[],sources={},MDs={})
relevant=re.compile(r' [TW] (?:glm_layer_(?:forward|mark)_r5670|infinilm::models::glm_moe_dsa::(?:DecoderProjectedR22|NativeAttentionProjectedR22|FeedForwardR19)::forward\()')
selected=set()
for line in text.splitlines():
 if relevant.search(line):
  obj=Path(line.split(':',1)[0]);q['symbols'].append(dict(object=str(obj),SHA256=sha(obj),symbol=line.split(' ',2)[-1]));selected.add(obj)
for obj in selected:
 md=obj.with_suffix('.d');assert md.exists(),str(md);text=md.read_text();q['MDs'][str(md)]=dict(SHA256=sha(md),object=str(obj),object_SHA256=sha(obj));files=set(host(n)for n in re.findall(r'[^\s\\]+\.(?:cpp|cu|hpp|h)',text))
 for s in files:
  if not s.exists():continue
  if s.suffix in ['.cpp','.cu']or any(str(s).endswith(x)for x in ['execution/submit.hpp','runtime/runtime.hpp','runtime/tensor.hpp','runtime/memory.hpp','glm_decode_leases_r1354.hpp','glm_layer_trace_r5670.hpp','glm_sparse_observer_r131.hpp']):
   q['sources'][str(s)]=dict(SHA256=sha(s),selected_by_MD=str(md),text=s.read_text())
q['implementation_source_candidates_not_MD_to_DSO_proven']={}
base=R/'candidates/rebased-5f91ad7-r3668/Model/GLM/InfiniLM/csrc'
for s in [base/'runtime/runtime.cpp',base/'runtime/tensor.cpp']:
 if s.exists():q['implementation_source_candidates_not_MD_to_DSO_proven'][str(s)]=dict(SHA256=sha(s),text=s.read_text(),DSO_compile_binding='UNKNOWN in this read')
runtime=R/'candidates/rebased-5f91ad7-r3668/Common/GLM/InfiniLM/python/infinilm/_lib/libinfinilm_runtime.so';q['Runtime_DSO']=dict(path=str(runtime),SHA256=sha(runtime))
q['Runtime_symbols']=[line for line in subprocess.check_output(['nm','-D','-C','--defined-only',str(runtime)],text=True,timeout=25).splitlines()if re.search(r' [TW] infinilm::runtime::Runtime::(?:submit|syncStream|memcpyD2H|setDeviceMemory|trimMemory|wait)',line)]
print(json.dumps(q))
