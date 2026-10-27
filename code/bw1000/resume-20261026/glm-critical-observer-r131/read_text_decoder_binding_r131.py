import pathlib,json,hashlib,shlex
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');host=lambda s:pathlib.Path(s.replace('/refactor',str(r),1));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();out={'scope':'corrective named current1760 text and decoder HOST provenance','records':[],'files':{}}
def add(n,text=False):
 p=host(n);b=p.read_bytes();assert len(b)<3000000;v={'SHA256':sha(p),'bytes':len(b)}
 if text:assert len(b)<300000;v['text']=b.decode('utf8')
 out['files'][n]=v
cp=json.loads(host('/refactor/candidates/GLM-M15-oproj-Model-r125/CPU-build.json').read_bytes());link=next(a for a in cp['commands']if '-shared'in a and '-o'in a)
for receipt in ['/refactor/candidates/glm-transient-layer-r5681/CPU-build.json','/refactor/candidates/glm-window-model-r5797/CPU-build.json','/refactor/candidates/glm-M16-r8861/CPU-build.json']:
 q=json.loads(host(receipt).read_bytes());add(receipt);out.setdefault('scalar_records',[]).extend(q.get('compiled',[]))
 for rec in q.get('compiled',[]):
  if rec['object']not in link or pathlib.PurePosixPath(rec['source']).name not in ['glm_moe_dsa_projected_decoder_r22.cpp','glm_moe_dsa_balanced_text_model_r126.cpp']:continue
  cmd=next(a for a in q['commands']if '-c'in a and rec['source']in a);d=cmd[cmd.index('-MF')+1];add(rec['object']);add(rec['source'],True);add(d,True);out['records'].append({'record':rec,'command':cmd,'MD':d})
  for n in shlex.split(host(d).read_text().replace('\\\n',' ').split(':',1)[1]):
   if pathlib.PurePosixPath(n).name in ['glm_moe_dsa_projected_text_model_r22.hpp','glm_moe_dsa_projected_decoder_r22.hpp','glm_layer_trace_r5670.hpp','glm_model_window_r5797.hpp']:add(n,True)
print(json.dumps(out,separators=(',',':')))
