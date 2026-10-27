import pathlib,json,hashlib,shlex
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
host=lambda s:pathlib.Path(s.replace('/refactor',str(r),1))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
out={'scope':'named current1760 HOST object/source/MD metadata only','records':[],'files':{}}
def add(name,text=False):
 p=host(name);b=p.read_bytes();assert len(b)<3000000
 v={'SHA256':sha(p),'bytes':len(b)}
 if text:assert len(b)<300000;v['text']=b.decode('utf8')
 out['files'][name]=v
qpath='/refactor/candidates/glm-transient-layer-r5681/CPU-build.json';q=json.loads(host(qpath).read_bytes());add(qpath)
for rec in q['compiled']:
 if pathlib.PurePosixPath(rec['object']).name not in ['private-7.o','private-8.o','private-9.o']:continue
 cmd=next(a for a in q['commands']if '-c'in a and rec['source']in a)
 d=cmd[cmd.index('-MF')+1];add(rec['object']);add(rec['source'],True);add(d,True)
 out['records'].append({'record':rec,'command':cmd,'MD':d})
 for n in shlex.split(host(d).read_text().replace('\\\n',' ').split(':',1)[1]):
  if pathlib.PurePosixPath(n).name in ['glm_moe_dsa_projected_text_model_r22.hpp','glm_moe_dsa_projected_decoder_r22.hpp','glm_layer_trace_r5670.hpp','glm_model_window_r5797.hpp','glm_moe_dsa_projected_attention_r22.hpp']:add(n,True)
for n in ['/refactor/candidates/GLM-M15-oproj-Model-r125/CPU-build.json','/refactor/candidates/GLM-M15-oproj-Model-r125/libGLM_M15_oproj_Model_r125.so']:add(n,n.endswith('.json'))
assert any(x['record']['object'].endswith('/private-8.o')for x in out['records'])
print(json.dumps(out,separators=(',',':')))
