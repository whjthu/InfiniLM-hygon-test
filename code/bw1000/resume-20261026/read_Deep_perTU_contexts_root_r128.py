"""Bounded metadata-only SSH stdin payload. No compile, subprocess, logs or env."""
import pathlib,json,hashlib,shlex
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
def path(p):return pathlib.Path(str(p).replace('/refactor',str(R),1))
def sha(p):return hashlib.sha256(path(p).read_bytes()).hexdigest()
def read(p):return json.loads(path(p).read_bytes())
out={'scope':'READONLY exact current nine HOST compile contexts; no compile/dlopen/weights/env/logs','files':{},'units':[]}
hp='/refactor/results/deep-c4-projection-model-r5920-build.json';h=read(hp)
recipe='/refactor/candidates/deep-c4-projection-model-r5920/compile.sh';raw=path(recipe).read_text()
assert sha(recipe)==h['compile_sha256'];assert str(h['ABI'])in ['0','1']
out['files'][hp]={'SHA256':sha(hp),'ABI':h['ABI'],'completed':h['completed']};out['files'][recipe]={'SHA256':sha(recipe),'text':raw}
env={'BASE':'/refactor/candidates/deep-c4-projection-model-r5920','COMMON':'/refactor/candidates/rebased-5f91ad7-r3668/Common/Deep/InfiniLM','LM':'/refactor/candidates/rebased-5f91ad7-r3668/Model/Deep/InfiniLM','SDK':'/refactor/install/hygon-deep-5f91ad7-r3672','HELPER':'/refactor/candidates/models-5f91ad7-r3675/Deep/helpers','ABI':str(h['ABI'])}
a=shlex.split(raw.replace('\\\n',' '))
for i,x in enumerate(a):
 for k,v in env.items():x=x.replace('${'+k+'}',v).replace('$'+k,v)
 assert '$'not in x,('unexpanded recipe token',x);a[i]=x
oi=a.index('-o');flags=[x for i,x in enumerate(a)if i not in[oi,oi+1]and not x.endswith(('.cpp','.o'))and x!='-shared']
for u in h['units']:
 if pathlib.Path(u['object']).name not in ['unit-1.o','unit-10.o','unit-20.o','unit-21.o']:continue
 assert sha(u['object'])==u['object_sha256'];assert sha(u['source'])==u['source_sha256'];assert u['source']in a
 out['units'].append({'object':u['object'],'object_SHA256':u['object_sha256'],'source':u['source'],'source_SHA256':u['source_sha256'],'compile_argv':flags+['-MD','-MF',u['depfile'],'-c',u['source'],'-o',u['object']],'provenance':'Exact flags algorithm of executed5920 builder, pinned compile.sh + actual ABI/build/unit hashes. Reconstructed recipe, not an observed process argv.','build_receipt_SHA256':sha(hp),'recipe_SHA256':sha(recipe),'dependencies':u['dependencies']})
for rp,key in [('/refactor/results/deep-owner-device-CPU-r8918.json','new_compiled_units'),('/refactor/results/deep-attention-inner-CPU-r9193a.json','new_compiled_units'),('/refactor/candidates/deep-c4-smallt-r9252/CPU-build.json','compiled')]:
 q=read(rp);out['files'][rp]={'SHA256':sha(rp),'completed':q['completed']}
 assert q['completed']
 for u in q[key]:
  if pathlib.Path(u['source']).name not in ['deepseek_v4_native_routed_decoder_r20.cpp','deepseek_v4_swa_attention.cpp','deepseek_v4_hca_attention_batch_r784.cpp','deepseek_v4_csa_attention_batch_r784.cpp','deep_c4_smallt_r9252.cpp']:continue
  assert sha(u['object'])==u['object_SHA256']and sha(u['source'])==u['source_SHA256'];assert 'compile_argv'in u
  out['units'].append(dict(u,build_receipt_SHA256=sha(rp),provenance='Actual successful CPU receipt per-object compile_argv'))
assert len(out['units'])==9
cp='/refactor/candidates/deep-c4-smallt-model-r9254/CPU-build.json';c=read(cp);assert sha(cp)=='43d4478102462e26bbfb0784942ec2f8f05228f53fb42eae0b23c051d1ed6d4c';assert c['completed']
objects=c['link']['full']['objects'];assert all(objects[u['object']]==u['object_SHA256']for u in out['units']);out['current_link']=c['link']['full'];out['current_CPU_receipt_SHA256']=sha(cp)
print(json.dumps(out))
