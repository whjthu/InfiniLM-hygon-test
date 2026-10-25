import pathlib,json,hashlib,subprocess
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';b=json.loads((r/'results/deep-latent-prefill-CPU-r8938.json').read_bytes());out={'CPU_only':True,'sources':[],'compute_symbol_object':[]}
def host(p):return r/pathlib.Path(p).relative_to('/refactor')
for p,h in b['link']['full']['objects'].items():
 if any(s in p.lower()for s in ['latent','681']):
  nm=subprocess.check_output(['nm','-g','--defined-only',str(host(p))],text=True)
  if 'deep_latent_compute_r681'in nm:out['compute_symbol_object'].append({'object':p,'SHA256':h})
old=json.loads((r/'results/deep-c4-projection-model-r5920-build.json').read_bytes())
for u in old['units']:
 if any(s in pathlib.Path(u['source']).name for s in ['latent','attention_prepare']):
  p=host(u['source']);out['sources'].append({'source':u['source'],'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'text':p.read_text()})
print(json.dumps(out))
