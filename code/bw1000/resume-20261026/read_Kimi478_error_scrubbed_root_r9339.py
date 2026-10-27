import pathlib,json,re,subprocess
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');run=json.loads((r/'incoming/Kimi-ctx2k-Model-r127/launcher/run-context-public.json').read_bytes())['run']
def scrub(x):
 if isinstance(x,str):return re.sub(r'(?<![0-9a-fA-F])[0-9a-fA-F]{32,}(?![0-9a-fA-F])','<hex-redacted>',x)
 if isinstance(x,list):return [scrub(v)for v in x]
 if isinstance(x,dict):return {k:scrub(v)for k,v in x.items()if not any(t in k.lower()for t in ['nonce','argv','args','token_ids'])}
 return x
out={'scope':'Actualterminal478failure, bounded JSON diagnostics with all longhex scrubbed, noenv/rawargv/activation'}
for n in range(4):
 p=r/'results'/('Kimi-ctx2k-r127-'+run+'-node'+str(n)+'.json')
 if not p.exists():continue
 d=json.loads(p.read_bytes());out[str(n)]=scrub({k:d.get(k)for k in ['phase','error','traceback','shutdown_error','screen_gate','resident_stop','pending_budget','command_rejections','initial18_complete','completed']})
h=r/'results'/('Kimi-ctx2k-phase-A-model-r127-'+run+'.json');w=json.loads(h.read_bytes());out['health']=scrub({k:w.get(k)for k in ['completed','passed','audit','post_KFD_passed','explicit_target_failure']});out['health']['targets']=[scrub({k:g.get(k)for k in ['actual_handle','terminal','native_error','signal','cleanup_error']})for g in w.get('groups',[])]
print(json.dumps(out))
