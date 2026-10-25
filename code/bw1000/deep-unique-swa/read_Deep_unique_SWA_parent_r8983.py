import pathlib,json,hashlib,subprocess
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();qpath=r/'results/deep-reference-scratch-CPU-r8954.json';raw=qpath.read_bytes();q=json.loads(raw);assert h(qpath)=='092e50eafcd36058df8d28961b72c140419e08a8137d7b9ba77148ecf19687fd'
fix=lambda s:r/str(s).removeprefix('/refactor/')
u=q['new_compiled_units'][0];dep={n:{'recorded':v,'actual':h(fix(n))}for n,v in u['deps'].items()if 'deep_selection_r356' in n};assert dep and all(v['recorded']==v['actual']for v in dep.values())
found=[]
for n,v in q['link']['full']['objects'].items():
 p=fix(n);assert h(p)==v
 nm=subprocess.check_output(['nm','--defined-only',str(p)],text=True,timeout=5)
 if 'deep_packed_gather_r374' in nm:found.append({'object':n,'SHA256':v,'symbols':[x for x in nm.splitlines()if 'deep_packed_gather_r374' in x]})
assert len(found)==1
print(json.dumps({'CPU_only':True,'GPU':False,'receipt_SHA256':h(qpath),'source':u['source'],'source_SHA256':u['source_SHA256'],'object':u['object'],'object_SHA256':u['object_SHA256'],'actual_source_SHA256':h(fix(u['source'])),'selection_header':dep,'gather_provider_object':found,'objects_count':len(q['link']['full']['objects']),'compile_argv':u['compile_argv'],'scope':'Actual219receipt+MD deps and actual all-linked object SHA/nm only; noGPU/dlopen; r374source historical mapping still UNKNOWN unless retained source has same build receipt.'}))
