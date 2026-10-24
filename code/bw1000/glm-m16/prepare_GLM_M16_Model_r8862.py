import pathlib,json,hashlib,ast
r=pathlib.Path(__file__).resolve().parent;old=r/'GLM-exact-budget-Model-private-r8855';d=r/'GLM-M16-Model-private-r8862';d.mkdir(exist_ok=True);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
parent=json.loads((old/'GLM-CROSSREQUEST-Model-source-manifest-r8855.json').read_bytes());cpu=json.loads((r/'GLM-M16-CPU-observed-r8861-b.json').read_bytes())['receipt'];assert cpu['completed']and cpu['compiled_TUs']==7
oldlib='candidates/glm-exact-budget-r8854/libglm_exact_budget_r8854.so';newlib=cpu['library'].removeprefix('/refactor/')
names=set(parent['source_SHA256'])|{'GLM-transient-Model-observer-manifest-r8855.json','dispatch_GLM_resident_exact_Model_r8855.py'}
for name in names:
 raw=(old/name).read_bytes()
 if 'r8855'in name:raw=raw.replace(b'r8855',b'r8862').replace(oldlib.encode(),newlib.encode()).replace(parent['model_SHA256'].encode(),cpu['library_SHA256'].encode())
 (d/name.replace('r8855','r8862')).write_bytes(raw)
def edit(name,fn):
 p=d/name;s=p.read_text();s=fn(s);p.write_bytes(s.encode())
def probe(s):
 s=s.replace('prefill_micro in(4,8)and micro==8','prefill_micro==8 and micro in(8,16)')
 s=s.replace("len(e['rows'])==8)","len(e['rows'])in(8,16))")
 s=s.replace("'5707 ROWS16: resident exact5555 M8/original601 EOS tails; unselected host actual5096 when uniform; transientOFF'","'ROWS32: resident exact5555 M8/M16, original601 other EOS tails; currentvector routes32/64/128; transientOFF'")
 s=s.replace("labels=['perf-I512-B16-N-r0','perf-I512-B32-N-r0']","labels=['perf-I512-B32-M8-N-r0','perf-I512-B32-M16-N-r0']")
 s=s.replace("prompts=i512_inputs[:16] if '-B16-'in label else i512_inputs\n            generate(label,prompts,8,True", "prompts=i512_inputs;micro=16 if '-M16-'in label else 8\n            generate(label,prompts,micro,True")
 s=s.replace('ROWS32 TP8PP2/B16thenB32M8','ROWS32 TP8PP2/B32M8thenB32M16')
 s=s.replace('Same-load B16/B32 schedulingonly ordinaryfull128, O8fixed/VON/TOFF/fixedactualadmittedmask32rowbanks/crossOFF/sharedON; optionalNO byactualbudget; no windowprofile.','Same-load B32M8->B32M16 scheduling-only full128, prefillchunk8/VON/TOFF/PVOFF/fixedactualadmittedmask32rowbanks/crossOFF/sharedON; no optionalarms/windowprofile.')
 return s
edit('glm_prefill_micro_probe_r8862.py',probe)
edit('glm_crossrequest_resident_base_r8862.py',lambda s:s.replace("len(e['rows'])==8 for e in events","len(e['rows'])in(8,16) for e in events").replace("len(e['rows'])<8)","len(e['rows'])not in(8,16))"))
def audit(s):
 s=s.replace("row['batch']in(16,32) and row['micro']==8","row['batch']==32 and row['micro']in(8,16)")
 s=s.replace('flat in(4,8)','flat in(4,8,16)').replace("e['flat_tokens']in(4,8)","e['flat_tokens']in(4,8,16)")
 s=s.replace("e['round']>0 and len(e['rows'])==8", "e['round']>0 and len(e['rows'])in(8,16)")
 s=s.replace("e['round']>0 and len(e['rows'])<8", "e['round']>0 and len(e['rows'])not in(8,16)")
 s=s.replace('Same-load exactresident constructor32/fixedmask B16M8 thenB32M8','Same-load exactresident constructor32/fixedmask B32M8 thenB32M16')
 return s
edit('audit_GLM_prefill_Model_r8862.py',audit)
edit('GLM_transient_preflight_r8862.py',lambda s:s.replace("assert len(proof['reused_objects_SHA256'])==27","assert len(proof['reused_objects_SHA256'])==21 and proof['compiled_TUs']==7").replace("(8,2,640,32,32,8)","(8,2,640,32,32,[8,16])"))
def metadata(s):
 s=s.replace("assert receipt['resident_exact_budget']['resident_bank_bytes']", "budget_receipt=json.loads(mapped(manifest['budget_cpu_receipt_path']).read_bytes())\n assert budget_receipt['completed'] and budget_receipt['library_SHA256']==manifest['budget_parent_DSO_SHA256']\n assert budget_receipt['resident_exact_budget']['resident_bank_bytes']")
 s=s.replace("receipt['resident_exact_budget']['reserve_bytes']", "budget_receipt['resident_exact_budget']['reserve_bytes']").replace("receipt['resident_exact_budget']['allowance_bytes']", "budget_receipt['resident_exact_budget']['allowance_bytes']")
 return s
edit('glm_model_library_metadata_r8862.py',metadata)
for name in ['launch_GLM_prefill_Model_r8862.py','modern_glm_prefill_micro_r8862.py','glm_prefill_micro_Model_r8862_inside.sh']:
 edit(name,lambda s:s.replace('B16thenB32M8','B32M8thenM16').replace('B16M8->B32M8','B32M8->B32M16').replace('B16M8 thenB32M8','B32M8 thenB32M16').replace('B16first16subset->B32distinct','B32distinctM8->M16').replace('exact32-8853-','M16-8862-'))
m=dict(parent);m['model_SHA256']=cpu['library_SHA256'];m['DSO_SHA256']=dict(parent['DSO_SHA256']);m['DSO_SHA256'].pop(oldlib);m['DSO_SHA256'][newlib]=cpu['library_SHA256'];m['cpu_receipt_path']='/refactor/candidates/glm-M16-r8861/CPU-build.json';m['budget_cpu_receipt_path']='/refactor/candidates/glm-exact-budget-r8854/CPU-build.json';m['budget_parent_DSO_SHA256']=parent['model_SHA256'];m['CPU_native_receipt']={'path':'GLM-M16-CPU-observed-r8861-b.json','SHA256':sha(r/'GLM-M16-CPU-observed-r8861-b.json')};m['CPU_raw_receipt_SHA256']=m['CPU_native_receipt']['SHA256'];m['compiled_units']=cpu['compiled'];m['CPU_contracts']={'actualCPU':'910407.210;7hostTUs/21reusedobjects;50.655s;actualSlots+dispatchCPUcontractPASS;noGPU/dlopen','attention_helper':cpu['bounded_attention']}
m['scope']='One load TP8PP2/constructor32/cap640/active32: same32distinct477 I512O128 B32M8->B32M16;prefillchunk8; fixed actual admitted exact160MiB mask/6208MiBadmission/512MiBruntimefloor/noeviction/drain;45min2600inner;n12+n13 each8DCU/c64/240000M; no newkernel/SDK/observer.'
m['root_GO']='PENDING_READY_PACKAGE_ROOT_MODEL_GO';m['performance_micro']=[8,16];m.pop('decode_micro_fixed',None);m['decode_micro_candidates']=[8,16];m['perf_labels']=['perf-I512-B32-M8-N-r0','perf-I512-B32-M16-N-r0'];m['math']='Existing vector8528 kernel/SDK; host selected128routes; exact5555 backend1 original per-row QK/PV/scale/softmax/order; sameM8 branch; larger shape finaltokens qualification stillrequired.';m['scope_hit']='Botharms B32 same32distinct512inputs from477, full128IDs/EOS/QA; branchcounts show shared8/16 + vector32/64/128, preserve EOS tails; no rebaseline.';m['input_provenance']=dict(parent['input_provenance']);m['input_provenance'].pop('B16',None);m['input_provenance']['both_arms']='same32distinct/full512/no padding';m['diagnostic_budget']='45minhard2600inner proposed;561historical1982s=one load+mixedB16/B32arms. Here twoB32arms, expect~150additionalM8-equivalent seconds; M16 time/workspace UNKNOWN; no extraarm/retry/extension; reserve remaining QA/shutdown.';m['mechanical_revision']={'prior561':'qualified B16/B32M8 exactbanks; retained unchanged','current':'M16 hostguards7actualMDconsumers; originalCUDA/SDK21obj reused; no math/nativekernel changes'}
sidep=d/'GLM-transient-Model-observer-manifest-r8862.json';side=json.loads(sidep.read_bytes());side['source_SHA256']={n.replace('r8855','r8862'):sha(d/n.replace('r8855','r8862'))for n in json.loads((old/'GLM-transient-Model-observer-manifest-r8855.json').read_bytes())['source_SHA256']};sidep.write_bytes((json.dumps(side,indent=2)+'\n').encode());m['observer_manifest_SHA256']=sha(sidep)
m['source_SHA256']={n.replace('r8855','r8862'):sha(d/n.replace('r8855','r8862'))for n in parent['source_SHA256']};mp=d/'GLM-CROSSREQUEST-Model-source-manifest-r8862.json';mp.write_bytes((json.dumps(m,indent=2)+'\n').encode());mh=sha(mp)
ep=d/'dispatch_GLM_resident_exact_Model_r8862.py';s=ep.read_text();assert s.count('0e4810b6aff66102f76de677ed71d91caa5c4d988743233986a8c72920f4063a')==1;s=s.replace('0e4810b6aff66102f76de677ed71d91caa5c4d988743233986a8c72920f4063a',mh);ep.write_bytes(s.encode())
for p in d.glob('*.py'):ast.parse(p.read_bytes(),filename=str(p))
print(json.dumps({'CPU_only':True,'GPU':False,'directory':d.name,'manifest_SHA256':mh,'entry_SHA256':sha(ep),'DSO_SHA256':cpu['library_SHA256']}))
