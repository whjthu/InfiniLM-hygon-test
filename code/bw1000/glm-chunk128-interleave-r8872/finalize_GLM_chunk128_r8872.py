from pathlib import Path
import ast,hashlib,json
root=Path(__file__).resolve().parent;r=root/'GLM-chunk128-Model-private-r8872'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=r/'glm_prefill_micro_probe_r8872.py';s=p.read_text(encoding='utf-8')
a="                if stage == PP-1 and emit_generation:\n                    assert torch.isfinite(logits).all(), ('nonfinite', label, step, group)"
b="                if stage == PP-1:assert torch.isfinite(logits).all(), ('nonfinite',label,step,group,prompt_offset,prompt_count)\n                if stage == PP-1 and emit_generation:"
assert s.count(a)==1;s=s.replace(a,b)
s=s.replace("'actual':packet['ids'],'expected':expected}","'actual':packet['ids'],'expected':expected,'generated_prefix':generated,'tail_ready_times':tail_ready if participant==emitter else None}")
p.write_bytes(s.encode())
for q in r.glob('*.py'):compile(q.read_text(encoding='utf-8'),str(q),'exec')
# Execute the actual consumer's error predicate, without imports/device work.
tree=ast.parse(s);f=next(n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=='validate_decode')
gold=json.loads((r/'GLM477-frozen-gold-r8844.json').read_bytes());generated=[[g[0]]for g in gold['generated_ids'][:16]]
class Rejected(RuntimeError):pass
saved=[];ns={'reference_kind':'I512','historical_i512':gold,'generated':generated,'report':{},'label':'CPU-only','save':lambda:saved.append(True),'CandidateRejected':Rejected,'tail_ready':[[]for _ in range(16)],'participant':8,'emitter':8}
exec(compile(ast.fix_missing_locations(ast.Module(body=[f],type_ignores=[])),'<actual validate_decode>','exec'),ns)
right=[g[1]for g in gold['generated_ids'][:8]]
ns['validate_decode']({'ids':right},list(range(8)))
try:ns['validate_decode']({'ids':[right[0]+1]+right[1:]},list(range(8)))
except Rejected:pass
else:raise AssertionError('actual consumer token mismatch failed to stop')
assert saved==[True]and ns['report']['first_decode_failure']['generated_prefix']==generated
ob=r/'GLM-transient-Model-observer-manifest-r8872.json';o=json.loads(ob.read_bytes())
for n in o['source_SHA256']:o['source_SHA256'][n]=sha(r/n)
ob.write_bytes((json.dumps(o,ensure_ascii=False,indent=2)+'\n').encode())
mp=r/'GLM-CROSSREQUEST-Model-source-manifest-r8872.json';m=json.loads(mp.read_bytes())
for n in m['source_SHA256']:m['source_SHA256'][n]=sha(r/n)
m['observer_manifest_SHA256']=sha(ob)
mp.write_bytes((json.dumps(m,ensure_ascii=False,indent=2)+'\n').encode())
ep=r/'dispatch_GLM_resident_exact_Model_r8872.py';e=ep.read_text(encoding='utf-8')
tree=ast.parse(e);assign=next(n for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Attribute)and t.attr=='argv'for t in n.targets));old=assign.value.elts[1].value
assert len(old)==64;e=e.replace(old,sha(mp));ep.write_bytes(e.encode())
rp=root/'GLM-chunk128-review-package-r8872.json';review=json.loads(rp.read_bytes());review.update(manifest_SHA256=sha(mp),entry_SHA256=sha(ep),actual_Python_compile='all private Python modules compiled in memory without imports/GPU',actual_consumer_token_reject={'correct':True,'wrong':True,'failure_prefix_atomically_saved_by_real_callback':True},position_errata='O128 consumes prompt512+127decode inputs: final rowlength639, last position638; capacity640 unchanged.',source_card='GLM70-chunked-prefill-source-decision-r8871.json')
rp.write_bytes((json.dumps(review,ensure_ascii=False,indent=2)+'\n').encode());print(json.dumps(review))
