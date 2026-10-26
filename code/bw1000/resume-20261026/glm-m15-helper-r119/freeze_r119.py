import pathlib,json,hashlib,datetime
o=pathlib.Path(__file__).resolve().parent;r=pathlib.Path(r'C:\Users\WangHaojie\Documents\ChatGPT\海光集群')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
idx=json.loads((o/'source-hash-index-r119.json').read_bytes());assert all(sha(pathlib.Path(x['path']))==x['SHA256'] for x in idx['inputs'])
plan=json.loads((o/'candidate-plan-r119.json').read_bytes());assert plan['execution']['SSH']==plan['execution']['GPU']==0
s=(o/'glm_fp8_runtime_m15_r119.cpp').read_text();c=(o/'glm_fp8_cache_m15_r119.cpp').read_text()
assert 'bool m15_enabled{false}' in s and 'if(m!=8&&m!=16&&m!=32&&m!=64&&!m15_admitted)return false;' in s
assert 'if(m15_admitted)' in s and 'glm_fp8_cache_run_r807(c.handle' in s and 'glm_fp8_cache_run_r119(c.m15_handle' in s
assert '(m!=15)' in c and 'glm_fp8_run_r776(i->second.plan,mode,out,x,xs,w,ws)' in c
assert 'byte_limit=192ull*1024*1024,entry_limit=10' in c
assert 16*(2*15*128+2*6144*128+4*15*6144)==31125504
extra=[r/'glm-transient-layer-r5680-src/glm_moe_dsa_attention_output_r809.cpp',r/'GLM-vector-Model-handoff-r8544.json',r/'GLM-irregular-vector-private-r9270b/terminal-canonical-r9271.json',o.parent/'glm-residual-r118/G115A-report-r118.json']
seen={x['path'] for x in idx['inputs']}
for p in extra:
 if str(p) not in seen:idx['inputs'].append(dict(path=str(p),SHA256=sha(p),bytes=p.stat().st_size))
idx['outputs']=[dict(path=p.name,SHA256=sha(p),bytes=p.stat().st_size) for p in o.iterdir() if p.is_file() and p.name not in ('source-hash-index-r119.json','completion-receipt-r119.json')]
(o/'source-hash-index-r119.json').write_text(json.dumps(idx,indent=2),encoding='utf8')
receipt=dict(schema=1,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),decision=plan['decision'],plan_SHA256=sha(o/'candidate-plan-r119.json'),source_index_SHA256=sha(o/'source-hash-index-r119.json'),runtime_source_SHA256=sha(o/'glm_fp8_runtime_m15_r119.cpp'),private_cache_source_SHA256=sha(o/'glm_fp8_cache_m15_r119.cpp'),patch_SHA256=sha(o/'source-delta-r119.patch'),input_hashes_match=True,static_source_invariants_pass=True,CPUcompile=0,GPU=0,SSH=0,not_runtime_ready=True)
(o/'completion-receipt-r119.json').write_text(json.dumps(receipt,indent=2),encoding='utf8');print(json.dumps(receipt))
