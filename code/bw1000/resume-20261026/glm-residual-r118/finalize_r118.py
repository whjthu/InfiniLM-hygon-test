import json,pathlib,hashlib,datetime
f=pathlib.Path(__file__).resolve().parent
r=pathlib.Path(r'C:\Users\WangHaojie\Documents\ChatGPT\海光集群')
read=lambda p:json.loads(p.read_bytes())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report=read(f/'G115A-report-r118.json')
report['review_SHA256']=sha(f.parent/'astra-review-115-current.json')
report.pop('review_SHA256_correction',None)
report['completed_UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat()
report['source_binding']['current_runtime_helper']={
 'local':'glm_fp8_runtime_batch_r3068.cpp','source_SHA256':'7f994d97fc3b2a50173cf318b24d08d0b714a2258e4c4e183aab1d2f237a9a44',
 'actual_object':'/refactor/candidates/glm-transient-layer-r5681/private-15.o','object_SHA256':'8c8000737f2c2421360cdd7dedf629d42bea902540f0996b43cf3c5807db21b7',
 'provenance':'r5681-GLM-transient-CPU-status-a.json CPU-build.json compiled record; B32 CPU-build retains matchingobject; c650 builder retainsit.',
 'dispatch':'M8/16 cached807; admittedM32/64 via batchpolicy; otherM>=32 prefill780; M1..31 except8/16 returnsfalse. T15false is source-established for thishelper, not a measuredsharedMLP wallshare.'}
report['source_binding']['shared_FFN']={
 'local':'glm_moe_dsa_routed_ffn_vector_r8540.cpp','source_SHA256':'c716b37fd3fe786cece00a5b4a309d9eb13b76e4bbdb8cc8843be936c708e64b',
 'actual_object':'/refactor/candidates/glm-vector-model-r8542/vector-0.o','object_SHA256':'4b36b711d264f8e7f28ea048d3f6176ffeec9291618ea106ba4a6ba8f3941c41',
 'provenance':'GLM-vector-Model-handoff-r8544.json + retainedB32/currentc650objectmap',
 'shared_down':'Fp8MlpR19::forward_local always calls ex::Fp8BlockLinear for down afterquant, at bothT15andT16. Therefore no evidenceofT15-specific807cachecliff insharedDOWN.',
 'shared_gate_up':'Both call Fp8Projection::forward. Actualprivate-8.o source80ac6223f8fb61a9408f39172ebba773218bf936a93507b385140df96ac8d64e mapped by5681receipt, but exactsourceplaintext absent locally. OtherretainedFp8Projection implementations callr809thenfallback; cannot substitute theirhash tocloseactualprivate8. Gate/upactualbranchprovenanceUNKNOWN; noadditionalSSH authorized/used.',
 'counts_source_implied_not_measured':'Actual755has10T15decodecalls/epoch, sparse35/40layers perstage. Therefore350/400sharedMLPinvocations and350/400sharedDOWNproviderattempts perTPowner perepoch; gate/up350/400each bycallsite, actualcachehitcountUNKNOWN. Across8TPowners:2800/3200sharedDOWNattempts. These are not timings, not255/actualFPS counters.',
 'STOP':'Do not claimhugeallocation/dequantcliff orproposeT15sharedcachecomparator untilprivate8source/providerfallback/actualcounter binding closes. No implementation orGPU.'}
report['dedup'].append('737 alreadyqualified routedT13/T15synthetichiddenwholeprojectionchain+5.51/+5.70%,T16-0.45%;739Modelimage CPUlink exists and755excludesitsobjects. Notnewraggedcandidate orsharedMLP proof.582preparedW testedM8o_proj, notcurrentT15sharedMLP; methoddedupstillapplies.')
report['power2_padding_CPU_feasibility']={
 'decision':'PARK_NO_IMPLEMENTATION_NO_SECOND_EXPERIMENT_CARD',
 'authorized_direction':'Userallowspower2padding, but755native realrow/past/leaseledger andavailabledummyKV isolation musthold.',
 'source_constraints':'755frontend permits<=32rows butadmitsmax16real and32totaluniquerowlifetimes. LateT15 startsafteroneEOS, yet endedrealrowstate isnot a freshdummy; reusingwithoutreset violateslengthledger/lease/KV lifetime. Reservednewdummyrow competeswith32realrowcapacity unlessnativecontexts/hostbanks/cachebounds expand, which isnotclosed. DummypastmustnotcopyrealKVorchangesampling. Needall78layerdummyreset/lease proof andnormalcleanup; actualfreebytes isnotcapacityproof.',
 'scope':'Noassertionthatpaddingworks; noT15vs16causalclaim. Anyfuturematchedcontext testmustexcludeallpadtokens/requestsfromQA/useful/latencydenominators andretainrealarrival/firsttoken. Existingpaddingroute dedupnotexhaustivelyclosed inthisboundedtask.'}
report['single_future_diagnostic_card']['source_gate_extra']='Beforefutureprofilebuild resolveprivate8source andcurrentbridge source/objectmapping fromretainedreceipt. UntilthenSOURCEPARK; noadditionalcard orblindT15sharedcache/padding implementation.'
(f/'G115A-report-r118.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
b=read(f/'residual-budget-r118.json')
b['measurement_caveats'][0]='Localemitterprojection attribution was subsequently supplemented byonebounded16exactresultscalarread. All16SHA matchretainedaudit; seeallowner-summary755-r118.json. No synchronizedglobalcriticalpathclaim.'
(f/'residual-budget-r118.json').write_text(json.dumps(b,ensure_ascii=False,indent=2),encoding='utf8')
idx=read(f/'source-hash-index-r118.json');paths={x['path'] for x in idx['files']}
extra=['r5681-GLM-transient-CPU-status-a.json','glm_fp8_runtime_batch_r3068.cpp','glm_moe_dsa_routed_ffn_vector_r8540.cpp','GLM-vector-Model-handoff-r8544.json','GLM-irregular-vector-private-r9270b/terminal-canonical-r9271.json','GLM-irregular-vector-Model-private-r9274/source-exposure-budget-r9274.json','GLM-irregular-vector-Model-private-r9274/CPU-build.json','InfiniLM-hygon-test/AGENTS.md','InfiniLM-hygon-test/coordination/PROTOCOL.zh-CN.md','InfiniLM-hygon-test/coordination/QUALITY_POLICY.zh-CN.md','InfiniLM-hygon-test/coordination/ASTRA_REVIEW.zh-CN.md']
for p in [r/x for x in extra]+[f/x for x in ['analyze_r118.py','read755_safe_r118.py','fetch755_r118.py','finish_r118.py','finalize_r118.py','allowner-scalar755-r118.json','allowner-summary755-r118.json']]:
 if str(p) not in paths:idx['files'].append(dict(path=str(p),SHA256=sha(p),bytes=p.stat().st_size));paths.add(str(p))
for x in idx['files']:
 if pathlib.Path(x['path']).parent==f:
  x.update(SHA256=sha(pathlib.Path(x['path'])),bytes=pathlib.Path(x['path']).stat().st_size)
assert all(sha(pathlib.Path(x['path']))==x['SHA256'] for x in idx['files'])
(f/'source-hash-index-r118.json').write_text(json.dumps(idx,ensure_ascii=False,indent=2),encoding='utf8')
b['source_hash_index_SHA256']=sha(f/'source-hash-index-r118.json')
(f/'residual-budget-r118.json').write_text(json.dumps(b,ensure_ascii=False,indent=2),encoding='utf8')
receipt=dict(schema=1,report_SHA256=sha(f/'G115A-report-r118.json'),budget_SHA256=sha(f/'residual-budget-r118.json'),source_index_SHA256=sha(f/'source-hash-index-r118.json'),index_sources=len(idx['files']),all_source_hashes_match=True,all16_result_hashes_match=True,CPU_partition_assertions_passed=True,decision=report['decision'])
(f/'completion-receipt-r118.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
print(json.dumps(receipt))
