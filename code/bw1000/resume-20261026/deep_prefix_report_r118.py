import hashlib,json
from pathlib import Path
from datetime import datetime,timezone

OUT=Path(__file__).parent
REPO=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
review=OUT/'astra-review-115-current.json'
prior=REPO/'Deep-MTP-feasibility-r9287.json'
assert sha(review)=='5b5a91b011d3c28e29dca102a1e0344501bd5aed2e7ed8b70368eb8a94a2636c'
assert sha(prior)=='14e009f91f31fecb047a873a8f2b74335fb65cba348c455b19c6bbc293e5e4bc'
b=load(REPO/'Deep-MTP-audit-source-bundle-r9287.json')
oracle=load(OUT/'deep_prefix_oracle_result_r118.json')
sources=[]
for name,rec in b['prior_relevant_sources'].items():
    text=rec.get('text','')
    if not text:continue
    anchors=[]
    needles=['void PagedLatentCache::store','void PagedLatentCache::reset','stride_(',
             'auto pending=compressor_->prepare','auto index_pending=indexer_->prepare',
             'cache_->store(r,device_ids(commit_slots)','void CsaAttention::reset',
             'void HcaAttention::reset','void SwaAttention::reset']
    for n,line in enumerate(text.splitlines(),1):
        if any(x in line for x in needles):anchors.append(dict(line=n,text=line[:900]))
    sources.append(dict(path=name,SHA256=rec['SHA256'],anchors=anchors))
sources.append(dict(path=str(REPO/'Deep-C4-smallT-private-r9252/deep_c4_smallt_r9252.cpp'),
                    SHA256=sha(REPO/'Deep-C4-smallT-private-r9252/deep_c4_smallt_r9252.cpp'),
                    anchors=[dict(line=n,text=line[:1500]) for n,line in enumerate((REPO/'Deep-C4-smallT-private-r9252/deep_c4_smallt_r9252.cpp').read_text().splitlines(),1) if 'bool all_selected=' in line or 'void C4Indexer::commit' in line]))
bytes_budget={
 'units':'bytes per active physical row per TP owner; not multiplied by8 for per-device fit',
 'SWA_count':2,'CSA_count':21,'HCA_count':20,
 'per_layer_ring_two_overwritten_logical_payload':2*584,
 'ring_all43_two_logical_payload':43*2*584,
 'ring_all43_retained_BF16_KV_alternative':43*2*512*2,
 'CSA_per_layer_score512_two':2*4*512*2,
 'CSA_per_layer_score128_two':2*4*128*2,
 'CSA_all21_retained_projection_scores':21*2*4*(512+128)*2,
 'HCA_per_layer_projection_scores_two':2*2*512*2,
 'HCA_all20_retained_projection_scores':20*2*2*512*2,
 'CSA_per_layer_base_compressor_state512':8*4*512*2,
 'CSA_per_layer_base_compressor_state128':8*4*128*2,
 'HCA_per_layer_base_compressor_state512':128*2*512*2,
 'all43_existing_compressor_state':21*8*4*(512+128)*2+20*128*2*512*2,
 'CSA_all21_two_selection_rows_topk512_int64_upper':21*2*512*8,
 'all43_new_boundary_latents_BF16_upper':41*512*2,
 'CSA_all21_new_boundary_index_packed_upper':21*(128+4),
 'target_four_stream_residual_two_positions':2*4*4096*2,
 'greedy_ids_two_positions_int64':2*8,
 'CAPI_float32_host_logits_two_positions_if_retained':2*129280*4,
 'one_ring_page_block256_stride':((256*584+575)//576)*576,
 'all43_one_shadow_ring_page_each_upper':43*((256*584+575)//576)*576,
 'notes':[
  '584 is physical logical-entry payload from page layout formula, not a promise that bytes are contiguous; page padding and written flags must be preserved.',
  'Projected-score layout comes from local compressor substrate SHA listed below, not newly rebound current DSO object. Actual live allocation and compositor ABI remain unqualified.',
  'Existing prepare clones full all-row compressor state once per compressor. Additional full-state copies sum6103040 bytes per row; do not count old base references as new allocations.',
  'An optimized overlay could retain two projection records and apply accepted writes only, avoiding full-state-per-position snapshots. This requires a new row-slice persist operation or preallocated COW state; current commit swaps full next_state.',
  'Ring overlay gather is a new interface. Retaining BF16 KV alone does not restore physical packed bytes after in-place mutation. Native slot packing and COW ownership need qualification.',
  'Upper sums exclude tensor allocator granularity, compressed-cache page COW, rank votes/fences, MTP weights/workspaces, attention banks/full index scores/topk temporaries, host vectors, and driver/CCL memory. No fit or peak bound claim.'
 ]}
bytes_budget['retained_payload_upper_using_BF16_KV_and_greedy_ids']=sum(bytes_budget[k] for k in [
 'ring_all43_retained_BF16_KV_alternative','CSA_all21_retained_projection_scores',
 'HCA_all20_retained_projection_scores','CSA_all21_two_selection_rows_topk512_int64_upper',
 'all43_new_boundary_latents_BF16_upper','CSA_all21_new_boundary_index_packed_upper',
 'target_four_stream_residual_two_positions','greedy_ids_two_positions_int64'])
report={
 'schema':1,'card':'D115A','revision':'r118','UTC':datetime.now(timezone.utc).isoformat(),
 'decision':'STOP_CURRENT_DSO_ATOMIC_PREFIX_API_MISSING; SYMBOLIC_CONTRACT_EXPRESSIBLE_ONLY',
 'scope':'Incremental CPU accepted-prefix contract and bounded oracle only; no MTP implementation, checkpoint-key reinventory, native math, compilation, loading, or performance test.',
 'inputs':[dict(path=str(review),SHA256=sha(review)),dict(path=str(prior),SHA256=sha(prior)),
           dict(path=str(REPO/'Deep-MTP-audit-source-bundle-r9287.json'),SHA256=sha(REPO/'Deep-MTP-audit-source-bundle-r9287.json'))],
 'policies_read':['InfiniLM-hygon-test/AGENTS.md','coordination/PROTOCOL.zh-CN.md','coordination/ASTRA_REVIEW.zh-CN.md','coordination/QUALITY_POLICY.zh-CN.md'],
 'incremental_findings':[
  'Length truncation fails even at K1: past128 verify positions128/129 overwrites ring0/1; accepting only anchor must retain old ring1 bytes and validity. Compressor future scores, new compressed/key entries, selection and revisions also cannot be repaired by length alone.',
  'Each HCA/CSA writes ring and commits compressor(s) inside attention before decoder FFN completes. Any later layer/FFN/head failure can leave earlier caches advanced. Existing permodule poison blocks reuse; it is fail-stop, not rollback.',
  'C4Indexer owner/revision protects an individual pending, but revision is global and increments on full commit. It does not provide a model-wide row transaction or invalidation spanning43 layers. Current reset clears all rows; per-row cancellation/reuse needs a new generation contract.',
  'Constant-history-size transaction is expressible with deferred overlays, accepted projected-score slices and compressed-boundary filter. It is unavailable through existing CAPI/attention/decoder methods. No native atomic43-state proof was attempted or claimed.',
  'Two-position verification fails both original and smallT allselected guards since each request.tokens=2. Charge query projection/quantization, index score/TP reduction/topk plus normal compressors for all21 CSA layers. Also current C4 tile projection guard requires requests.size()==tokens and tokens==1 per request, so that path cannot be inherited.'
 ],
 'two_position_contract':{
  'entry':'Target cache past=P; current target-generated anchor is atP; draft proposal atP+1 comes from MTP(previous target residual,anchor). Never substitute normalized4096 hidden for four-stream residual.',
  'verify':'One tentative target call [anchor,draft], logical positions[P,P+1], both target logits/greedy IDs and both [4,4096] residuals. proposal is compared with target logits after anchor; replacement/bonus drawn from the appropriate current target distribution.',
  'accept0':'Commit m=1 input (anchor) at all43 layers; discard proposal state; emit replacement from logits(anchor), unless terminal policy stops. No additional target replay is allowed in speed-capable path.',
  'accept1':'Commit m=2 inputs; emit accepted proposal and, if nonterminal, bonus next anchor from logits(proposal). Terminal accepted EOS emits no bonus.',
  'zero_input_abort':'Cancellation/prepublication error commits m=0 and emits nothing; this is distinct from accept0 drafts, which commits anchor m=1.',
  'EOS':'Known EOS anchor short circuits before draft/verify. Rejected EOS proposal is ignored. Accepted EOS proposal stops after prefix2. EOS replacement from first logit commits only anchor prefix1. Terminal row release may skip MTP catchup; it must be explicit, not silently treated as aligned live state.',
  'cap':'Cap remaining1 emits only first permitted token. If proposal would be accepted but no budget for bonus, skip bonus and release terminal row; retain/report cap, never label it EOS.',
 },
 'per_layer_write_set':{
  'all43':'Physical row+generation, model epoch/revision; wrapper/decoder/attention lengths; raw ring positions(P+t)%128 for t<m and corresponding written flags. Tentative KV and ring overlay never touch published base.',
  'SWA_layers':[0,1],
  'CSA_layers':list(range(2,43,2)),
  'HCA_layers':list(range(3,43,2)),
  'CSA_compressors':'For D512 and D128, persist only score rows t<m to slot(row*2+floor((P+t)/4)%2)*4+(P+t)%4. Preserve other old slots. Keep boundary outputs only when t<m and(P+t+1)%4==0. compressed/key length=floor((P+m)/4). Store packed key128+float scale4 only for accepted boundaries; discard future written flags. last_selection retains the last committed query selection, or an explicitly specified accepted-prefix vector.',
  'HCA_compressor':'Persist D512 score rows t<m to row*128+(P+t)%128. Keep boundary outputs only when t<m and(P+t+1)%128==0. compressed length=floor((P+m)/128).',
  'metadata':'Every pending carries model owner, owner-thread/rank identity, parent epoch, row generation, row past and exact proposal count. Seal validates all43 plans and every participating owner before publication; duplicate, foreign, stale, cross-generation or incomplete cohort is rejected before any write.',
  'failure_atomicity':'Required future design stages validated new packed slots/row compressor states as immutable overlays, synchronizes all owner streams, then a shared coordinator publishes one epoch after all8 owners acknowledge43-layer readiness. No next forward/emit before all publish acknowledgements. In-place writes require exact undo bytes or page COW before mutation. Current tensor-only cache access cannot do this. GPU/device loss after publication remains fail-stop; oracle does not prove recovery.',
  'reset_cancel':'Abort invalidates model epoch and pending handles; row reset increments generation before row reuse and clears all43 row state+MTP row state. Existing whole-model reset is the only available fallback and disrupts other rows; not a proposed serving solution.'},
 'boundary_examples':[
  dict(P=2,positions=[2,3],accept0_m1='No C4 boundary accepted; prepared boundary at position3 discarded.',accept1_m2='C4 entry0 accepted.'),
  dict(P=3,positions=[3,4],accept0_m1='C4 entry0 accepted; score at4 discarded.',accept1_m2='C4 entry0 plus next partial group score at4.'),
  dict(P=126,positions=[126,127],accept0_m1='No HCA entry0; rejected boundary at127 discarded.',accept1_m2='HCA entry0 accepted.'),
  dict(P=127,positions=[127,128],ring_slots=[127,0],accept0_m1='Keep old ring0; commit HCA entry0 and C4 entry31.',accept1_m2='Replace ring0 and begin next partial groups.'),
  dict(P=128,positions=[128,129],ring_slots=[0,1],accept0_m1='Replace ring0 only; old ring1 remains authoritative.'),
  dict(P=2047,positions=[2047,2048],indexer='Visible compressed counts512/512; still full indexer since tokens==2. AtP2050 second query sees513 and real topk pruning matters.')],
 'bytes_per_row':bytes_budget,
 'catchup_and_calls':[
  dict(case='nonterminal reject accept0',target_verify_calls=1,target_input_positions=2,target_replay_calls=0,mtp_proposal_calls=1,additional_mtp_catchup_calls=0,mtp_input_positions=1,max_new_emitted_tokens=1),
  dict(case='nonterminal accept1',target_verify_calls=1,target_input_positions=2,target_replay_calls=0,mtp_proposal_calls=1,additional_mtp_catchup_calls=1,mtp_input_positions=2,max_new_emitted_tokens=2,
       detail='Before next proposal, consume accepted draft with target residual after anchor to advance MTP state; then next proposal consumes bonus anchor with target residual after accepted draft. Could pack these2 MTP inputs into one future call, but still charges2 positions and requires aligned both residuals.'),
  dict(case='accepted draft EOS / replacement EOS / cap release',target_verify_calls=1,target_input_positions=2,target_replay_calls=0,mtp_proposal_calls=1,additional_mtp_catchup_calls=0,
       detail='No next live cycle; terminal state released explicitly. Actual useful emitted count can be below1+acceptance.'),
  dict(case='known EOS anchor',target_verify_calls=0,mtp_proposal_calls=0,additional_mtp_catchup_calls=0),
  dict(case='diagnostic restore+accepted replay fallback',target_verify_calls=1,additional_target_replay_calls=1,
       detail='Replays1 or2 accepted inputs after exact snapshot restoration; correctness diagnostic only, likely destroys K1 call reduction. Full history replay at each cycle is STOP.')],
 'oracle':dict(path=str(OUT/'deep_prefix_oracle_result_r118.json'),SHA256=sha(OUT/'deep_prefix_oracle_result_r118.json'),
               source=str(OUT/'deep_prefix_oracle_r118.py'),source_SHA256=sha(OUT/'deep_prefix_oracle_r118.py'),**{k:v for k,v in oracle.items() if k!='case_matrix'}),
 'source_evidence':sources,
 'substrate_only_sources':[dict(path=str(REPO/p),SHA256=sha(REPO/p),qualification='Local source layout/algorithm substrate only; this audit does not establish current linked object identity.') for p in ['deep_compressor_position_proof_r8726.cpp','v4_compress_public_kernel.cu','deepseek_v4_native_routed_decoder_r20.cpp']],
 'STOP_gap':'No new native work authorized. Existing DSO cannot defer/restore complete host validity and private lengths/revisions, cannot gather from tentative ring overlays, cannot atomically select prefix across43 layers/8 owners, and cannot return target hidden/all-position outputs through current CAPI. CPU symbolic PASS does not close these gaps. Root should park MTP implementation/GPU pending a deliberate source-level investment decision.',
 'performance':'UNKNOWN. Required measured cyclecost < (1+actual_acceptance)*matched_ARstepcost, counting MTP proposal+accepted-token catchup, full21-layer verify indexer, overlays/fences/commit and terminal/ragged behavior. No2x claim; no15TPS prediction; no multiplication of per-layer component gains.',
 'execution':dict(GPU=0,SSH=0,Slurm=0,weights=0,compile=0,dlopen=0,upload=0,Git=0,repo_mutations=0,oracle_local_CPU_only=True),
}
dst=OUT/'Deep-prefix-transaction-private-r118.json'
dst.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
index=dict(schema=1,card='D115A',revision='r118',status='COMPLETE_CPU_STOP_CURRENT_NATIVE',
           report=dict(path=str(dst),SHA256=sha(dst)),
           oracle=dict(path=str(OUT/'deep_prefix_oracle_result_r118.json'),SHA256=sha(OUT/'deep_prefix_oracle_result_r118.json')),
           source=dict(path=str(OUT/'deep_prefix_oracle_r118.py'),SHA256=sha(OUT/'deep_prefix_oracle_r118.py')),
           generator=dict(path=str(Path(__file__)),SHA256=sha(Path(__file__))))
(OUT/'Deep-prefix-transaction-index-r118.json').write_text(json.dumps(index,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(report_SHA256=sha(dst),index_SHA256=sha(OUT/'Deep-prefix-transaction-index-r118.json'),bytes=bytes_budget),ensure_ascii=False))
