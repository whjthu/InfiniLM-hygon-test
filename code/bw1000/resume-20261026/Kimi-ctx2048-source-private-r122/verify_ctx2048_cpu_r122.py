import pathlib,json,hashlib,sys,ast
D=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(D));from ctx2048_contract_r122 import Geometry,feature_frame_bytes,MAX_BODY
G=1024**3;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();g=Geometry().validate();checks=0
for row in range(1,17):
 assert len(g.table(row))==8 and min(g.table(row))>=1 and max(g.table(row))<132
 for pos in [0,255,256,511,512,1023,1024,1535,2047]:assert g.slot(row,pos)//256 in g.table(row);checks+=1
assert len({p for row in range(1,17)for p in g.table(row)})==128 and max(g.table(16))==128
for row,pos in [(0,0),(17,0),(1,-1),(1,2048)]:
 try:g.slot(row,pos)
 except AssertionError:checks+=1
 else:raise AssertionError('invalidslotaccepted')
assert g.admit(1024,1024)and g.admit(576,1024)and not g.admit(1025,1024)and not g.admit(1024,1025);checks+=4
assert feature_frame_bytes(1024)==73400320 and feature_frame_bytes(1024)>64*1024**2 and feature_frame_bytes(1024)<=MAX_BODY
try:feature_frame_bytes(1025)
except AssertionError:checks+=1
else:raise AssertionError('oversizeframeaccepted')
# PureCPU lifecycle reference, not native execution: retainedprefix across1k and terminalcontinuation0.
for start in [1016,1023,1024,2039,2040,2047]:
 width=min(8,2048-start);assert width>=1
 for accepted in range(width):
  retained=accepted+1;assert start+retained<=2048;active,shadow=1,9
  # Nonterminal commits/replay advance exactretained and swap banks; terminal doesnotcommit shadow.
  newpast=start+retained;active,shadow=shadow,active;assert active==9 and shadow==1
  terminalpast=start;assert terminalpast==start;checks+=1
old=D.parent/'Kimi-cap8-micro1-Model-private-r9293';data=json.loads((old/'frozen-data-r9293.json').read_bytes());proof={x['id']:x for x in data['prompt_proof']};mmlu=data['MMLU_planned32'];assert len(mmlu)==32
for q in mmlu:pr=proof[q['id']];assert pr['all_model_IDs_valid']and g.admit(pr['input_tokens'],1024);checks+=1
assert proof['mmlu-test-11704']['input_tokens']==576
native=(D/'sources-before/pair_probe.cpp').read_bytes().decode();cache=(D/'sources-before/actual-kimi_engine_input_r145.hpp').read_bytes().decode();payload=(D/'sources-before/batch_payload_r9122c.hpp').read_bytes().decode();taps=(D/'sources-before/batch_live_taps_r9122.hpp').read_bytes().decode();allocator=(D/'sources-before/kimi_compact_allocate_r4974.cpp').read_bytes().decode();pair=(D/'sources-before/kimi_pair_state_r9006.hpp').read_bytes().decode()
assert 'if(c->pages_per_row_r4974!=4)'in native and 'table(count*4)'in native and 'total[i]>1024'in native
assert 'if(c->num_blocks()==4*(rows+1))return 4'in cache and '8*(rows'not in cache
assert 'x[1]+x[5]>1024'in payload and 'blocks->shape()!=infinilm::runtime::Shape{C,4}'in taps
assert 'num_blocks() / 4'in allocator and 132//4==33 and 33!=g.recurrent_pool
assert 'size_t(past)+ids.size()>1024'in pair;checks+=6
budget_source=json.loads((D.parent/'Kimi-PP2-cap32-budget-private-r9201/source-proof-r9201.json').read_bytes());cfg=budget_source['config'];lin=cfg['linear_attn_config'];assert lin['num_heads']==96 and lin['head_dim']==128 and cfg['num_attention_heads']==96
per_page=2*256*(96//16)*192*2;assert per_page==1179648
rows=[];observed=6381633536
for stage,(a,b)in enumerate([(0,46),(46,93)]):
 kda=sum(a<=v-1<b for v in lin['kda_layers']);mla=b-a-kda;oldpages=68;newpages=132;delta=(newpages-oldpages)*per_page*mla;RNNrow=(3*6*128*3+6*128*128)*2
 rows.append({'stage':stage,'range':[a,b],'KDA_layers':kda,'MLA_layers':mla,'oldpages':oldpages,'newpages':newpages,'old_and_new_recurrent_pool':17,'additional_KV_payload_bytes_perowner':delta,'additional_KV_payload_GiB':delta/G,'counterfactual_if_allocator_unfixed_pool33_extra_RNN_bytes':(33-17)*RNNrow*kda,'conditional_free_if_only_KV_delta':observed-delta if stage==1 else None,'conditional_tail_floor_margin_GiB':(observed-delta-4*G)/G if stage==1 else None})
assert rows[1]['additional_KV_payload_bytes_perowner']==981467136 and rows[1]['conditional_free_if_only_KV_delta']>4*G
for p in D.rglob('*.py'):assert b'\r'not in p.read_bytes();ast.parse(p.read_bytes())
report={'schema':1,'passed':True,'CPU_only':True,'GPU':False,'native_dlopen':False,'Runtime_created':False,'compiled':False,'uploaded':False,'new_Slurm_steps':0,'checks':checks,'status':'UNCHANGED_NATIVE_BLOCKED_EXACTGUARDS; 132pagegeometryandbudgetreviewable_notGPUqualified','geometry':{'logical':8,'physical':16,'KDA_pool':17,'pages_per_row':8,'pages':132,'reserved_blocks':4,'max_physical_page_used':128,'max_context':2048,'input_max':1024,'output_cap':1024},'budget':rows,'budget_scope':'cachepayloadonlyconditional; additionalforwardworkspace/prefill1024/fragmentation/allocatorlifetime/hostbuffer UNKNOWN; cannotapproveModelonstaticmargin','old_observed_min_bytes':observed,'old_observation_source':'safe-terminal-or-progress-r9302-a.json completedMMLUg2OFF, no new474poll','all32MMLU_eligible_under_newlimits':True,'previous11704_eligible_576plus1024':True,'draft_frame_bytes1024':feature_frame_bytes(1024),'new_MAX_BODY':MAX_BODY,'draft_rawcache_max8x2048x35840BF16_bytes':8*2048*35840*2,'draft_peak_working_storage':'UNKNOWN old+stagedconcat/packedquerycontexts coexist; notonlyrawcachebytes','no_current_frozen_source_modified':True,'native_base_SHA256':'3057262312679f86bcdb181ce198b9b1f1d2fccc59fc204b4adbc197132b60dd'};p=D/'CPU-budget-boundary-r122.json';p.write_bytes((json.dumps(report,indent=2)+'\n').encode());print(json.dumps({'passed':True,'checks':checks,'tail_extra_KV_bytes':rows[1]['additional_KV_payload_bytes_perowner'],'conditional_tail_free_bytes':rows[1]['conditional_free_if_only_KV_delta'],'conditional_tail_marginGiB':rows[1]['conditional_tail_floor_margin_GiB'],'GPU_GO':False}))
