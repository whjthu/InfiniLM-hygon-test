import ast,hashlib,json,pathlib
r=pathlib.Path(__file__).resolve().parent
def put(n,s):
 if n.endswith('.py'):ast.parse(s)
 (r/n).write_bytes(s.encode())
for n in ['glm_prefill_micro_probe_r5848.py','modern_glm_prefill_micro_r5848.py','GLM_transient_preengine_r5848.py','GLM_transient_preflight_r5848.py','glm_prefill_micro_Model_r5848_inside.sh','launch_GLM_prefill_Model_r5848.py','watch_GLM_prefill_Model_terminal_r5848.py','audit_GLM_prefill_Model_r5848.py']:
 s=(r/n).read_text().replace('5848','8701').replace('candidates/glm-shared-validation-model-r5817/libglm_shared_validation_model_r5817.so','candidates/glm-prefill-window-r8701/libglm_prefill_window_r8701.so').replace('1af0b95df9db1c68f9e8c00124dbb0b331cd1b3650afd4506e3befa30f98be9d','fe179cf20aba957b2b6dbd53349dd10dbf650ef9c2ef7a0e70fb843cfc36ef6f')
 if n.startswith('glm_prefill_micro_probe'):
  s=s.replace('profile = OwnerProfile(lib, report, stage, rank)','''profile = OwnerProfile(lib, report, stage, rank)
    lib.modern_glm_window_prepare_r5797.argtypes=[ctypes.c_void_p]
    lib.modern_glm_window_arm_r5795.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int]
    lib.modern_glm_window_read_r5795.argtypes=[ctypes.c_void_p];lib.modern_glm_window_read_r5795.restype=ctypes.c_char_p
    prepare_code=lib.modern_glm_window_prepare_r5797(ctx)if stage==0 else 0
    vote=torch.tensor([float(prepare_code==0)],device='cuda');control(vote);assert vote.item()==16
    report['prefill_window_pool_prepared']=True;report['prefill_window_samples_r8701']=[]
    def arm_window(label,step,group):
        if label!='prof-I512-O8-diag' or step!=0:return
        cohort=0 if group==list(range(8))else 1 if group==list(range(8,16))else -1;assert cohort>=0
        code=lib.modern_glm_window_arm_r5795(ctx,0,cohort,1)if stage==0 else 0
        v=torch.tensor([float(code==0)],device='cuda');control(v);assert v.item()==16
    def read_window(label,step,group):
        if label!='prof-I512-O8-diag' or step!=0 or stage!=0:return
        raw=lib.modern_glm_window_read_r5795(ctx);assert raw
        sample=json.loads(raw);cohort=group[0]//8
        assert (sample['mode'],sample['stage'],sample['layer'],sample['rank'],sample['round'],sample['cohort'],sample['past'])==('prefill-round0',0,cohort,rank,0,cohort,0)
        assert sample['observed_rows']==group and sample['observed_pasts']==[0]*8 and sample['observed_counts']==[512]*8 and sample['matched_requests']==255 and 2<=sample['marker_count']<=512
        report['prefill_window_samples_r8701'].append({'phase':label,**sample});save()''')
  s=s.replace('\n                begin = time.perf_counter(); cpu_begin=time.thread_time();native_start_realtime_ns=time.time_ns()', '\n                arm_window(label,step,group)\n                begin = time.perf_counter(); cpu_begin=time.thread_time();native_start_realtime_ns=time.time_ns()')
  s=s.replace('profile.capture(label,step,group,len(flat),descriptions)\n                if capture_prefix', 'profile.capture(label,step,group,len(flat),descriptions)\n                read_window(label,step,group)\n                if capture_prefix')
  a=s.index('    perf_labels=');b=s.index('    except CandidateRejected',a)
  s=s[:a]+'''    labels=['perf-I512-O8-baseline','prof-I512-O8-diag','anchor-short-N-K4'];report['resident_plan']=[{'label':x}for x in labels];rejected=False
    report['optional_second_cycle_skipped']=report['second_cycle_not_planned']=True
    report['performance_adoption']=False;report['prefill_window_scope']='Same-load full128 O8baseline then full128 O8diagnostic. Stage0cohort0layer0/cohort1layer1; original512eventpool/stream/drain. Adjacentstream/host/CPUintervals includequeues/waits, notGPUbusy. All16 armbeforeCAPI votes onlydiagnostic; no extraModel sync.'
    try:
        cohort_epoch=time.monotonic();baseline_wall=None
        for label in labels[:2]:
            if label.startswith('prof-'):
                remaining=report['remaining_process_seconds_before_cohorts']-(time.monotonic()-cohort_epoch);need=baseline_wall*2+180
                report['diagnostic_budget']={'remaining_seconds':remaining,'required_twice_baseline_plus_QA180_seconds':need}
                vote=torch.tensor([float(remaining>need)],device='cuda');control(vote)
                if vote.item()!=16:report['diagnostic_skipped_hard_budget_reserve']=True;save();break
            before=time.monotonic();generate(label,i512_inputs,8,True,128,set(payload['chat_eos_token_ids']),False,1,True,False,True,'I512',0,False,-1,True,8)
            if baseline_wall is None:baseline_wall=time.monotonic()-before
        generate(labels[2],unique_inputs,8,True,4,set(payload['chat_eos_token_ids']),False,1,True,False,True,'short',0,False,-1,True,8)
''' +s[b:]
 if n.startswith('GLM_transient_preflight'):
  a=s.index(" receipt=r/'candidates/glm-shared-validation-model-r5817/CPU-build.json'");b=s.index(" assert (m['TP']",a)
  s=s[:a]+''' receipt=r/'candidates/glm-prefill-window-r8701/CPU-build.json';proof=json.loads(receipt.read_text());assert proof['completed']and proof['parent_DSO_SHA256']=='1af0b95df9db1c68f9e8c00124dbb0b331cd1b3650afd4506e3befa30f98be9d'and proof['library_SHA256']==m['model_SHA256']and len(proof['compiled'])==2 and len(proof['reused_objects_SHA256'])==26
 assert hashlib.sha256(pathlib.Path(proof['library']).read_bytes()).hexdigest()==m['model_SHA256']
''' +s[b:]
 if n.endswith('_inside.sh'):s=s.replace('2650 python','3500 python')
 if n.startswith('modern_'):
  s=s.replace('import time\n','import time\nPROCESS_START_R8701=time.monotonic()\n',1)
  s=s.replace('        run_cohorts(lib,ctx,stage,rank,payload,control,check,ptr,stream,vocab,report,report_path,chat_gate)',"        report['remaining_process_seconds_before_cohorts']=3500-(time.monotonic()-PROCESS_START_R8701)\n        run_cohorts(lib,ctx,stage,rank,payload,control,check,ptr,stream,vocab,report,report_path,chat_gate)")
 if n.startswith('launch_'):
  s=s.replace('a02r4n12','a02r4n14').replace('a02r4n13','a02r4n15').replace('00:45:00','01:00:00')
  # Manifest and observer hashes are sealed after all files below are final.
  s=s.replace("assert hashlib.sha256(m.read_bytes()).hexdigest()=='524adcfc7c203c741c8a4b9fbf298c234ddbe95de1ca0ad5094a0925f1011133'","assert hashlib.sha256(m.read_bytes()).hexdigest()==ROOT_REVIEWED_MANIFEST_SHA256")
  s=s.replace("assert hashlib.sha256(side.read_bytes()).hexdigest()=='6d01b33d0faa7805fa5ebafcdc7171eecc6fd7351efd3dd31974516cd94078f1'","assert hashlib.sha256(side.read_bytes()).hexdigest()==json.loads(m.read_text())['observer_manifest_SHA256']")
  s="import sys\nassert len(sys.argv)==2,'RootreviewedmanifestSHArequired_NO_GPU'\nROOT_REVIEWED_MANIFEST_SHA256=sys.argv[1]\n"+s
 if n.startswith('watch_'):s=s.replace('a02r4n12','a02r4n14').replace('a02r4n13','a02r4n15')
 if n.startswith('audit_'):
  s=s.replace('==len(plan)==5','==len(plan)==3')
  a=s.index(" d['ABBA']={}");b=s.index(' return d',a)
  s=s[:a]+''' d['prefill_windows']=[]
 for stage in (0,1):
  for rank in range(8):
   owner=json.loads((r/'results'/f'glm-transient-Model-r8701-{run}-stage{stage}-rank{rank}.json').read_text())
   samples=owner['prefill_window_samples_r8701'];assert len(samples)==(2 if stage==0 else 0)
   for sample in samples:
    c=sample['cohort'];assert(sample['mode'],sample['stage'],sample['layer'],sample['rank'],sample['round'],sample['past'])==('prefill-round0',0,c,rank,0,0)
    assert sample['observed_rows']==list(range(c*8,c*8+8))and sample['observed_pasts']==[0]*8 and sample['observed_counts']==[512]*8 and sample['matched_requests']==255
    intervals=sample['intervals'];largest=max(intervals,key=lambda v:v['stream_timeline_ms'])
    text=str(largest).lower()
    chain=('NativeAttentionProjectedR22::forward -> sparse_attention_mapped_bounded(mode2) -> pertokenQK/Softmax/PV'if 'attn.row.compute'in text else'DecoderProjectedR22::forward -> FeedForwardR19::forward -> originalgroupedroutedFP8/TPAllReduce'if any(k in text for k in['ffn','moe','expert'])else'DecoderProjectedR22::forward -> NativeAttentionProjectedR22::forward prepare/indexer/projectedKV/output boundary'if 'attn.'in text else'DecoderProjectedR22::forward -> RMSNorm/residualAdd/originallayerentryreturn boundary')
    d['prefill_windows'].append({'stage':stage,'rank':rank,'layer':sample['layer'],'cohort':c,'largest_adjacent_stream_interval':largest,'largest_adjacent_host_interval':max(intervals,key=lambda v:v['host_wall_ms']),'largest_adjacent_ownerCPU_interval':max(intervals,key=lambda v:v['threadCPU_ms']),'specific_callchain_candidate':chain,'intervals':intervals,'scope':'Candidate based on actual largest named interval; queue/dependency/hostenqueue included; not GPUbusy or exclusiveoperatorfraction'})
 d['candidate_rejected']=rejected;d['selected_counts']=chosen;d['ABBA']={};d['performance_adopted']=False
 d['qualified']=jobs[0][1:3]==['COMPLETED','0:0']and not rejected and len(d['prefill_windows'])==16
 d['scope']='Fullgold O8baseline128 + O8diag128 one-load; stage0layer0/cohort0 andlayer1/cohort1. All16normal/QA/rawCPUargmax/originalpost. Observer diagnosis only, not speed adoption.'
''' +s[b:]
 s=s.replace('Only round0 grouping4/8 differs; decode micro8/feedback cohorts8, same original5817 native/math/provider/cap16x640;', 'Round0 O8 fixed, baseline128 then diagnostic128; two observerTUs over original1af0/math/provider/cap16x640;')
 s=s.replace('PrefillMicro8/4 solevariable; original5817 oneweightload, decodeMicro8 fixed, actualfirsttoken/full128IDs/EOS/QA fixed4ordinaryarms; no newnative. Notyetqualified.', 'Oneweightload O8baseline128 + O8diagnostic128; twoeventobserverTUs, no mathchange. Stage0cohort0layer0/cohort1layer1 scope requires GPUmetadata. 60minhard/45minfirstdecisiontarget notguaranteed; no performanceadoption.')
 put(n.replace('5848','8701'),s)
print('Host baseline/diag driver, actual scope audit and60min launcher private AST ready; noModelGO')
