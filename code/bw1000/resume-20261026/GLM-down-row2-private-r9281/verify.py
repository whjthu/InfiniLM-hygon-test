import json,pathlib,sys
q=json.loads(pathlib.Path(sys.argv[1]).read_bytes());assert q['completed']and q['normal']and q['completed_math']and q['policy_closed']and q['input_readonly']and q['weight_readonly_full_byte_sum']and q['scale_readonly_sum']and q['actual_counter_errors']==0;assert len(q['trials'])==16 and len(q['warm'])==4 and [v['T']for v in q['performance']]==[16,8]
for T in [16,8]:
 z=[a for a in q['trials']if a['T']==T];assert len(z)==8
 for a in z:
  hit=True;assert a['actual_provider_counts_delta']==[3 if hit else 0,0 if hit else 3,3 if hit else 0,0 if hit else 3,0,3*T if hit else 0,3*T if hit else 0,0];assert a['actual_row2_branch_counts_delta']==([3,0,0,0]if a['mode']==0 else[0,3,0,0]);assert a['finite']and a['output_canary']and all(v['relative_RMS']<=.02 for v in a['quality_vs_first_fullchain'])
assert all(x['expert_max']-x['expert_min']+1>x['U']for x in q['routing']);print(json.dumps({'qualified':True,'performance_admission_passed':q['performance_admission_passed'],'scope':'singleGPU trainedlayer38rank0 routedprojectionchain; synthetic hidden/Torchrouter, notwholeMoE/TP/Modelquality/speed','actual_step':q['actual_step']}))
