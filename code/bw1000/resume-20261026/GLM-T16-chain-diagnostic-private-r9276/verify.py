import json,pathlib,sys
q=json.loads(pathlib.Path(sys.argv[1]).read_bytes());assert q['completed']and q['normal']and q['completed_math']and q['policy_closed']and q['input_readonly']and q['weight_readonly_full_byte_sum']and q['scale_readonly_sum']and q['actual_counter_errors']==0;assert len(q['measurements'])==16 and len(q['warm'])==1
for z in q['measurements']:
 assert z['finite']and z['output_canary']and z['actual_provider_counts_delta']==[1,0,1,0,0,16,16,0]and all(v['relative_RMS']<=.02 for v in z['quality'])
assert q['activity_available'],'NO_DEVICE_ACTIVITIES_not_reliable_diagnostic';assert q['routing'][0]['T']==16
print(json.dumps({'qualified':True,'scope':'singleGPU T16 routedchain diagnostic only; notModelgain','actual_step':q['actual_step']}))
