import pathlib,tempfile,json,time,types,sys
from host_observer_r9183 import HostObserver,TimedControl,Reporter
from shard_head_contract_r9183 import read_head
with tempfile.TemporaryDirectory(prefix='r9183-observer-')as tmp:
 root=pathlib.Path(tmp);obs=HostObserver();report={'node':0,'run':'cpuonly','phase':'decode','history':[{'ids':list(range(256))}for _ in range(1024)]};logger=Reporter(root/'full.json',report,obs)
 for i in range(3):logger.progress({'label':'samecase','steps':8*(i+1),'active_C':16})
 assert not(root/'full.json').exists()and logger.progress_path.stat().st_size<1024
 first=obs.snapshot();control=TimedControl(types.SimpleNamespace(collect=lambda label,value:[value]),obs);assert control.collect('memory-guard:case',{'x':1})==[{'x':1}];logger.set_logging_diag(True);logger.progress({'label':'samecase','steps':32,'active_C':16});assert(root/'full.json').stat().st_size>100000;logger.set_logging_diag(False);logger.progress({'label':'samecase','steps':40,'active_C':16});stats=obs.delta(first);assert stats['counters']['full_report_write_calls']==1 and stats['counters']['scalar_progress_write_calls']==1 and stats['counters']['StageControl_collect_calls']==1
 assert json.loads((root/'full.json').read_bytes())['history']==report['history'];logger.full();assert obs.values['full_report_write_calls']==2 and obs.values['scalar_progress_write_calls']==4
io={'CPU_only':True,'GPU':False,'tinyprogress_no_fullhistory_write':True,'legacy_logging_diag_only_when_requested':True,'full_raw_history_preserved':True,'serialize_write_bytes_calls_times_real_tempfiles':dict(obs.values),'nested_Control_guard_times_not_pureIO':True}
# Actual schema validator rejects the former full-storage class and wrong owner rows.
for node in range(4):
 tail=node>=2;owners=[]
 for dev in range(8):
  rank=node%2*8+dev;o={'device':dev,'model_constructed':True,'model_RTTI':'KimiK3ForConditionalGenerationShardR9183','TP':16,'tp_rank':rank,'PP':2,'PP_stage':node//2,'head_expected':tail}
  if tail:o.update(head_constructed=True,head_RTTI='ShardOnlyVocabHeadR9183',head_is_ReplicatedLinear=False,full_GPU_head_retained=False,physical_head_shape=[10240,7168],head_parameter_bytes=146800640,global_row_begin=rank*10240,global_row_end_exclusive=(rank+1)*10240,postload_rows_verified=True,CPU_expected_row_digests=[1,2,3],GPU_actual_row_digests=[1,2,3])
  owners.append(o)
 info={'schema':1,'head_kind':'ShardOnlyVocabHeadR9183','legacy_full_shard_policy_switch_supported':False,'expected_CPU_source_registered':tail,'owners':owners};lib=types.SimpleNamespace(modern_kimi_shard_head_info_r9183=lambda:json.dumps(info).encode());read_head(lib,node,2,True)
 if tail:
  for key,value in [('full_GPU_head_retained',True),('physical_head_shape',[163840,7168]),('GPU_actual_row_digests',[9,2,3])]:
   old=info['owners'][0][key];info['owners'][0][key]=value
   try:read_head(lib,node,2,True)
   except AssertionError:pass
   else:raise AssertionError(key)
   info['owners'][0][key]=old
io['head_type_shape_and_loaded_source_schema_positive_negative']=True;pathlib.Path(__file__).with_name('CPU-observer-head-contract-r9183.json').write_bytes(json.dumps(io,indent=2).encode());print(json.dumps(io))
