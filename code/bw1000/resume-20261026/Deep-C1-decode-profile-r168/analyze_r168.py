"""Compact C1 diagnosis; no token text/activations, no Model speedup claim."""
from pathlib import Path
import json,sys,hashlib,statistics,math
src=Path(sys.argv[1]);dst=Path(sys.argv[2]);assert not dst.exists();d=json.loads(src.read_bytes())
assert d['completed']and d['engine_worker_shutdown_completed']and d['grid_complete']and d['owner_capture_complete']
assert d['library_sha256']=='ac9370c8f188bfda8555718c52690b914883fc6b02b9267cbd279be3ecdd3835'
trials=d['quality_trials'];assert len(trials)==4 and all(t['B']==1 for t in trials)
out={'scope':'Actual currentC1 fourfixedQA diagnostic, notnewoptimization/cleanTPS/HTTP/qualitynoninferiority','source':str(src),'source_SHA256':hashlib.sha256(src.read_bytes()).hexdigest(),'Native_SHA256':d['library_sha256'],'loaded_once_s':d.get('loading_seconds'),'profile_steps':[],'requests':[],'quality':d['grid']['1'],'all4_actual_T1_frames':True,'hardware_peak':'UNKNOWN','main_bucket_interpretation':'16 old stages nested; individualstage sums over43seriallayers perdevice, never sum devices/nested stages. GPUevent streamspan includes host launch/dependency/commwait, notGPUbusy; hostwall-threadCPU is inclusivewait/preemption clue, notnetwork time.'}
for t in trials:
 rec=t['records'][0];req=t['requests'][0];stamps=[e['time']for e in req['events']if e.get('visible')and e['kind']=='committed'];rate=(len(stamps)-1)/(stamps[-1]-stamps[0])if len(stamps)>1 else None
 gaps=[];windows=t['windows']
 for a,b in zip(windows,windows[1:]):gaps.append(max(0.,b['return_monotonic']-b['seconds']-a['return_monotonic']))
 off=[w['seconds']for w in windows if not w['prefill']and not w['profiled']]
 out['requests'].append({'id':rec['id'],'correct':bool(rec['score']['flexible_correct']),'stop_reason':rec['stop_reason'],'visible_tokens':len(stamps),'user_decode_tok_s_including_observer':rate,'TTFT_s':stamps[0]-req['arrival']if stamps else None,'OFF_CAPI_decode_median_s':statistics.median(off)if off else None,'host_intercall_gap_median_s':statistics.median(gaps)if gaps else None,'host_intercall_gap_max_s':max(gaps,default=None),'adjacent_OFF_PROFILE_OFF':t['adjacent_OFF_PROFILE_OFF'],'scope':'EmitterCAPIreturn, arrivals includeprefill; observerandIOincluded, notcleanbenchmark'})
for p in d['decode_profiles']:
 assert p['geometry']['tokens']==p['geometry']['requests']==1 and p['owner_rows']==p['inner_rows']==344 and p['inner_event_validation_passed']and p['native_window_closed']
 stages=p['old_perstage_perdevice_ms'];host=p['old_host_wall_perstage_perdevice_ms'];cpu=p['old_thread_cpu_perstage_perdevice_ms']
 chosen=['decoder_envelope','metadata_validation_readback','attention','routed_experts','shared_experts','TP_reduction','FFN_norm_sync','router','output_cast']
 buckets={n:{'stream_ms_by_device':stages[n],'host_wall_ms_by_device':host[n],'thread_cpu_ms_by_device':cpu[n],'maximum_owner_stream_ms':max(stages[n]),'maximum_owner_host_wall_ms':max(host[n])}for n in chosen}
 out['profile_steps'].append({'label':p['label'],'geometry':{k:p['geometry'][k]for k in ['tokens','requests','past','sizes']},'Native_CAPI_s':p['native_forward_wall_s'],'observer_total_s':p['observer_total_wall_s'],'arm_s':p['arm_wall_s'],'packet_IO_and_validation_s':p['Python_validation_and_packet_io_wall_s'],'buckets':buckets,'attention_inner_six_siblings_ms_by_device':p['inner_persegment_perdevice_ms'],'attention_kind_buckets':p['kinds'],'actual_C1_counter_contract':p['C1_actual_attention_contract'],'scalar_packets':p['scalar_packets'],'attention_collective_limit':'output_projection_collective combinesprojection/cast/collective/sync; no individual NIC/sync split'})
dst.write_bytes((json.dumps(out,indent=2)+chr(10)).encode());print(json.dumps({'saved':str(dst),'profiles':len(out['profile_steps']),'quality':out['quality']}))
