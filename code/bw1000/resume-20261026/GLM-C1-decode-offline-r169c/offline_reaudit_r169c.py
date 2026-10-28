"""Read original 787 traces in place; export only compact aggregates, never raw event tapes."""
import pathlib,json,hashlib,ast,runpy,sys,statistics,collections,os
D=pathlib.Path(__file__).resolve().parent;R=pathlib.Path.home()/'hygon-inference-refactor-20261014';P=R/'incoming/GLM-C1-decode-Model-r169b';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();sys.path.insert(0,str(D));sys.path.insert(1,str(P));from profile_consumer_r169c import consume
output=R/'results/GLM-C1-decode-offline-r169c.json';assert not output.exists();receipt={'completed':False,'CPU_only':True,'GPU_initialized':False,'raw_trace_exported':False,'original_GPU_rerun':False,'actual_GPU_step':'910376.787','original_audit_failure_preserved':True};reads={}
try:
 m=json.loads((D/'source-manifest-r169c.json').read_bytes())
 for n,h in m['files_SHA256'].items():assert sha(D/n)==h,n
 assert sha(P/'GPU-manifest.json')==m['original_Model_source_manifest_SHA256'];assert sha(P/'audit_r169b.py')==m['original_audit_SHA256']
 source=(P/'audit_r169b.py').read_text();ns=runpy.run_path(str(P/'audit_r169b.py'),run_name='preserved_r169b_audit')
 launch=json.loads((R/'GLM-C1-decode-Model-r169b-launch.json').read_bytes());run=launch['run'];pins=json.loads((D/'owner-source-pins-r169c.json').read_bytes());expected={R/'results'/('glm-arrival-r169b-'+run+'-stage'+str(o['stage'])+'-rank'+str(o['rank'])+'.json'):o for o in pins};assert len(expected)==16
 old_consumer=runpy.run_path(str(P/'profile_consumer_r169.py'),run_name='preserved_r169_consumer')['consume']
 def read_owner(path):
  assert path in expected and path not in reads
  pin=expected[path];assert sha(path)==pin['SHA256'];q=json.loads(path.read_bytes());assert q['actual_producer']=='910376.787'
  on=q['episodes'][1];assert on['observer_enabled']and on['sparse_profile']['coverage']=='UNKNOWN'and on['sparse_profile']['error_class']=='ValueError'
  try:old_consumer(on['sparse_observer'],on['native_calls'],q['native_events'])
  except ValueError as error:assert str(error)=='invalid event tape'
  else:raise AssertionError('old failure not reproduced')
  fixed=consume(on['sparse_observer'],on['native_calls'],q['native_events']);assert fixed['coverage']=='COMPLETE';on['sparse_profile']=fixed
  reads[path]={'stage':pin['stage'],'rank':pin['rank'],'source_SHA256':pin['SHA256'],'old_consumer_failure_reproduced':True,'new_consumer_coverage':'COMPLETE'}
  return q
 tree=ast.parse(source);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='audit');edits=0
 for node in ast.walk(fn):
  if isinstance(node,ast.Assign)and len(node.targets)==1 and isinstance(node.targets[0],ast.Name)and node.targets[0].id=='q'and ast.unparse(node.value)=='json.loads(p.read_bytes())':node.value=ast.Call(func=ast.Name(id='read_owner',ctx=ast.Load()),args=[ast.Name(id='p',ctx=ast.Load())],keywords=[]);edits+=1
 assert edits==1;ns['read_owner']=read_owner;exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(P/'audit_r169b.py')+'+offline_read_hook','exec'),ns);original=ns['audit']();assert len(reads)==16
 for path,item in reads.items():assert sha(path)==item['source_SHA256']
 def aggregate_profile(owner):
  windows=owner['profile']['windows'];assert len(windows)==3;groups=collections.defaultdict(list);types=collections.defaultdict(list);detail=collections.defaultdict(list);projections=collections.defaultdict(lambda:collections.defaultdict(list));occurrences=collections.Counter()
  for w in windows:
   for key in ['owner_CAPI','PP_incoming','PP_outgoing']:
    for clock,value in w[key].items():groups[key+'.'+clock].append(value)
   bykind=collections.defaultdict(float)
   for layer in w['layers']:bykind[layer['kind']]+=layer['stream_interval_ms']
   for key,value in bykind.items():types[key].append(value)
   for row in w['nested_detail_not_additive']:detail[row['name']].append(row['stream_interval_ms'])
   for row in w['projection_occurrences_not_additive']:
    occurrences[row['name']]+=1
    for key in ['quant','after_quant_to_before_original_fence','quant_plus_projection']:projections[row['name']][key].append(row[key]['stream_interval_ms'])
  return {'stage':owner['stage'],'rank':owner['rank'],'source_SHA256':owner['SHA256'],'actual_step':owner['actual_step'],'normal':owner['normal'],'samples':3,'main_stream_interval_and_host_span_ms_medians':{k:statistics.median(v)for k,v in groups.items()},'layer_kind_sum_stream_interval_ms_medians':{k:statistics.median(v)for k,v in types.items()},'representative_detail_inclusive_stream_interval_ms_medians':{k:statistics.median(v)for k,v in detail.items()},'projection_occurrence_counts_all_3_samples':dict(occurrences),'projection_inclusive_stream_interval_ms_medians':{k:{z:statistics.median(v)for z,v in fields.items()}for k,fields in projections.items()},'overhead':owner['overhead'],'TP_only':'UNKNOWN','GPU_busy':'UNKNOWN','nested_not_additive':True}
 owners=[aggregate_profile(o)for o in original['owners']]
 rows=[]
 for row in original['episodes']:
  keep=['label','observer_enabled','clean_TPS','n_requests','max_admitted_live','task_tok_s','visible_tokens','task_seconds','correct','EOS','caps','prefill_input_per_HOST_CAPI_s','all_decode_visible_per_HOST_CAPI_s_including_early','post_allprefill_common_output_tok_s']
  item={k:row[k]for k in keep};item['aggregate_metrics']={k:row['metrics'][k]for k in ['user_decode_tok_s','TTFT_seconds','ITL_seconds','overall_output_tok_s_per_card','output_cap_requests']};rows.append(item)
 receipt.update(completed=True,recovered_audit_qualified=original['qualified'],execution_qualified=original['execution_qualified'],quality_passed=original['quality_passed'],all16_profile_coverage=original['all16_profile_coverage'],all16_overhead_screen_passed=original['all16_overhead_screen_passed'],scope=original['scope'],token_identity_gate=False,original_audit_predicates_unchanged=True,only_original_audit_AST_change='q=read_owner(p) with in-memory reparse only',source_manifest_SHA256=sha(D/'source-manifest-r169c.json'),owners=owners,episodes=rows,all16_original_sources_unchanged=True,source_identity=list(reads.values()),no_per_event_timestamps_or_native_event_arrays_in_output=True)
except BaseException as error:receipt.update(error_class=type(error).__name__,source_files_read=len(reads))
with output.open('x')as f:json.dump(receipt,f,indent=2)
print(json.dumps({'completed':receipt['completed'],'recovered_audit_qualified':receipt.get('recovered_audit_qualified',False),'path':str(output),'SHA256':sha(output),'bytes':output.stat().st_size,'error_class':receipt.get('error_class'),'raw_trace_exported':False}));assert receipt['completed']
