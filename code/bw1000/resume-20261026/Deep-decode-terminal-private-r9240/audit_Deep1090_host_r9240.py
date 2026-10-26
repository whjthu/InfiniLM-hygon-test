import pathlib,json,sys,hashlib
R=pathlib.Path.home()/'hygon-inference-refactor-20261014'
def scalar_host_path(name):
 v=pathlib.PurePosixPath(name);assert str(v).startswith('/refactor/results/deep-decode-profile-r9234-910374.1090-')and'..'not in v.parts;return R/'results'/v.name
p=pathlib.Path(sys.argv[1]);q=json.loads(p.read_bytes());assert q['completed']and q['collection_finished']and q['engine_worker_shutdown_completed']and q['grid_complete']and q['owner_capture_complete']and q['instrumented_observation_only']and not q['performance_qualified'];assert len(q['decode_profiles'])==2 and len(q['quality_trials'])==1;assert {v['capture']for v in q['decode_profiles']}=={'B32','EOS_tail'}
for x in q['decode_profiles']:
 assert x['owner_rows']==344 and x['inner_rows']==344 and x['all_inner_GPUevents_destroyed']and x['owner_event_validation_passed']and x['inner_event_validation_passed']and x['native_window_closed'];T=x['geometry']['tokens'];assert T==x['geometry']['requests']and(T==32 or 2<=T<8);assert x['actual_counter_delta']['FP32']==[43]*8;assert all(v[10]==43 for v in x['actual_counter_delta']['MoE']);assert set(x['scalar_packets'])=={'owner','inner'}
 for v in x['scalar_packets'].values():assert hashlib.sha256(scalar_host_path(v['path']).read_bytes()).hexdigest()==v['SHA256']
print(json.dumps({'passed':True,'step':q['step'],'scope':'ActualB32+naturalEOS2..7decodeprofile, notcleanperformance; originalmath+policy5; noModelgainclaim','SHA256':hashlib.sha256(p.read_bytes()).hexdigest()}))
