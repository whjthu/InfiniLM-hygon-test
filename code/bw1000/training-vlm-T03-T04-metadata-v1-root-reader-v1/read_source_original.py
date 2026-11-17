import pathlib,hashlib,json,os,time
R=pathlib.Path('/home/whj/nvidia-baseline/alignment-assets/training-vlm-T03-T04-metadata-v1')
records={}
for n in ('intent.json','outcome.json'):
 p=R/n
 assert p.is_file() and not p.is_symlink() and p.stat().st_size<=262144
 b=p.read_bytes();v=json.loads(b)
 assert v['request_id']=='bc17d9a5727f48f58240c5bdd1af9d39' and v['publication']=='3155d5b88cbafdad6f349a8743c9ef6d3c08536c'
 records[n]=dict(SHA256=hashlib.sha256(b).hexdigest(),bytes=len(b),value=v)
v=records['outcome.json']['value'];i=records['intent.json']['value']
assert v['passed'] is True and v['bundle_SHA256']=='fa6c2447da866d1b8ca3dcd2f6ae6fa935fdd27084e73d31bd9c7d4483deb599' and v['bundle_bytes']==7937372
assert v['source_bytes_read']==32327445 and v['member_count']==36 and v['source_files_read']==67
assert v['weight_payload_bytes_read']==v['initial_tensor_bytes_read']==v['actual_GPU_cards']==0
p=pathlib.Path('/proc')/str(i['pid'])/'stat'
if not p.exists():current=dict(original_PID_absent=True,original_owned_process_live=False)
else:
 fields=p.read_text().rsplit(')',1)[1].split();ticks=int(fields[19]);current=dict(original_PID_absent=False,startticks_match=ticks==i['startticks'],original_owned_process_live=ticks==i['startticks'] and fields[0]!='Z')
assert current['original_owned_process_live'] is False
summary={k:v.get(k) for k in ('passed','request_id','publication','epoch','deadline','pid','startticks','uid','boot_id','source_manifest_SHA256','metadata_only','full_weight_identity','member_count','source_files_read','source_bytes_read','bundle_bytes','bundle_SHA256','weight_payload_bytes_read','initial_tensor_bytes_read','actual_GPU_cards','CPU_affinity','nice','RLIMIT_AS_bytes')}
print(json.dumps(dict(observed_epoch=time.time(),original_records={n:{k:x[k] for k in ('SHA256','bytes')} for n,x in records.items()},source=summary,source_process=current,raw_model_or_original32JSON_re_read=False,new_probe_or_Slurm=False)))