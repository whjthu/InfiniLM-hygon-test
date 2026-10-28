import json,pathlib,sys,copy
from deep_owner_consumer_r168a import validate_inner,native_expected_kinds
def main():
 kinds=[0,0]+[1]*20+[2]*21
 rows=[]
 for d in range(8):
  for l in range(43):rows.append(dict(device=d,layer=l,kind=kinds[l],tokens=1,requests=1,owner=1,thread=1,events_created=7,events_queried=7,events_destroyed=7,create_during_device=d,query_during_device=d,destroy_during_device=d,whole_stream_ms=6.,original_final_drain_host_ms=0.,segments=[dict(segment=i,start_ms=float(i),end_ms=float(i+1),stream_ms=1.,host_wall_ms=1.,thread_cpu_ms=.2)for i in range(6)]))
 packet=dict(schema=1,profile='attention-inner-r9193',GPU_handles_retained=0,rows=rows,scope='CPU_SCHEMA_ONLY_NO_GPU_TIMING')
 v=validate_inner(json.dumps(packet),kinds,1,1);assert v['owner_rows']==344
 for field in ['tokens','requests','kind','events_destroyed']:
  bad=copy.deepcopy(packet);bad['rows'][0][field]=99
  try:validate_inner(json.dumps(bad),kinds,1,1)
  except AssertionError:pass
  else:raise AssertionError('badshape accepted '+field)
 text=pathlib.Path(__file__).with_name('native-Frame-support-r9193.hpp').read_text();assert 'if(!tokens||!count)'in text and 'tokens<2'not in text
 assert 'torch'not in sys.modules
 return dict(passed=True,CPU_only=True,GPU=False,Native_loaded=False,actual_T1_geometry=False,schema_mock_T1_344rows=True,native_source_nonzero_geometry=True,malformed_fields_rejected=4,main_attention_layers=43,expected_FP32_batch_attention_calls_at_C1=0,actual_GPU_capture_required=True)
if __name__=='__main__':print(json.dumps(main()))
