import pathlib,json,hashlib
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');source=R/'results/Deep353r1-1261-current-C1-profile-qualified-summary.json';q=json.loads(source.read_bytes());labels=['prepare_projection_norm_rope','cache_compression_index_metadata','latent_attention','inverse_output_rope','output_projection_collective','commit_cache'];out=[]
for request in q['profiles']:
 for v in request['profiles']:
  p=R/pathlib.Path(v['scalar_packets']['inner']['path']).relative_to('/refactor');assert hashlib.sha256(p.read_bytes()).hexdigest()==v['scalar_packets']['inner']['SHA256'];raw=json.loads(p.read_bytes());kinds={str(k):{name:[0.]*8 for name in labels}for k in [0,1,2]}
  for row in raw['rows']:
   for seg in row['segments']:kinds[str(row['kind'])][labels[seg['segment']]][row['device']]+=seg['stream_ms']
  out.append(dict(request_index=request['request_index'],decode_call_1based=v['geometry']['decode_call_1based'],kinds=kinds))
print(json.dumps(dict(actual_GPU='910374.1261',source_compact_summary_path=str(source),source_compact_summary_SHA256=hashlib.sha256(source.read_bytes()).hexdigest(),raw_packets_exported=False,scope='Perkind segment sum across sameowner sequential layers; 0SWA2/1HCA20/2CSA21. No sum across owners or nested ranges. Instrumented only, not isolatedkernel time.',samples=out)))
