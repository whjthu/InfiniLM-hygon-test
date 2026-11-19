import hashlib,json,pathlib,datetime
B=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')
runs=('training-qwen-first-resumed-forward-v1','training-qwen-resumed-decoder-internals-v1')
out={}
for run in runs:
 top=B/run
 raw=(top/'native-outcome.json').read_bytes()
 assert 0<len(raw)<=1048576
 d=json.loads(raw)
 assert d['normal_native_exit'] and d['result']['all_input_fingerprints_equal']
 ck=d['result']['checkpoint20']
 out[run]=dict(outcome_hash=hashlib.sha256(raw).hexdigest(),checkpoint20_pins={k:ck[k] for k in ('adapter_model.safetensors','optimizer.pt','scheduler.pt',*[f'rng_state_{i}.pth' for i in range(8)])})
 if run==runs[0]:
  path=top/'continuous/capture/rank-02/receipt.json'
  raw=path.read_bytes()
  assert 0<len(raw)<=1048576
  z=json.loads(raw)
  out['first_capture_receipt_shape']={k:(list(v)[:22] if isinstance(v,dict) else [len(v),list(v[0])[:18] if v and isinstance(v[0],dict) else []] if isinstance(v,list) else type(v).__name__) for k,v in z.items()}
a,b=(out[r]['checkpoint20_pins'] for r in runs)
print(json.dumps(dict(at_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),metadata=out,checkpoint20_identity_equal={k:a[k]==b[k] for k in a})))
