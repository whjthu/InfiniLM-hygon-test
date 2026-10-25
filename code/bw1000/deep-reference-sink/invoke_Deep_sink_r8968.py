import pathlib,json,hashlib,runpy
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';mp=r/'incoming/Deep-reference-sink-manifest-r8968.json';assert hashlib.sha256(mp.read_bytes()).hexdigest()=='433def70a11f124c9025e87a44008f59bf79e0464182768e10ebe982c0956c74';m=json.loads(mp.read_bytes())
for group,base in [('incoming_source_SHA256',r/'incoming'),('existing_remote_SHA256',r)]:
 for n,h in m[group].items():assert hashlib.sha256((base/n).read_bytes()).hexdigest()==h,n
bp=r/m['cpu_receipt_path'];q=json.loads(bp.read_bytes());assert q['completed']and q['CPU_only']and q['libraries']['full']['SHA256']==m['DSO_SHA256'];assert q['actual_Model_consumer_required_symbols']['passed']
runpy.run_path(str(r/'incoming/launch_Deep_reference_sink_component_r8968.py'),run_name='__main__')
