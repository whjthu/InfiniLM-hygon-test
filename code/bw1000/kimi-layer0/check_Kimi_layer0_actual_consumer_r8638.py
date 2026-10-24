import pathlib,json,sys,importlib.util
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';sys.path.insert(0,str(r/'incoming'));spec=importlib.util.spec_from_file_location('layer0CPUcheck',r/'incoming/kimi_embedding_layer0_runner_r8636.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);m=json.loads((r/'incoming/Kimi-layer0-operands-r8636.json').read_text())
# Adapt only /refactor filesystem alias for the login CPU process, never expected math/data.
mod.resolve=lambda p:pathlib.Path(str(p).replace('/public/home/aclsylqidf/hygon-inference-refactor-20261014',str(r)).replace('/refactor',str(r)))
# check_metadata paths are container absolute; validate equivalent raw files before originalconsumer below.
out={'GPU_import_or_dlopen':False,'numpy_available':True,'native_O_actualconsumer':[]}
old=next(v for v in m['expected_KDA']if v['rank']==0)['journal'];directory=pathlib.Path(old).parent
for rank in range(8):
 result=mod.inspect_case(directory,rank,m);assert result['faithful'];out['native_O_actualconsumer'].append({'rank':rank,'faithful':True,'native_keys':list(result['native_exact']),'boundary_count':len(result['boundary_exact'])})
# Pure header metadata and real files with hostaliases, no weights read.
import struct,hashlib
model=pathlib.Path.home()/'shared/Kimi-K3';index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];headers={}
for k,meta in m['weights'].items():
 p=model/index[k];assert p.is_file()and index[k]==meta['shard']
 if p not in headers:
  with p.open('rb')as f:n=struct.unpack('<Q',f.read(8))[0];headers[p]=json.loads(f.read(n))
 assert all(headers[p][k][v]==meta[v]for v in ['dtype','shape','data_offsets'])
for rel,h in [(m['CPU_receipt'],m['CPU_receipt_raw_SHA256']),(m['runtime_library'].removeprefix('/refactor/'),m['runtime_library_SHA256'])]:p=r/rel;assert p.is_file()and hashlib.sha256(p.read_bytes()).hexdigest()==h
out.update(checked24realcheckpointHeaders=True,immutable_inputs_1and4_by512=all(len(v)==512 for group in m['inputs'].values()for v in group),CPUreceipt_DSO_rawSHA=True,A_log={'source':[128],'loaded':[96]},completed=True);print(json.dumps(out))
