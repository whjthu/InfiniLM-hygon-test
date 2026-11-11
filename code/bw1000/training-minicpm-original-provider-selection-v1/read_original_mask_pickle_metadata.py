from pathlib import Path
import collections,hashlib,io,json,pickle,time,zipfile
p=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-minicpm-native-masks-reference-v1/masks/rank-00/forward-0001-masks.pt');start=time.monotonic();assert p.stat().st_size==PIN['bytes'];h=hashlib.sha256()
with p.open('rb') as f:
 for raw in iter(lambda:f.read(1024**2),b''):
  assert time.monotonic()-start<20;h.update(raw)
assert h.hexdigest()==PIN['SHA256']
class MetadataUnpickler(pickle.Unpickler):
 def find_class(self,module,name):
  if (module,name)==('collections','OrderedDict'):return collections.OrderedDict
  if (module,name)==('torch','BoolStorage'):return 'torch.BoolStorage'
  if (module,name)==('torch._utils','_rebuild_tensor_v2'):return self.tensor
  raise ValueError('unapproved pickle global '+module+'.'+name)
 def persistent_load(self,pid):
  assert isinstance(pid,tuple) and len(pid)==5 and pid[0]=='storage' and pid[1]=='torch.BoolStorage';return dict(storage_dtype=pid[1],storage_elements=pid[4],storage_key=pid[2])
 @staticmethod
 def tensor(storage,offset,size,stride,requires_grad,hooks):return dict(**storage,storage_offset=offset,shape=list(size),stride=list(stride),requires_grad=requires_grad)
with zipfile.ZipFile(p) as z:
 keys=[n for n in z.namelist() if n.endswith('/data.pkl')];assert len(keys)==1 and z.getinfo(keys[0]).file_size<=1024**2
 raw=z.read(keys[0]);v=MetadataUnpickler(io.BytesIO(raw)).load()
records=v['records'];assert len(records)==186
cases={};providers=collections.Counter()
for x in records:
 r=x['native_record'];assert r['mask']['shape']==r['shape'] and r['mask']['storage_dtype']=='torch.BoolStorage';key=json.dumps([r['shape'],r['stride'],r['dtype'],r['source_operator']],separators=(',',':'));providers[r['source_operator']]+=1
 if key not in cases:cases[key]=dict(shape=r['shape'],stride=r['stride'],dtype=r['dtype'],provider=r['source_operator'],source_device=r['source_device'],calls=0,first_module=x['module'])
 cases[key]['calls']+=1
print(json.dumps(dict(scope='Original one-packet mask metadata only, no Torch import or tensor storage deserialization',passed=True,packet_bytes=p.stat().st_size,packet_SHA256=h.hexdigest(),pickle_bytes=len(raw),pickle_SHA256=hashlib.sha256(raw).hexdigest(),records=len(records),providers=dict(providers),cases=list(cases.values()),CPU_seconds=time.monotonic()-start,GPU_cards=0,raw_tensor_transfer_bytes=0,torch_imported=False,tensor_storage_deserialized=False,Model_GO=False)))
