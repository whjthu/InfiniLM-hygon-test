"""New fullcheckpoint source descriptors and lifetime tests; no Native/SDK."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import weakref
D=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('checkpoint',D/'checkpoint.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
class Tensor:
 def __init__(self,dtype='bf16'):self.dtype=dtype;self.shape=[2,2]
 def contiguous(self):return self
 def view(self,dtype):self.dtype=dtype;return self
 def data_ptr(self):return 2**40+id(self)%1000000
 def numel(self):return 4
 def element_size(self):return 2 if self.dtype=='bf16'else 1
class Torch:float32='f32';bfloat16='bf16';uint8='u8';float8_e4m3fn='fp8'
class Shard:
 def __init__(self,refs):self.refs=refs
 def __enter__(self):return self
 def __exit__(self,*a):pass
 def get_tensor(self,name):
  t=Tensor('fp8'if name.endswith('scale')else'bf16');self.refs.append(weakref.ref(t));return t
class Handle:
 rank=0
 def __init__(self,keys):self.names=keys;self.loaded=0;self.done=False;self.calls=0
 def keys(self):return self.names
 def load_batch(self,records):
  self.calls+=1
  for r in records:
   assert r['owner']is not None and r['pointer']>2**32 and r['shape']==[2,2]and r['dtype']in [0,1,2]
  self.loaded+=len(records)
 def finish(self):self.done=True
 def geometry(self):return dict(weights_loaded=self.done,Model_READY=self.done,loaded_parameters=self.loaded)
class CheckpointTest(unittest.TestCase):
 def test_complete78layers_sources_survive_typed_return(self):
  with tempfile.TemporaryDirectory()as tmp:
   keys=['model.layers.%d.weight'%i for i in range(78)]+['model.embed_tokens.weight','model.norm.weight','lm_head.weight','model.layers.0.scale']
   path=Path(tmp);(path/'model.safetensors.index.json').write_text(json.dumps(dict(weight_map={k:'part-'+str(i%2)for i,k in enumerate(keys)})))
   refs=[];h=Handle(keys);report={};events=[];mem=[]
   m.load_checkpoint(h,model_path=path,torch=Torch,safe_open=lambda*a,**kw:Shard(refs),report=report,deadline=lambda x:None,memory=mem.append,mark=lambda x,**kw:events.append(x),progress=lambda*a,**kw:None)
   self.assertEqual(h.loaded,82);self.assertEqual(h.calls,2);self.assertTrue(report['all78layers_loaded']);self.assertEqual(report['actual_layers'],list(range(78)));self.assertEqual(len(mem),3);self.assertTrue(all(r()is None for r in refs))
 def test_partial_layer_fixture_rejected_before_load(self):
  with tempfile.TemporaryDirectory()as tmp:
   p=Path(tmp);(p/'model.safetensors.index.json').write_text(json.dumps(dict(weight_map={'model.layers.0.weight':'part'})))
   with self.assertRaises(AssertionError):m.load_checkpoint(Handle(['model.layers.0.weight']),model_path=p,torch=Torch,safe_open=None,report={},deadline=None,memory=None,mark=None)
if __name__=='__main__':unittest.main(verbosity=2)
