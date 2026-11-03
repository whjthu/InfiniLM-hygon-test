"""Exact two-node component FFI and retained-input leaf; no loading on import."""
from pathlib import Path,PurePosixPath
import argparse,ctypes,hashlib,json,math,struct,sys

def load_input(root,card):
 root=Path(root);rows=[];refs=[]
 for v in card['files']:
  path=root/Path(v['path']).relative_to('/refactor');raw=path.read_bytes()
  assert len(raw)==v['bytes']and hashlib.sha256(raw).hexdigest()==v['SHA256'];rows.append(raw);refs.append(dict(path=str(path),SHA256=v['SHA256'],bytes=len(raw)))
 assert len(rows)==2 and rows[0]==rows[1]and len(rows[0])==16*7168*2
 row=rows[0][:7168*2];assert sys.byteorder=='little'
 values=[struct.unpack('<f',struct.pack('<I',v<<16))[0]for(v,)in struct.iter_unpack('<H',row)]
 assert all(math.isfinite(v)for v in values)
 return row,dict(parent_files=refs,selected_row=0,parent_shape=[16,7168],selected_shape=[1,1,7168],dtype='BF16',byteorder='little',bytes=len(row),SHA256=hashlib.sha256(row).hexdigest(),historical_teacher=True,current_32card_model_activation=False)

def payload(config,weights,input_pointer,raw_directory,node_rank,master_addr,master_port):
 assert type(node_rank)is int and node_rank in[0,1]and type(master_port)is int and 1024<master_port<65536
 assert type(input_pointer)is int and input_pointer>0 and PurePosixPath(raw_directory).is_absolute()
 assert isinstance(config,dict)and not config.get('infinilm_kimi_long_agent_r179',False)
 seen=set()
 for v in weights:
  assert isinstance(v['name'],str)and v['name']and v['name']not in seen;seen.add(v['name'])
  assert type(v['pointer'])is int and v['pointer']>0 and v['dtype']in[0,1,2]
  assert isinstance(v['shape'],list)and v['shape']and all(type(x)is int and x>0 for x in v['shape'])
 q=dict(config=config,weights=weights,input_pointer=input_pointer,raw_directory=str(raw_directory),node_rank=node_rank,master_addr=master_addr,master_port=master_port,layer=1)
 return json.dumps(q,separators=(',',':')).encode()

def bind(library):
 run=library.modern_kimi_lowrank_full_moe_r437;run.argtypes=[ctypes.c_char_p];run.restype=ctypes.c_int
 result=library.modern_kimi_lowrank_full_moe_result_r437;result.argtypes=[];result.restype=ctypes.c_char_p
 return run,result

def contract():
 # Only the CABI leaf is recorded; real payload construction + identity is shared.
 class Function:
  def __init__(self,f):self.f=f
  def __call__(self,*a):return self.f(*a)
 class Library:pass
 library=Library();seen=[]
 library.modern_kimi_lowrank_full_moe_r437=Function(lambda raw:seen.append(json.loads(raw))or 0)
 library.modern_kimi_lowrank_full_moe_result_r437=Function(lambda:b'{"completed":true,"recorded_CABI":true}')
 run,result=bind(library)
 for node in[0,1]:
  raw=payload({'model_type':'recording-only-config'},[dict(name='recorded.weight',pointer=1,dtype=1,shape=[1])],2,'/recording-only-output',node,'127.0.0.1',40917)
  assert run(raw)==0 and json.loads(result())['recorded_CABI']
 assert [x['node_rank']for x in seen]==[0,1]and all(v['layer']==1 for v in seen)
 assert run.argtypes==[ctypes.c_char_p]and run.restype==ctypes.c_int and result.argtypes==[]and result.restype==ctypes.c_char_p
 return dict(passed=True,actual_shared_payload_and_FFI_handoff=2,recording_leaf='Two CABI functions only; no Native execution/Runtime/weights',GPU=0,Native_dlopen=False)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--check-contract',action='store_true');p.add_argument('--fixture-root');a=p.parse_args();q=contract()
 if a.fixture_root:
  card=json.loads(Path(__file__).with_name('FIXTURE_CARD.json').read_bytes());_,info=load_input(a.fixture_root,card);q['real_retained_C1_input']=info
 print(json.dumps(q))
