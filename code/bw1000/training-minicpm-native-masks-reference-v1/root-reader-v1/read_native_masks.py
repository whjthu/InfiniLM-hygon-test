"""Independently validate every actual native model mask packet on CPU only."""
import hashlib,json,os,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
signal.alarm(600)
assert os.environ['CUDA_VISIBLE_DEVICES']==os.environ['HIP_VISIBLE_DEVICES']==os.environ['ROCR_VISIBLE_DEVICES']=='-1'
import torch
assert not torch.cuda.is_initialized();torch.set_num_threads(2)
sys.path.insert(0,str(ROOT/'framework'))
from vlm.runtime.lora_dropout_session import input_identity

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
 return h.hexdigest()

def require(value,message):
 if not value:raise ValueError(message)

def load(path):return torch.load(path,map_location='cpu',weights_only=True)
started=time.monotonic();outcome=json.loads((ROOT/'outcome.json').read_bytes())
require(outcome['native_returncode']==0 and outcome['keeper_passed'],'native/keeper terminal not qualified')
require(outcome['mask_rank_records']==outcome['capture_rank_records']==outcome['native25_rank_records']==8,'missing original rank terminals')
config=json.loads((ROOT/'src/MASK_CONFIG.json').read_bytes());rank_rows=[];all_files=[];total_mask_bytes=0;total_calls=0
for rank in range(8):
 maskroot=ROOT/'masks'/f'rank-{rank:02d}';capture=ROOT/'capture'/f'rank-{rank:02d}'
 complete=maskroot/'complete.json';raw=complete.read_bytes();summary=json.loads(raw);contract=summary['contract']
 require(summary['mode']=='capture' and len(summary['files'])==8,'missing forward packets')
 require(contract['rank']==rank and contract['world_size']==8 and contract['planned_forwards']==8 and contract['accumulation']==4,'rank or forward contract differs')
 require(contract['binding']=={k:config[k] for k in ('case','recipe_SHA256','initial_trainable_SHA256','expected_dropout_modules')},'model/recipe contract differs')
 initial=load(capture/'initial.pt');parameters=initial['parameters'];widths={n.replace('.lora_A.default.weight','.lora_dropout.default'):p.shape[-1] for n,p in parameters.items() if '.lora_A.default.weight' in n};del initial,parameters
 inventory=contract['module_inventory'];require(len(inventory)==len(widths)==186 and {x['name'] for x in inventory}==set(widths),'actual model module inventory differs')
 require(all(x['p']==.05 and x['inplace'] is False for x in inventory),'dropout recipe changed')
 progress=json.loads((maskroot/'progress.json').read_bytes());require(progress['completed_forwards']==progress['planned_forwards']==8 and progress['failure'] is None,'partial/failed packet progress')
 rank_files=[];rank_mask_bytes=0;rank_calls=0
 for ordinal,info in enumerate(summary['files'],1):
  require(info['name']==f'forward-{ordinal:04d}-masks.pt' and info['micro_ordinal']==ordinal and info['global_step']==(ordinal-1)//4,'forward ordering differs')
  path=maskroot/info['name'];require(path.is_file() and not path.is_symlink() and path.stat().st_size==info['bytes'] and sha(path)==info['SHA256'],'actual packet SHA differs')
  packet=load(path);require(packet['contract']==contract and len(packet['records'])==186,'packet module contract differs')
  inp=load(capture/f'forward-{ordinal:02d}-input.pt');expected={'global_step':(ordinal-1)//4,'micro_ordinal':ordinal,'input_SHA256':input_identity(inp['args'],inp['kwargs'])};del inp
  require(packet['identity']==expected and info['input_SHA256']==expected['input_SHA256'],'complete actual model input differs')
  names=set();identities=[]
  for index,row in enumerate(packet['records']):
   name=row['module'];record=row['native_record'];mask=record['mask'];require(row['call_ordinal']==index and name in widths and name not in names,'duplicate/missing/foreign module or call order')
   require(record['probability']==.05 and record['source_operator']=='aten::native_dropout' and record['source_device']=='cuda:'+str(rank),'not the actual native A100 sampler/device')
   require(mask.device.type=='cpu' and mask.dtype==torch.bool and mask.is_contiguous() and not mask.requires_grad and list(mask.shape)==record['shape'],'incomplete mask tensor')
   require(record['shape'][-1]==widths[name] and record['dtype'] in ('torch.float32','torch.bfloat16'),'actual LoRA input shape/dtype differs')
   require(len(record['shape'])==len(record['stride']) and all(x>0 for x in record['shape']),'invalid actual input metadata')
   require(hashlib.sha256(mask.view(torch.uint8).numpy().tobytes()).hexdigest()==record['mask_SHA256'],'full mask bytes differ')
   identities.append({'module':name,'call_ordinal':index,'mask_SHA256':record['mask_SHA256']});names.add(name);rank_mask_bytes+=mask.numel();rank_calls+=1
  require(names==set(widths),'missing actual LoRA module')
  pin=hashlib.sha256(json.dumps(identities,sort_keys=True,separators=(',',':')).encode()).hexdigest();require(pin==info['mask_identity_SHA256'],'ordered mask identity differs')
  row={'rank':rank,**info};rank_files.append(row);all_files.append(row);del packet
 require(rank_mask_bytes==summary['mask_bytes']==progress['mask_bytes'] and rank_calls==summary['native_dropout_calls']==progress['native_dropout_calls']==1488,'mask byte/call accounting differs')
 require(rank_mask_bytes<=1024**3,'per-rank mask budget exceeded');total_mask_bytes+=rank_mask_bytes;total_calls+=rank_calls
 rank_rows.append({'rank':rank,'manifest_SHA256':hashlib.sha256(raw).hexdigest(),'manifest_bytes':len(raw),'mask_bytes':rank_mask_bytes,'native_calls':rank_calls,'packets':rank_files})
require(total_calls==11904 and len(all_files)==64,'full planned mask denominator differs');require(not torch.cuda.is_initialized(),'reader initialized GPU')
value={'state':'ALL_EIGHT_RANK_NATIVE_MODEL_MASK_PACKETS_INDEPENDENTLY_VERIFIED','claim_commit':outcome['claim_commit'],'framework_commit':outcome['framework_commit'],'outcome_SHA256':sha(ROOT/'outcome.json'),'rank_count':8,'forward_packets':64,'native_dropout_calls':total_calls,'raw_mask_bytes':total_mask_bytes,'ranks':rank_rows,'wall_seconds':time.monotonic()-started,'GPU_initialized':False,'model_numerical_acceptance':False,'cross_platform_qualified':False,'formal_performance':False}
with (ROOT/'independent-native-mask-reference.json').open('x') as f:json.dump(value,f,indent=2)
print(json.dumps({k:v for k,v in value.items() if k!='ranks'}))
