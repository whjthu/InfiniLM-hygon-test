"""Native GPU dropout diagnostic: exact forward/backward/RNG and real input guards."""
import copy,datetime,gc,hashlib,importlib.metadata as metadata,json,os,signal,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
epoch=float(os.environ['DIAGNOSTIC_EPOCH']);deadline=float(os.environ['DIAGNOSTIC_DEADLINE'])
assert epoch<=time.time()<deadline and deadline-epoch==180
signal.alarm(max(1,min(120,int(deadline-time.time()-25))))
import torch
import torch.nn.functional as functional
from dropout_capture import capture_dropout,replay_dropout
torch.set_num_threads(2)
platform=os.environ['EXPECTED_PLATFORM'];assert platform in ('cuda','hip')
if platform=='cuda':
    assert str(torch.__version__)=='2.8.0a0+5228986c39.nv25.06' and torch.version.cuda=='12.9' and torch.version.hip is None
    assert metadata.version('torch')=='2.8.0a0+5228986c39.nv25.6'
else:
    assert str(torch.__version__)=='2.10.0' and torch.version.hip and metadata.version('torch')=='2.10.0+das.opt1.dtk2604.2606161421.g91a580'
assert torch.cuda.device_count()==1
torch.cuda.set_device(0);properties=torch.cuda.get_device_properties(0)
if platform=='cuda':assert uuid.UUID(str(properties.uuid))==uuid.UUID(os.environ['EXPECTED_UUID'].removeprefix('GPU-'))
FLOOR=16*1024**3
assert torch.cuda.mem_get_info()[0]>=FLOOR
torch.cuda.reset_peak_memory_stats()
plan=[{'rows':rows,'width':width,'dtype':dtype,'transpose':False,'zeros':False,'gradients':True}
      for dtype in ('float32','bfloat16') for rows in (247,4096) for width in (1024,2048,3584)]
plan.extend([{'rows':247,'width':1024,'dtype':'float32','transpose':True,'zeros':False,'gradients':True},
             {'rows':247,'width':1024,'dtype':'float32','transpose':False,'zeros':True,'gradients':True},
             {'rows':247,'width':1024,'dtype':'float32','transpose':False,'zeros':True,'gradients':False},
             {'rows':247,'width':1024,'dtype':'bfloat16','transpose':False,'zeros':True,'gradients':False}])
rows=[];calls=0;backwards=0;reference=None;started=time.monotonic()
def sha(value):return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def save(name,value):
    with (ROOT/name).open('x') as f:json.dump(value,f,indent=2)
def progress(failure=None):
    value={'planned_cases':plan,'completed_cases':rows,'completed_denominator':len(rows),'planned_denominator':len(plan),'native_F_dropout_calls':calls,'real_backwards':backwards,'failure':failure}
    tmp=ROOT/'progress.tmp';tmp.write_bytes(json.dumps(value).encode());tmp.replace(ROOT/'progress.json')
def input_tensor(spec):
    dtype=getattr(torch,spec['dtype']);count=spec['rows']*spec['width']
    host=torch.empty(count+512,dtype=dtype);host[:256].fill_(29);host[-256:].fill_(-31)
    payload=host[256:-256]
    if spec['zeros']:payload.zero_()
    else:payload.copy_(((torch.arange(count,dtype=torch.int64)%97)-48).float().div_(128).to(dtype))
    flat=host.to('cuda')
    value=(flat[256:-256].view(1,spec['width'],spec['rows']).transpose(1,2)
           if spec['transpose'] else flat[256:-256].view(1,spec['rows'],spec['width']))
    value.requires_grad_(spec['gradients'])
    return flat,value,host
def check_input(flat,host):assert torch.equal(flat.detach().cpu(),host),'GPU input payload or canary mutated'
def native(value):
    global calls
    calls+=1;return functional.dropout(value,p=.05,training=True,inplace=False)
def backward(value,output,enabled):
    global backwards
    if not enabled:return None
    output.sum().backward();backwards+=1
    grad=value.grad.detach().cpu().clone();assert torch.isfinite(grad).all();return grad
progress()
try:
    for ordinal,spec in enumerate(plan):
        assert time.time()<deadline-25 and torch.cuda.mem_get_info()[0]>=FLOOR
        flat,x,host=input_tensor(spec);source_input=x.detach().cpu().clone() if ordinal==0 else None
        torch.manual_seed(731);baseline=native(x);expected=baseline.detach().cpu().clone();native_rng=torch.cuda.get_rng_state().clone()
        expected_grad=backward(x,baseline,spec['gradients']);check_input(flat,host)
        del baseline,x,flat;gc.collect()
        flat,x,host=input_tensor(spec);torch.manual_seed(731)
        observed,record=capture_dropout(x,.05);calls+=1
        assert torch.equal(observed.detach().cpu(),expected),'capture changed native forward'
        assert torch.equal(torch.cuda.get_rng_state(),native_rng),'capture changed native RNG progression'
        gradient=backward(x,observed,spec['gradients'])
        if spec['gradients']:assert torch.equal(gradient,expected_grad),'capture changed native gradient'
        check_input(flat,host);del observed,x,flat;gc.collect()
        flat,x,host=input_tensor(spec);torch.manual_seed(919);control=native(x);other_rng=torch.cuda.get_rng_state().clone();del control
        torch.manual_seed(919);observed=replay_dropout(x,record);calls+=1
        assert torch.equal(observed.detach().cpu(),expected),'replay output differs from frozen native draw'
        assert torch.equal(torch.cuda.get_rng_state(),other_rng),'replay changed native RNG progression'
        gradient=backward(x,observed,spec['gradients'])
        if spec['gradients']:assert torch.equal(gradient,expected_grad),'replay gradient differs from frozen native draw'
        check_input(flat,host)
        if ordinal==0:
            path=ROOT/'native-GPU-reference.pt'
            with path.open('xb') as f:torch.save({'input':source_input,'output':expected,'gradient':expected_grad,'dropout_record':record,'spec':spec,'source_platform':platform,'note':'Synthetic operator input; actual native GPU mask, forward and gradient. Not model activation or training.'},f)
            h=hashlib.sha256()
            with path.open('rb') as f:
                for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
            reference={'name':path.name,'bytes':path.stat().st_size,'SHA256':h.hexdigest()}
        row={'ordinal':ordinal,'spec':spec,'shape':list(x.shape),'stride':list(x.stride()),'source_operator':record['source_operator'],
             'mask_SHA256':record['mask_SHA256'],'mask_bytes':record['mask'].numel(),'native_forward_SHA256':sha(expected),'native_gradient_SHA256':sha(expected_grad) if expected_grad is not None else None,
             'capture_forward_exact':True,'capture_gradient_exact':bool(spec['gradients']),'capture_RNG_exact':True,'replay_forward_exact':True,'replay_gradient_exact':bool(spec['gradients']),'replay_RNG_exact':True,
             'input_payload_and_guards_unchanged':True,'guard_elements_per_side':256,'guard_bytes_per_side':256*x.element_size()}
        del observed,x,flat,host,expected,expected_grad,gradient,record
        torch.cuda.empty_cache();torch.cuda.synchronize()
        row['free_device_bytes_after_case']=torch.cuda.mem_get_info()[0];assert row['free_device_bytes_after_case']>=FLOOR
        rows.append(row);progress()
    value=torch.ones(4,7,device='cuda');torch.manual_seed(731);_,record=capture_dropout(value,.05);calls+=1
    bad=[]
    for kind in ('missing','mutated','dtype','short','provider','shape','stride','probability'):
        altered=copy.deepcopy(record)
        if kind=='missing':altered.pop('source_operator')
        if kind=='mutated':altered['mask'][0,0].logical_not_()
        if kind=='dtype':altered['mask']=altered['mask'].float()
        if kind=='short':altered['mask']=altered['mask'][:,:2]
        if kind=='provider':altered['source_operator']='synthetic-mask'
        if kind=='shape':altered['shape']=[7,4]
        if kind=='stride':altered['stride']=[1,4]
        if kind=='probability':altered['probability']=float('nan')
        rng=torch.cuda.get_rng_state().clone()
        try:replay_dropout(value,altered)
        except ValueError:pass
        else:raise AssertionError('invalid frozen mask accepted: '+kind)
        assert torch.equal(rng,torch.cuda.get_rng_state()),'rejection consumed native RNG'
        bad.append(kind)
    torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
    assert peak<2*1024**3
    result={'state':'NATIVE_GPU_DROPOUT_CAPTURE_REPLAY_CONTRACTS_PASS','claim_commit':os.environ['DIAGNOSTIC_CLAIM'],'platform':platform,'torch':str(torch.__version__),'torch_distribution':metadata.version('torch'),'CUDA':torch.version.cuda,'HIP':torch.version.hip,
            'device_name':properties.name,'device_uuid':str(properties.uuid),'device_architecture':getattr(properties,'gcnArchName',None),'cases':rows,'completed_cases':len(rows),'planned_cases':len(plan),'bad_records_rejected':bad,
            'native_F_dropout_calls':calls,'real_backwards':backwards,'strict_free_floor_bytes':FLOOR,'peak_allocated_bytes':peak,'peak_reserved_bytes':reserved,'wall_seconds':time.monotonic()-started,'native_reference':reference,
            'model_loaded':False,'training_qualified':False,'formal_performance':False,'input_kind':'Controlled synthetic operator inputs with real native GPU RNG draws; no synthetic weights or model activation claim.'}
    save('result.json',result);print(json.dumps({k:v for k,v in result.items() if k!='cases'}))
except BaseException as error:
    progress({'error_type':type(error).__name__,'message':str(error)[:300]})
    save('failure.json',{'error_type':type(error).__name__,'message':str(error)[:300],'completed_cases':len(rows),'planned_cases':len(plan),'native_F_dropout_calls':calls,'real_backwards':backwards})
    raise
