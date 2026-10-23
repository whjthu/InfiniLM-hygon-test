import ast,hashlib,json,pathlib,tempfile
import numpy as np
from kimi_native_state_consumer_r8519 import read_snapshot,read_KDA_state_phase,target_intermediate,array_summary,gather_valid_KV
from kimi_native_causal_protocol_r8518 import compare_boundaries,select_stage,select_layer
from online_kimi_provider_r8517 import cpu_argmax_observation
from kimi_snapshot_budget_r8517 import snapshot_bound

r=pathlib.Path.cwd();checks=[]
with tempfile.TemporaryDirectory(prefix='Kimi-native-CPU-r8522-',dir=r)as tmp:
    root=pathlib.Path(tmp);rows=[]
    def store(point,value,**extra):
        name=point+'.bin';(root/name).write_bytes(value.tobytes())
        rec=dict(point=point,file=name,dtype={np.dtype('<i4'):'i32',np.dtype('<i8'):'i64',np.dtype('<u2'):'bf16'}[value.dtype],
            shape=list(value.shape),bytes=value.nbytes,source_layout=dict(shape=list(value.shape),strides=[s//value.itemsize for s in value.strides],**extra))
        rows.append(rec);return rec
    store('read_rows',np.array([0,7],np.int32));store('write_rows',np.array([3,9],np.int64))
    for phase,physical in [('before',7),('after',9)]:
        for kind,shape in [('conv',(1,6,3)),('ssm',(1,2,4,4))]:
            a=np.full(shape,physical,np.uint16);stride=int(np.prod(shape[1:]))
            store(kind+'_'+phase+'_row_'+str(physical),a,pool_shape=[33,*shape[1:]],pool_strides=[stride,*[s//2 for s in a.strides[1:]]],pool_row=physical,pool_row_offset_elements=physical*stride)
    for phase,physical in [('before',7),('after',9)]:
        state=read_KDA_state_phase(root,rows,phase,1)
        assert state['actual_state_row']==physical and all((a==physical).all()for a in state['tensors'].values())
    checks.append('Actual I32 read/I64 write metadata chooses7/9, not packed row1 or job physicalrow')
    raw=np.arange(1*9*3,dtype=np.uint16).reshape(1,9,3)
    assert np.array_equal(target_intermediate(raw,'input',4,7,9),raw[:,4:7])
    norm=np.arange(9*12*4,dtype=np.uint16).reshape(9*12,4)
    assert target_intermediate(norm,'normalized_f32',4,7,9).shape==(3,12,4)
    checks.append('Packed offsets and normalized-head flattening preserved')
    pages={2:np.zeros((2,1,256,1,1),np.uint16),9:np.zeros((2,1,256,1,1),np.uint16)}
    a=gather_valid_KV(pages,[2,9,0,0],257);pages[9][:,:,1:]=0xffff
    b=gather_valid_KV(pages,[2,9,0,0],257);assert array_summary(a)==array_summary(b)
    pages[9][:,:,0]=1;c=gather_valid_KV(pages,[2,9,0,0],257);assert array_summary(b)!=array_summary(c)
    checks.append('Unused KV page tail ignored; first valid changed element detected')
    logits=np.zeros((2,163840),np.uint16);logits[0,7]=0x3f80;logits[1,9]=0x4000
    v=cpu_argmax_observation(logits,[7,9]);w=cpu_argmax_observation(logits[:1],[7])
    assert v['native_equals_CPU']and v['per_request_BF16_logits_SHA256'][0]==w['per_request_BF16_logits_SHA256'][0]
    checks.append('Perrequest full-BF16 logit hash remains comparable across batch geometry')
    def packet(value,stage,layer):
        return [dict(jobs=[dict(kind='decode',boundaries=[dict(stage=stage,layer=layer,point='decoder',rank=0,hidden=3,packed_target_only=True,requests=[dict(fnv1a64=value,tokens=1,nonfinite=0)])])])]
    diffs=compare_boundaries(packet('a',2,48),packet('b',2,48));assert select_stage(diffs)==2 and select_layer(diffs)==48
    checks.append('Actual consumer comparison selects stage2 layer48 from parsed boundary contract')
    cuts=json.loads((r/'Kimi-postprefill-firstdecode-cuts-r8507.json').read_text())['cuts']
    bound=[]
    for cut in cuts:
        for record in cut['jobs']:
            if 0 in record['job']['keys']:
                bound.append(snapshot_bound(record['job'],state_rows='all',observe_kv=True))
    assert max(v['KDA_raw_bytes_upper']for v in bound)<512*1024**2
    assert max(v['MLA_raw_bytes_upper']for v in bound)<512*1024**2
    checks.append('Actual O5/W4 immutable target-bearing jobs bounded below512MiB raw clones/owner/selectedlayer')
sources=['kimi_firstdecode_consumer_r8517.py','online_kimi_provider_r8517.py','kimi_native_state_consumer_r8519.py','kimi_native_causal_protocol_r8518.py','kimi_snapshot_budget_r8517.py']
for name in sources:ast.parse((r/name).read_text())
out=dict(revision=8525,CPU_actual=True,GPU_executed=False,contracts=checks,
    source_sha256={name:hashlib.sha256((r/name).read_bytes()).hexdigest()for name in sources},
    limits='CPU fixtures do not prove GPU native hooks/state equality or final-model correctness')
p=r/'Kimi-native-consumer-CPU-r8525.json';assert not p.exists();p.write_text(json.dumps(out,indent=2));print(json.dumps(out))
