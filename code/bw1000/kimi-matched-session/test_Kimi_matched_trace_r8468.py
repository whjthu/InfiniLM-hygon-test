"""CPU contracts only: no Model, GPU or performance fixture."""
import tempfile,pathlib,copy,json
import numpy as np
from online_kimi_provider_r8467 import cpu_argmax_observation
from kimi_matched_trace_r8467 import digest,seal_reference,read_reference,replay,validate_records


def reject(fn):
    try:fn()
    except AssertionError:return
    raise AssertionError('invalid contract accepted')


logits=np.zeros((2,163840),dtype=np.uint16)
logits[0,4]=logits[0,5]=0x3f80;logits[1,9]=0x4000
o=cpu_argmax_observation(logits,[4,9]);assert o['native_equals_CPU'] and o['top2'][0]['tie_count']==2 and o['top2'][0]['margin']==0
assert not cpu_argmax_observation(logits,[5,9])['native_equals_CPU']
logits[0,0]=0x7fc0;assert not cpu_argmax_observation(logits,[4,9])['finite']
header=dict(scope='MATCHED_SHAPE_JOBTRACE_ONLY',rows=32,reference_flags=67,candidate_flags=71,eos_token_ids=[2],output_limits={'0':3,'1':3},identity={'DSO':'fixed-fixture'})
jobs=[dict(seq=0,kind='prefill',keys=[0,1],rows=[1,8],pasts=[0,0],inputs=[[10,11],[12]]),
      dict(seq=1,kind='decode',keys=[0,1],rows=[1,8],pasts=[2,1],inputs=[[20],[21]]),
      dict(seq=2,kind='decode',keys=[0],rows=[1],pasts=[3],inputs=[[22]])]
ids=[[20,21],[22,2],[23]]
records=[dict(job=j,job_SHA256=digest(j),reference_ids=out,oracle=dict(finite=True,native_equals_CPU=True,cpu_argmax_ids=out,native_ids=out)) for j,out in zip(jobs,ids)]
assert validate_records(header,records)=={0:[20,22,23],1:[21,2]}
bad=copy.deepcopy(records);bad[2]['job']['inputs']=[[999]];bad[2]['job_SHA256']=digest(bad[2]['job']);reject(lambda:validate_records(header,bad))
reject(lambda:validate_records(header,records[:-1]))
class Adapter:
    def __init__(self,wrong=False):self.i=0;self.wrong=wrong
    def forward(self,j):
        out=list(ids[self.i]);self.i+=1
        if self.wrong and self.i==2:out[0]=99
        return out
def collect(label,v):return [dict(job_SHA256=v['job_SHA256'],ids=None)]*3+[v]
with tempfile.TemporaryDirectory()as folder:
    p=pathlib.Path(folder)/'trace.json';h=seal_reference(p,header,records)
    frozen=read_reference(p,h,header['identity']);saved=[]
    assert replay(frozen,Adapter(),collect,saved.append)['qualified']
    assert saved[-1]['actual_outputs']=={0:[20,22,23],1:[21,2]}
    a=Adapter(True);saved=[];reject(lambda:replay(frozen,a,collect,saved.append))
    assert a.i==2 and saved[-1]['firstdiff']['actual_ids']==[99,2]
    assert saved[-1]['firstdiff']['qualification']=='REJECTED_FIRST_DIFF_DIAGNOSTIC_ONLY'
    reject(lambda:read_reference(p,h,{'DSO':'changed'}))
    original=p.read_bytes();p.write_bytes(original+b' ');reject(lambda:read_reference(p,h,header['identity']))
print(json.dumps(dict(CPU_contract_passed=True,GPU='NOT_RUN',checks=['BF16 CPU deterministic argmax and first-token tie','nonfinite and nativeCPU mismatch rejection','physicalrow/past/autoregressive prefix/EOS/completeness','immutableSHA/DSOidentity rejection','allcandidateIDs saved BEFORE firstdiff assertion; no continuation/replayTPS'])))
