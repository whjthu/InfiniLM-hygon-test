"""CPU injected adapter/control, real finite consumer and actual cut files.

Exercises the conditional coarse->decoder->detail path, not native math or TPS.
"""
import copy,hashlib,json,os,pathlib,tempfile
import kimi_native_causal_protocol_r8518 as protocol

r=pathlib.Path.cwd();cuts=json.loads((r/'Kimi-postprefill-firstdecode-cuts-r8507.json').read_text())['cuts']
calls=[];resets=[]
class Adapter:
    full_logit_reference=True
    def __init__(self,*args,**kwargs):pass
    def close(self):pass
    def forward(self,j):
        directory=os.environ.get('INFINILM_KIMI_TRACE_DIR','')
        # Non-target jobs have no trace env, but the immutable prefix sequence
        # still identifies O vsW: W has4/2rows, O has1row in this actual cut.
        source=cuts[0]if len(j['rows'])==1 else cuts[1]
        expected=source['jobs'][j['seq']]['frozen_actual_ids']
        assert j==source['jobs'][j['seq']]['job']
        calls.append(copy.deepcopy(j))
        self.reference_observation=dict(finite=True,native_equals_CPU=True,
            per_request_BF16_logits_SHA256=['cpu-injected'for _ in j['keys']],
            top2=[dict(first_id=v)for v in expected])
        if directory:
            layers=list(map(int,os.environ['INFINILM_KIMI_TRACE_BOUNDARY_LAYERS_R8506'].split(',')))
            points=os.environ['INFINILM_KIMI_TRACE_BOUNDARY_POINTS_R8506'].split(',')
            lines=[]
            for layer in layers:
                if layer not in range(24):continue
                for point in points:
                    if point=='stage_input'and layer!=0:continue
                    if point in ('output_attn_res','final_norm'):continue
                    lines.append(dict(stage=0,rank=0,layer=layer,point=point,hidden=7168,
                        packed_target_only=True,requests=[dict(fnv1a64='W'if len(j['rows'])>1 else'O',tokens=len(j['inputs'][0]),nonfinite=0)]))
            (pathlib.Path(directory)/'stage0-rank0.jsonl').write_text('\n'.join(map(json.dumps,lines))+'\n')
        return expected
class Control:
    def collect(self,label,value):
        if label.startswith('native-phase:'):
            return [value]+[dict(value,node=i,jobs=[])for i in range(1,4)]
        return [copy.deepcopy(value)for _ in range(4)]
protocol.BorrowedReference=Adapter
with tempfile.TemporaryDirectory(prefix='Kimi-full-CPU-r8524-',dir=r)as tmp:
    root=pathlib.Path(tmp);(root/'incoming').mkdir();(root/'results').mkdir()
    (root/'incoming/Kimi-postprefill-firstdecode-cuts-r8507.json').write_bytes((r/'Kimi-postprefill-firstdecode-cuts-r8507.json').read_bytes())
    report={};g=dict(node=0,r=root,tag='cpu-fixture',lib=None,ctx=1,control=Control(),report=report,check=lambda x:None,tp_nodes=1,pp=4,ranges=[[0,24],[24,47],[47,71],[71,93]],memory=lambda:[(6*1024**3,64*1024**3)]*8,save=lambda:None)
    protocol.run_native_causal(g,resets.append)
    c=report['native_causal']
    assert c['status']=='FINITE_DIAGNOSIS_COMPLETE'and c['first_differing_stage_observed']==0 and c['first_differing_layer_observed']==0
    assert len(calls)==54 and len(resets)==12 and len(c['phases'])==12
    assert len(list((root/'results').rglob('*outputs.json')))==12
    assert not c['performance_claim']and not c['full_state_equality_claim']
out=dict(revision=8524,CPU_actual=True,GPU_executed=False,native_or_network_execution=False,
    actual_immutable_cut_jobs_consumed=54,reset_phases_exercised=12,
    conditional_paths=['coarse','firststage decoder','firstlayer detail'],
    before_assert_outputs_and_fullprefix_namespace=True,
    source_sha256={p:hashlib.sha256((r/p).read_bytes()).hexdigest()for p in ['kimi_native_causal_protocol_r8518.py','kimi_firstdecode_consumer_r8517.py']},
    limits='Injected CPU adapter/control; nativehooks, actualowner barriers and GPU cleanup remain unmeasured')
p=r/'Kimi-causal-full-consumer-CPU-r8524.json';assert not p.exists();p.write_text(json.dumps(out,indent=2),encoding='utf8',newline='\n');print(json.dumps(out))
