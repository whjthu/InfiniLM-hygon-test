"""CPU injected CAPI/control; nine actual immutable jobs, no journal parsing."""
import pathlib,json,tempfile,hashlib
import kimi_missing_detail_protocol_r8533 as p
r=pathlib.Path.cwd();cuts=json.loads((r/'Kimi-postprefill-firstdecode-cuts-r8507.json').read_text())['cuts'];calls=[]
class Lib:
    resets=0
    def modern_kimi_vocab_head_get_r8341(self):return 0
    def modern_kimi_engine_reset_r2475(self,ctx):self.resets+=1;return 0
class Adapter:
    full_logit_reference=True
    def __init__(self,*a,**kw):pass
    def close(self):pass
    def forward(self,job):
        cut=cuts[0]if len(job['rows'])==1 else cuts[1]
        assert job==cut['jobs'][job['seq']]['job']
        ids=list(cut['jobs'][job['seq']]['frozen_actual_ids']);calls.append(job)
        mismatch=len(job['rows'])==1 and job['seq']==0
        if mismatch:ids[0]=(ids[0]+1)%163840
        self.reference_observation=dict(finite=True,native_equals_CPU=not mismatch,
            per_request_BF16_logits_SHA256=['cpu-injected']*len(ids),top2=[dict(first_id=v)for v in ids])
        return ids
class Control:
    def collect(self,label,value):return [dict(value)for _ in range(4)]
p.BorrowedReference=Adapter
with tempfile.TemporaryDirectory(prefix='Kimi-capture-only-r8534-',dir=r)as tmp:
    root=pathlib.Path(tmp);(root/'incoming').mkdir();(root/'results').mkdir()
    (root/'incoming/Kimi-postprefill-firstdecode-cuts-r8507.json').write_bytes((r/'Kimi-postprefill-firstdecode-cuts-r8507.json').read_bytes())
    report={};lib=Lib();g=dict(r=root,tag='cpu',node=0,control=Control(),lib=lib,ctx=1,
        report=report,tp_nodes=1,pp=4,check=lambda v:None,memory=lambda:[(6*1024**3,64*1024**3)]*8,save=lambda:None)
    p.run(g)
    assert len(calls)==9 and lib.resets==2
    assert report['native_capture']['status']=='CAPTURE_COMPLETE_OFFLINE_PENDING'
    assert len(report['native_capture']['cases'])==2 and report['serving_protocol_complete']
    assert report['native_capture']['native_comparison']=='NOT_RUN_INSIDE_MODEL'
    assert len(list((root/'results').rglob('*outputs.json')))==2
    assert not report['native_capture']['cases'][0][0]['all_prefix_IDs_equal']
    assert report['native_capture']['cases'][1][0]['all_prefix_IDs_equal']
    prior=json.loads(next((root/'results').rglob('Oconfirm-node0-outputs.json')).read_text())
    assert prior[0]['firstdiff']==0 and prior[0]['actual_ids']!=prior[0]['frozen_ids']
out=dict(revision=8535,CPU_actual=True,GPU_executed=False,source_SHA256=hashlib.sha256((r/'kimi_missing_detail_protocol_r8533.py').read_bytes()).hexdigest(),
    actual_cut_jobs=9,reset_phases=2,both_captures_finish_without_journals=True,historical_ID_and_CPUargmax_mismatch_saved_and_W_still_captured=True,
    scope='Injected CPUadapter/control, no native math/owner/socket qualification',
    complementary_actual_proofs=['Kimi-actual-mixed-journal-CPU-r8531.json','Kimi599-actual-raw-CPU-r8532.json'])
(r/'Kimi-missing-capture-CPU-r8535.json').write_text(json.dumps(out,indent=2),encoding='utf8',newline='\n');print(json.dumps(out))
