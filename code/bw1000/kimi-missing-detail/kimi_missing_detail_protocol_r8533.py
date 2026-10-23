"""Capture O confirmation and missing W, then leave all parsing until terminal.

No inherited eight-QA/B16 runs are repeated: unchanged native DSO passed those
in599. Frozen actual prefix IDs are persisted each step; this is diagnosis only.
"""
import hashlib,json,pathlib,time,os
from online_kimi_provider_r8517 import KimiLoadedAdapter
from kimi_firstdecode_capture_r8535 import replay_cut

class BorrowedReference(KimiLoadedAdapter):
    def close(self):self.closed=True

def run(g):
    r,tag,node,control,lib,ctx=g['r'],g['tag'],g['node'],g['control'],g['lib'],g['ctx']
    report=g['report'];assert (g['tp_nodes'],g['pp'])==(1,4)
    assert lib.modern_kimi_vocab_head_get_r8341()==0
    p=r/'incoming/Kimi-postprefill-firstdecode-cuts-r8507.json'
    assert hashlib.sha256(p.read_bytes()).hexdigest()=='fedca78c195d1a5a91d5c3491855308ce4723b886a3540d7baa7b57d128ce05f'
    cuts={v['label']:v for v in json.loads(p.read_text())['cuts']}
    root=r/'results'/('Kimi-missing-detail-r8533-'+tag);root.mkdir(exist_ok=True)
    report.update(scope='Only O confirmation plus missing W native layer0 capture; no TPS or modelqualification',
        native_capture=dict(status='RUNNING',cases=[],raw_root=str(root),
            inherited_599='before8QA + historical B16M4 anchor passed on identical DSO; not rerun',
            before_pool_read_semantics='UNKNOWN; originalinitflag notcaptured; before pool observation not readproof'),
        QA=[],GPU_qualification=False)
    points=['entry','attn_res','attn_norm','attn_output','attn_added',
            'ffn_res','ffn_norm','ffn_output','exit','decoder']+['residual_slot_'+str(i)for i in range(9)]
    os.environ['INFINILM_KIMI_MOE_EXECUTION_R4665']='1'
    control.collect('missing-detail-serving-policy1',{})
    for label,source in [('Oconfirm','O1'),('Wmissing','W1')]:
        begin=time.perf_counter();report['phase']='capture-'+label;g['save']()
        g['check'](lib.modern_kimi_engine_reset_r2475(ctx))
        control.collect('capture-reset:'+label,{'all8_native_reset_completed':True})
        assert all(free>5*1024**3 for free,total in g['memory']()),'5GiB phase reserve'
        def memory_gate():assert all(free>1024**3 for free,total in g['memory']())
        adapter=BorrowedReference(lib,ctx,stage=node,stages=4,rows=32,check=g['check'],
            memory_gate=memory_gate,destroy=lambda p:None,full_logit_reference=True)
        saved=[];output=root/(label+'-node'+str(node)+'-outputs.json')
        def persist(v):
            saved.append(v);tmp=output.with_suffix('.tmp');tmp.write_text(json.dumps(saved,indent=2));tmp.replace(output)
        records=replay_cut(adapter,cuts[source],stage=node,case=label,root=root,
            collect=control.collect,persist=persist,layers=[0],points=points,
            hash_only=True,detail_layer=0,observe_kv=False,state_rows='all')
        adapter.close()
        # No journal discovery, raw array reading or crosscase hashing here.
        # Every owner completes BOTH captures before any offline consumer starts.
        packet=dict(label=label,node=node,source=source,outputs_path=str(output),
            actual_jobs=len(records),all_prefix_IDs_equal=all(v['firstdiff']is None for v in records),
            wall_s=time.perf_counter()-begin)
        allowners=control.collect('capture-complete:'+label,packet)
        report['native_capture']['cases'].append(allowners);g['save']()
    report['native_capture'].update(status='CAPTURE_COMPLETE_OFFLINE_PENDING',
        generation_acceptance='FAILED_OBSERVED historical O/W finaltokens',
        native_comparison='NOT_RUN_INSIDE_MODEL',performance_qualified=False)
    report['serving_protocol_complete']=True;report['phase']='capture-complete';g['save']()
