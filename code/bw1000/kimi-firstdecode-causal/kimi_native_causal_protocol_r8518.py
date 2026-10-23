"""Finite diagnosis on an already-loaded four-stage Engine; no service TPS.

Reset + immutable whole prefixes retain all real companion jobs. Capturing a
target's boundaries does not establish equality of uncaptured full native state.
Outer historical runner retains Engine ownership and normal destroy.
"""
import hashlib,json,os,pathlib,time
from online_kimi_provider_r8517 import KimiLoadedAdapter
from kimi_firstdecode_consumer_r8517 import replay_cut
from kimi_native_state_consumer_r8519 import summarize_native_target
from kimi_snapshot_budget_r8517 import snapshot_bound

CUT_SHA='fedca78c195d1a5a91d5c3491855308ce4723b886a3540d7baa7b57d128ce05f'
DECODER_POINTS=['entry','attn_res','attn_norm','attn_output','attn_added',
                'ffn_res','ffn_norm','ffn_output','exit']

class BorrowedReference(KimiLoadedAdapter):
    def close(self):self.closed=True

def read_boundary_records(trace_root):
    records=[]
    if not trace_root:return records
    for p in sorted(pathlib.Path(trace_root).glob('stage*-rank*.jsonl')):
        for line in p.read_text().splitlines():
            v=json.loads(line)
            assert v['packed_target_only'] and len(v['requests'])==1
            records.append(v)
    return records

def boundary_index(case):
    out={}
    for packet in case:
        for job in packet['jobs']:
            for v in job['boundaries']:
                key=(job['kind'],v['stage'],v['layer'],v['point'],v['rank'])
                assert key not in out,key
                out[key]=dict(fnv1a64=v['requests'][0]['fnv1a64'],
                    tokens=v['requests'][0]['tokens'],hidden=v['hidden'],
                    nonfinite=v['requests'][0]['nonfinite'])
    return out

def compare_boundaries(left,right):
    a,b=boundary_index(left),boundary_index(right)
    assert a.keys()==b.keys(),('boundary_geometry_missing',sorted(a.keys()-b.keys()),sorted(b.keys()-a.keys()))
    return [dict(key=list(k),left=a[k],right=b[k])for k in sorted(a)if a[k]!=b[k]]

def select_stage(differences):
    # The earliest changed stage is observational. Its incoming residual/history
    # may still differ even when the selected incoming hidden equals.
    return min((v['key'][1]for v in differences),default=None)

def select_layer(differences):
    return min((v['key'][2]for v in differences if v['key'][3]=='decoder'),default=None)

def compare_native(left,right):
    def index(case):
        out={}
        for packet in case:
            for job in packet['jobs']:
                for native in job.get('native',[]):
                    for category in ('states','intermediates'):
                        for point,v in native[category].items():
                            key=(job['kind'],native['stage'],native['layer'],category,point,native['rank'])
                            assert key not in out
                            out[key]={k:v[k]for k in ('shape','storage_dtype','bytes','SHA256')}
        return out
    a,b=index(left),index(right)
    assert a.keys()==b.keys(),'native selected boundary missing in comparison'
    return [dict(key=list(k),left=a[k],right=b[k])for k in sorted(a)if a[k]!=b[k]]

def run_native_causal(g,reset):
    node,r,tag,lib,ctx=g['node'],g['r'],g['tag'],g['lib'],g['ctx']
    control,report,check=g['control'],g['report'],g['check']
    assert g['pp']==4 and g['tp_nodes']==1 and node in range(4)
    p=r/'incoming/Kimi-postprefill-firstdecode-cuts-r8507.json'
    assert hashlib.sha256(p.read_bytes()).hexdigest()==CUT_SHA
    cuts={v['label']:v for v in json.loads(p.read_text())['cuts']}
    root=r/'results'/('Kimi-native-causal-r8518-'+tag)
    root.mkdir(exist_ok=True)
    report['native_causal']=dict(status='RUNNING',raw_root=str(root),
        acceptance='FAILED_OBSERVED historical cross-policy IDs remain failed',
        scope='Teacher-forced O1/W1 history + firstdecode; NOT service or speedup',
        phases=[],source_cut_SHA256=CUT_SHA)
    def phase(mode,label,cut,layers,points,detail=None):
        begin=time.perf_counter();reset(mode+'-'+label)
        report['phase']='native-'+mode+'-'+label;g['save']()
        local=[]
        def persist(value):
            local.append(value)
            path=root/(mode+'-'+label+'-node'+str(node)+'-outputs.json')
            tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(local,indent=2));tmp.replace(path)
        def memory_gate():
            assert all(free>1024**3 for free,total in g['memory']()),'diagnostic 1GiB forward reserve'
        adapter=BorrowedReference(lib,ctx,stage=node,stages=4,rows=32,
            check=check,memory_gate=memory_gate,destroy=lambda p:None,
            full_logit_reference=True)
        records=replay_cut(adapter,cut,stage=node,case=mode+'-'+label,
            root=root,collect=control.collect,persist=persist,layers=layers,
            points=points,hash_only=True,detail_layer=detail,
            observe_kv=detail is not None,state_rows='all')
        adapter.close()
        jobs=[]
        for v in records:
            if v['packed_target']is None:continue
            directory=v['perstage_trace_dirs'][node]
            native=summarize_native_target(directory,v['job'],v['packed_target']) if detail is not None else []
            budget=snapshot_bound(v['job'],state_rows='all',observe_kv=detail is not None)
            assert all(x['raw_bytes']<=max(budget['KDA_raw_bytes_upper'],budget['MLA_raw_bytes_upper'])for x in native)
            observation=v['emitter_observation'];target=v['packed_target']
            emitter_target=dict(BF16_logits_SHA256=observation['per_request_BF16_logits_SHA256'][target],top2=observation['top2'][target])
            jobs.append(dict(kind=v['job']['kind'],seq=v['seq'],
                boundaries=read_boundary_records(directory),native=native,
                snapshot_budget=budget,emitter_target=emitter_target))
        result=dict(node=node,label=label,mode=mode,jobs=jobs,
                    wall_s=time.perf_counter()-begin,
                    outputs_path=str(root/(mode+'-'+label+'-node'+str(node)+'-outputs.json')))
        packets=control.collect('native-phase:'+mode+':'+label,result)
        report['native_causal']['phases'].append(packets);g['save']()
        return packets
    def quartet(mode,layers,points,detail=None):
        values={}
        for label,source in [('Oa','O1'),('Ob','O1'),('Wa','W1'),('Wb','W1')]:
            values[label]=phase(mode,label,cuts[source],layers,points,detail)
        repeat_O=compare_boundaries(values['Oa'],values['Ob'])
        repeat_W=compare_boundaries(values['Wa'],values['Wb'])
        cross=compare_boundaries(values['Oa'],values['Wa'])
        summary=dict(mode=mode,repeat_O=repeat_O,repeat_W=repeat_W,cross=cross)
        def emitter_index(case):return {job['kind']:job['emitter_target']for job in case[3]['jobs']}
        summary['target_full_logits']=dict(Oa=emitter_index(values['Oa']),Ob=emitter_index(values['Ob']),Wa=emitter_index(values['Wa']),Wb=emitter_index(values['Wb']))
        if detail is not None:
            summary['native_repeat_O']=compare_native(values['Oa'],values['Ob'])
            summary['native_repeat_W']=compare_native(values['Wa'],values['Wb'])
            summary['native_cross']=compare_native(values['Oa'],values['Wa'])
        report['native_causal'][mode]=summary;g['save']()
        assert not repeat_O and not repeat_W,'same-trace boundary instability: evidence saved, stop'
        assert not summary.get('native_repeat_O')and not summary.get('native_repeat_W'),'same-trace valid native state instability: evidence saved, stop'
        return cross
    ranges=g['ranges'];assert ranges==[[0,24],[24,47],[47,71],[71,93]]
    coarse_layers=sorted(set([v for a,b in ranges for v in (a,b-1)]+[93]))
    coarse=quartet('coarse',coarse_layers,['stage_input','decoder','output_attn_res','final_norm'])
    stage=select_stage(coarse)
    report['native_causal']['first_differing_stage_observed']=stage;g['save']()
    if stage is not None:
        begin,end=ranges[stage]
        decoder=quartet('decoder',list(range(begin,end)),['decoder'])
        layer=select_layer(decoder)
        report['native_causal']['first_differing_layer_observed']=layer;g['save']()
        if layer is not None:
            # Existing native detail hooks capture actual read/write state union
            # and existing MLA intermediates. No state clone or math override.
            residual=['residual_slot_'+str(i)for i in range(9)]
            quartet('detail',[layer],DECODER_POINTS+residual+['decoder'],detail=layer)
    report['native_causal'].update(status='FINITE_DIAGNOSIS_COMPLETE',
        generation_acceptance='NOT_QUALIFIED; frozen history replay is diagnostic',
        performance_claim=False,full_state_equality_claim=False,
        raw_activations='REMOTE_ONLY',all_reset_barriers_completed=True)
    g['save']()
