"""CPU control/schema fixtures through actual consumers, no fake GPU numbers."""
import json,tempfile,pathlib,os
from kimi_static_localize_r8572 import jobs,choose_detail
from kimi_static_cut_capture_r8572 import replay_cut
plan=json.loads(pathlib.Path('Kimi-static-localize-plan-r8572.json').read_text())
O,N=jobs(plan,'O',33),jobs(plan,'N',33)
assert len(O['jobs'])==37 and len(N['jobs'])==34
for x,y in zip(O['jobs'][4:],N['jobs'][1:]):assert {k:v for k,v in x['job'].items()if k!='seq'}=={k:v for k,v in y['job'].items()if k!='seq'}
observed=[];saved=[]
class Adapter:
    full_logit_reference=False
    def forward(self,j):
        observed.append((j['seq'],os.environ.get('INFINILM_KIMI_TRACE_DIR'),os.environ.get('INFINILM_KIMI_TRACE_FULL_FIRST'),os.environ.get('INFINILM_KIMI_TRACE_MLA_FULL')))
        return [7]*len(j['keys']) # Deliberate mismatch must be persisted, not gold.
def collect(label,v):return[dict(v)for _ in range(4)]
with tempfile.TemporaryDirectory()as td:
    replay_cut(Adapter(),O,stage=0,case='fixture',root=td,collect=collect,persist=saved.append,layers=[0],points=['decoder'],hash_only=False,detail_layer=0,trace_ordinals=(0,1,31,33))
    assert len(saved)==37 and len(observed)==37
    assert [v[0]for v in observed if v[1]]==[0,4,34,36]
    assert all(v[2]=='0'and v[3]is None for v in observed if v[1])
    assert any(v['firstdiff']is not None for v in saved)
    base=pathlib.Path(td)
    def fixture(name,changed=False,point='decoder'):
        dirs=[]
        for stage in range(4):
            d=base/(name+str(stage));d.mkdir();dirs.append(str(d))
            for rank in range(8):
                row=dict(stage=stage,rank=rank,layer=stage*24,point=point,tokens=512,hidden=7168,packed_target_only=True,
                    requests=[dict(request=0,fnv1a64='DIFF'if changed and stage==0 and rank==0 else'SAME',nonfinite=0)],
                    original_shape=[1,512,7168],original_strides=[3670016,7168,1],original_offsets=[0,512])
                (d/f'stage{stage}-rank{rank}.jsonl').write_text(json.dumps(row)+'\n')
            (d/f'stage{stage}-rank0-layer0-kda-call0.jsonl').write_text('mixed raw journal must be ignored')
        return dict(records=[dict(packed_target=0,job=O['jobs'][0]['job'],perstage_trace_dirs=dirs)])
    a,b=fixture('a'),fixture('b',True);assert choose_detail(a,b)['detail_layer']==0
    c,d=fixture('c',False,'stage_input'),fixture('d',True,'stage_input');assert choose_detail(c,d)['status']=='UNKNOWN_CARRY_OR_FINAL_RESIDUAL_BOUNDARY'
out=dict(CPU_only=True,GPU=False,actual_consumers=['jobs','replay_cut','coarse_index','choose_detail'],O_jobs=37,N_jobs=34,
    same_decodeM4_rows_pasts_teacher_inputs=True,trace_only_selected_ordinals=True,full_rank0_6_selector_preserved=True,
    mismatch_actuals_persisted=True,mixed_journal_schema_fixture=True,carry_gap_stops_UNKNOWN=True,
    scope='Control/schema fixtures only; no Model/activation/numerical/performance qualification')
pathlib.Path('Kimi-static-localize-CPU-r8573.json').write_text(json.dumps(out,indent=2),encoding='utf-8',newline='\n');print(json.dumps(out))
