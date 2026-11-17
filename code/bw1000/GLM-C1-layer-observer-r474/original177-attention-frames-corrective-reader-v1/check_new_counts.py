"""New count/phase regression only; synthetic events, no original artifacts or GPU."""
import copy,json,sys
from pathlib import Path
from attention_evidence import attention_windows

def frame(rank=0):
    marks=[]
    def mark(name):
        t=len(marks)/1000;marks.append(dict(name=name,owner_host_s=t,gpu_from_first_s=t))
    def detail(layer,name):mark(f'detail.layer.{layer}.attnd1.'+name)
    def pair(layer,name,body=lambda:None):
        detail(layer,name+'.begin');body();detail(layer,name+'.return')
    def projection(layer,m,n,k,row=0,site='projection',column=0):
        detail(layer,f'{site}.tuple.M{m}.N{n}.K{k}.row_offset{row}.column_offset{column}.input_BF16.contiguous_checked.canonical_stride_inferred_{k}_1.raw_stride_UNKNOWN.kernel_provider_UNKNOWN')
        detail(layer,site+'.dispatch.cached_r809.kernel_provider_UNKNOWN')
        pair(layer,site+'.original_success_fence')
    for layer in range(78):
        mark(f'layer.{layer}.'+('full' if layer==0 else 'shared')+'.begin')
        if layer in (0,3):
            def prepare():
                projection(layer,1,2048,6144);projection(layer,1,1024,2048,rank*1024);projection(layer,1,576,6144)
            pair(layer,'prepare.forward_latent',prepare)
            if layer==0:
                mark('detail.layer.0.attn.indexer.features.begin')
                projection(layer,1,128,6144);projection(layer,1,256,2048,rank*256)
                mark('detail.layer.0.attn.indexer.features.return')
            pair(layer,'fresh.expand_cached',lambda:projection(layer,1,1792,512,rank*1792))
            detail(layer,'selection.complete_prefix');detail(layer,'v398.enabled')
            def hist():
                detail(layer,'history.dispatch.standard');projection(layer,1313,1792,512,rank*1792)
            pair(layer,'history.selected_value',hist)
            pair(layer,'attention.selected');pair(layer,'attention.result_copy')
            pair(layer,'output.forward',lambda:projection(layer,1,6144,1024,site='output',column=rank*1024))
            pair(layer,'output.original_stage_finish_or_stream_wait')
            mark(f'detail.layer.{layer}.FFN.begin')
            n=768 if layer==0 else 128
            projection(layer,1,n,6144,rank*n);projection(layer,1,n,6144,rank*n)
            mark(f'detail.layer.{layer}.FFN.local.gate_up.return');mark(f'detail.layer.{layer}.FFN.end')
        mark(f'layer.{layer}.end')
    return dict(rank=rank%8,tokens=1,row_past_count=[[0,1312,1]],full_detail_layer=0,shared_detail_layer=3,event_count=len(marks),marks=marks)

def main():
    rows=[]
    def check(name,f,rank,accept):
        try:r=attention_windows(f,rank);actual=True
        except (ValueError,KeyError,TypeError) as e:actual=False;r=type(e).__name__
        rows.append(dict(name=name,expected=accept,actual=actual,passed=actual==accept))
        if accept and actual:
            assert [len(x['tuples']) for x in r]==[10,8]
            assert [sum(t['caller_phase']=='FFN_gate_up' for t in x['tuples']) for x in r]==[2,2]
            assert [sum(w['caller_phase']=='FFN_gate_up' for w in x['windows']) for x in r]==[2,2]
    for rank in (0,8,15):check('fullframe_rank'+str(rank),frame(rank),rank,True)
    base=frame()
    def bad(name,change):
        f=copy.deepcopy(base);change(f['marks']);f['event_count']=len(f['marks']);check(name,f,0,False)
    bad('extra_projection_tuple',lambda m:m.insert(8,copy.deepcopy(m[2])))
    bad('missing_projection_dispatch',lambda m:m.pop(next(i for i,x in enumerate(m) if '.projection.dispatch.' in x['name'])))
    bad('missing_success_fence',lambda m:m.pop(next(i for i,x in enumerate(m) if '.projection.original_success_fence.begin' in x['name'])))
    bad('wrong_ffn_width',lambda m:next(x for x in m if '.N768.K6144.' in x['name']).__setitem__('name',next(x for x in m if '.N768.K6144.' in x['name'])['name'].replace('.N768.','.N128.')))
    bad('wrong_history_M',lambda m:next(x for x in m if '.M1313.' in x['name']).__setitem__('name',next(x for x in m if '.M1313.' in x['name'])['name'].replace('.M1313.','.M1.')))
    bad('partial_prefix',lambda m:next(x for x in m if x['name'].endswith('selection.complete_prefix')).__setitem__('name','detail.layer.0.attnd1.selection.partial_prefix'))
    bad('missing_FFN_anchor',lambda m:m.pop(next(i for i,x in enumerate(m) if x['name']=='detail.layer.0.FFN.begin')))
    def move_gate(m):
        a=next(i for i,x in enumerate(m) if x['name']=='detail.layer.0.FFN.begin');b=next(i for i,x in enumerate(m) if x['name']=='detail.layer.0.FFN.local.gate_up.return');m[a],m[b]=m[b],m[a]
    bad('FFN_anchor_reordered',move_gate)
    bad('wrong_indexer_shape',lambda m:next(x for x in m if '.N256.K2048.' in x['name']).__setitem__('name',next(x for x in m if '.N256.K2048.' in x['name'])['name'].replace('.N256.','.N128.')))
    result=dict(schema='new-synthetic-generic-projection-phase-count-check-v1',normal=3,reject=9,checks=rows,passed=all(r['passed'] for r in rows),remote=False,original_artifact_reads=0,GPU=False,old_fixture_replayed=False)
    raw=(json.dumps(result,indent=2)+'\n').encode();assert json.loads(raw)==result
    out=Path(sys.argv[1]);out.parent.mkdir(parents=True,exist_ok=False);out.write_bytes(raw)
    print(json.dumps(result));return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
