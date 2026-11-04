"""Finite offline0/2/7 timing/cost supplement to the frozen strict reader.

No collection, HTTP, tokenizer, Runtime or job. Original scalar HTTP walls have
no independently archived HTTP start/end pair; backend spans are single-owner
monotonic only. Mixed W2/W4 resource rates are not online throughput or SLOs.
"""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import importlib.util
import json
import math
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('index');p.add_argument('strict_audit');p.add_argument('--output',required=True);a=p.parse_args()
    parent=Path(__file__).resolve().parents[1]/'source-v1'
    sha=lambda b:hashlib.sha256(b).hexdigest()
    assert sha((parent/'analyze_original.py').read_bytes())=='692eb8a6c9afbf7b5645a46d5d3f58a8203bd89c89a20ee1ffa5e8f4887403aa'
    sys.path.insert(0,str(parent));spec=importlib.util.spec_from_file_location('frozen_r464_original_reader',parent/'analyze_original.py');f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
    index=json.loads(Path(a.index).read_bytes());audit=json.loads(Path(a.strict_audit).read_bytes())
    assert audit['supplied_slots']==[0,2,7]and audit['verified_HTTP']==192 and audit['unverified_or_missing_HTTP']==320 and not audit['full512_evidence_verified']
    assert all(s['state']=='FULL_ORIGINALS_VERIFIED'and not s['failures']and s['normal_owners']==16 for s in audit['slots']if s['supplied'])
    store=f.Originals();store.add(index['CPU918_archive'],f.CPU918)
    for archive in index['execution_archives']:store.add(archive['path'],archive['SHA256'])
    assert store.archives==audit['input_and_execution_archives']
    cases,_,_,_=f.pinned_inputs(store);case={c['id']:c for c in cases}
    scorer_path=Path(__file__).resolve().parents[2]/'GLM-W4-fewshot-quality-r444/scoring.py'
    score=f.leaf('frozen_r444_quality_for192',scorer_path,'c9c39a5fc185ed6bfc5f48c2b547d4ef36a4cfed351626835793fcf0ab5b521c')
    rows=[];out=[];total_card_s=0;total_visible=total_correct_visible=total_correct=0
    for item in index['slots']:
        slot=item['slot'];client=store.data(item['client_path']);owner=store.data(item['run_directory']+'/owner-1-0.json');native=owner['service_requests']
        assert len(client['rows'])==len(native)==64
        first=native[0]['admission']['accepted_monotonic'];last=native[-1]['ended_monotonic'];span=last-first
        assert math.isfinite(span)and span>0
        accounting=[line.split('|')for line in store.raw(item['accounting_path']).decode().splitlines()if'|'in line]
        gpu=f.BINDINGS[slot]['GPU'];actual=[r for r in accounting if r[0]==gpu];assert len(actual)==1 and actual[0][2:4]==['COMPLETED','0:0']
        gpu_s=int(actual[0][4]);assert gpu_s>0;card_s=16*gpu_s;total_card_s+=card_s
        local=[]
        for i,(c,r)in enumerate(zip(client['rows'],native)):
            assert c['local_owner_index']==i and c['generated_IDs_SHA256']==r['generated_token_ids_SHA256']
            if i:assert native[i-1]['ended_monotonic']<=r['admission']['accepted_monotonic']
            where=PurePosixPath(item['client_path']).parent/('%03d-%s-W%d'%(i,c['case_id'],c['window']))
            wire=store.data(str(where/'response.wire'));message=wire['choices'][0]['message']
            graded=score.grade(case[c['case_id']],message.get('content')or'',natural_EOS=c['natural_EOS'],cap_hit=c['cap_hit'],unexpected_tool=bool(message.get('tool_calls')))
            assert all(graded[k]==c[k]for k in ['correct','parse_failure','answer_matches_gold','natural_EOS','cap_hit','parsed_answer'])
            timing=f.observed_clocks(r,c);visible=timing['visible_tokens']
            local.append(dict(case_id=c['case_id'],window=c['window'],timing=timing,**graded))
            total_visible+=visible;total_correct_visible+=visible*int(graded['correct']);total_correct+=int(graded['correct'])
        rows.extend(local)
        windows={}
        for w in[2,4]:
            subset=[r for r in local if r['window']==w];http=math.fsum(r['timing']['HTTP_wall_reported_client_s']for r in subset)
            windows[str(w)]=dict(requests=len(subset),correct=sum(r['correct']for r in subset),visible_tokens=sum(r['timing']['visible_tokens']for r in subset),quality_correct_visible_tokens=sum(r['timing']['visible_tokens']*int(r['correct'])for r in subset),HTTP_scalar_wall_sum_s=http,correct_request_per_HTTP_scalar_second=sum(r['correct']for r in subset)/http,scope='Serialized recorded HTTPwall scalar denominator; not complete cohort wall, TTFT, streaming or per-token clientSLO.')
        out.append(dict(slot=slot,requests=64,correct=sum(r['correct']for r in local),GPU_step=gpu,GPU_ElapsedRaw_s=gpu_s,allocated_compute_cards=16,allocated_GPU_card_seconds=card_s,single_leader_first_admission_to_last_owner_end_s=span,correct_requests_per_owner_cohort_second=sum(r['correct']for r in local)/span,visible_tokens_per_owner_cohort_second=sum(r['timing']['visible_tokens']for r in local)/span,quality_correct_visible_tokens_per_owner_cohort_second=sum(r['timing']['visible_tokens']*int(r['correct'])for r in local)/span,windows=windows))
    assert len(rows)==192 and total_correct==186
    result=dict(scope='Original192/512 only; old886+terminal2/7; no new execution',index_SHA256=sha(Path(a.index).read_bytes()),strict_audit_SHA256=sha(Path(a.strict_audit).read_bytes()),strict_reader_SHA256=sha((parent/'analyze_original.py').read_bytes()),slots=out,independently_wire_rescored_correct_HTTP=total_correct,visible_tokens=total_visible,quality_correct_visible_tokens=total_correct_visible,allocated_GPU_card_seconds_including_load_and_cleanup=total_card_s,mixed_window_correct_requests_per_allocated_card_second=total_correct/total_card_s,mixed_window_correct_visible_tokens_per_allocated_card_second=total_correct_visible/total_card_s,recomputed_timing=f.timing_summary(rows),planned_HTTP=512,verified_HTTP=192,missing_HTTP=320,full_quality_or_NI=False,limits=['Owner span uses one stage1rank0 monotonic clock and includes interrequest client/validation/dispatch gaps; omits firstHTTP-before-admission/lastHTTP-after-ownerend.','No cross-process or cross-node absolute clock subtraction; no sum of per-slot cohort rates as simultaneous throughput.','GPU-cardsecond rate is a mixedW2/W4 resource-cost measure including load/cleanup, not an arm comparison or production peak.','HTTPwall is the original scalar; client start/end clocks absent; nonstream client gives no incremental clientTTFT/tokenSLO.'])
    assert result['recomputed_timing']==audit['timing']
    Path(a.output).write_bytes((json.dumps(result,indent=2,allow_nan=False)+'\n').encode())
    print(json.dumps(dict(verified_HTTP=192,missing_HTTP=320,independently_rescored_correct=total_correct,allocated_GPU_card_seconds=total_card_s)))


if __name__=='__main__':main()
