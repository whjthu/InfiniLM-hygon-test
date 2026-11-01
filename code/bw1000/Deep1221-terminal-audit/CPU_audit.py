from pathlib import Path
import datetime,hashlib,json,math,statistics,subprocess,sys
from fractions import Fraction
from types import SimpleNamespace
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
S=R/'incoming/Deep-semantic68-owner-r320';pin='076fdb2e6bf89d0cd2adf4cc7b26dea13f47556b7efa88c936f0db3556e988f7'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
host=lambda p:R/str(p).removeprefix('/refactor/').removeprefix(str(R)+'/')
assert sha(S/'SOURCE_MANIFEST.json')==pin
sm=json.loads((S/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(S/n)==h for n,h in sm['files_SHA256'].items())
sys.path.insert(0,str(S))
import client_owner_r281 as owner
import client_multicall_r281 as multicall
import client_quality_r281 as quality
P=R/'results/Deep-latest-full-owner-r287-910374.1221.json';q=json.loads(P.read_bytes())
W=R/'results/Deep-latest99-semantic68-Model-r322-watch.json';w=json.loads(W.read_bytes())
assert q['completed']and q['passed']and q['actual_HTTP']==q['actual_Native']==68
assert q['actual_step']=='910374.1221'and q['source_manifest_SHA256']==pin
assert q['normal_server_shutdown']and q['normal_Native_shutdown']and not q['cleanup']['forced_cleanup']and not q['cleanup']['errors']
assert q['owner_exit_observation']['all8_unique_threads_exited']
assert len({v['tid']for v in q['owner_exit_observation']['owners']})==8
assert all(not v['task_exists_after_destroy']for v in q['owner_exit_observation']['owners'])
assert w['completed']and w['post_empty']and not w['post']['kfd_pids']and w['post']['HTTP43882_bindable']
assert w['post']['actual_step']=='910374.1222'
accounts=[s.split('|')for s in subprocess.check_output(['sacct','-S','2026-10-31','-j','910374,910407','-n','-P','-o','JobID,JobName%120,State,ExitCode,Elapsed,NodeList%120'],text=True,timeout=30).splitlines()]
accounts=[v for v in accounts if v[0]in ['910374.1221','910374.1222','910407.636']]
assert len(accounts)==3 and all(v[2:4]==['COMPLETED','0:0']for v in accounts)
args=SimpleNamespace(expected_step='910374.1221',manifest_sha=pin)
requests=q['requests'];assert len(requests)==68 and len({r['request_id']for r in requests})==68
metrics=[]
for i,r in enumerate(requests):
    checked=owner.row(P,args,r['request_id'],i,r['admission']['diagnostic_user'],8192 if 2<=i<4 else 4096,lambda:8000)
    assert checked['native_finish_reason']=='eos_token'
    metrics.append(dict(index=i,request_id=r['request_id'],prompt_tokens=r['prompt_tokens'],
        generated_tokens=r['actual_generated_tokens'],natural_EOS=True,**owner.backend_metrics(r)))
phase_hashes={};multicalls=[];scores={}
for name,phase in q['phases'].items():
    p=host(phase['path']);assert sha(p)==phase['SHA256'];v=json.loads(p.read_bytes())
    assert v==phase['result']and v['completed']and v['passed'];phase_hashes[name]={'path':str(p),'SHA256':sha(p)}
mcroot=host(q['phases']['multicall']['path']).parent
for k,mode in enumerate(['chat','thinking']):
    folder=mcroot/mode;wires=[json.loads((folder/('request-%03d.wire'%n)).read_bytes())for n in [1,2]]
    first=wires[0]['choices'][0];second=wires[1]['choices'][0]
    calls=multicall.validated_calls(first['message'],first['finish_reason'])
    reversed_results,expected=multicall.execute_and_reverse(calls,folder/'workspace')
    body=json.loads((folder/'request-002.request.json').read_bytes())
    assert body['messages'][-2:]==reversed_results
    presentation=multicall.check_final(second['message'],second['finish_reason'],expected)
    assert [v['id']for v in wires]==[requests[2*k+j]['request_id']for j in range(2)]
    multicalls.append({'mode':mode,'HTTP':2,'natural_stop':True,'presentation':presentation,
        'nonce_integer_association_correct':True,'original_opaque_ID_reverse_preserved':True,
        'wire_SHA256':[sha(folder/('request-%03d.wire'%n))for n in [1,2]]})
root=host(q['phases']['quality']['path']).parent
for suite,filename in [('GSM8K','gsm8k-fixed32.json'),('MMLU','mmlu-fixed32.json')]:
    fixture=S/'benchmark'/filename;cases=json.loads(fixture.read_bytes())['cases'];rows=[]
    published=q['phases']['quality']['result']['suites'][suite]
    assert sha(fixture)==published['fixture_SHA256']and len(cases)==32
    for i,case in enumerate(cases):
        folder=root/suite/('%02d'%i);wirefile=folder/'request-001.wire';wire=json.loads(wirefile.read_bytes())
        body=json.loads((folder/'request-001.request.json').read_bytes());question=case['question']
        if suite=='MMLU':question+='\n'+'\n'.join('%s. %s'%(chr(65+k),v)for k,v in enumerate(case['choices']))
        assert body['messages'][-1]['content']==question and body['max_tokens']==4096
        content=wire['choices'][0]['message'].get('content')or''
        value=quality.numeric(content)if suite=='GSM8K'else quality.letter(content)
        gold=Fraction(case['gold_numeric'].replace(',',''))if suite=='GSM8K'else case['answer']
        correct=value is not None and value==gold
        assert wire['choices'][0]['finish_reason']=='stop'
        assert published['rows'][i]['correct']is correct and not published['rows'][i]['cap_hit']
        rows.append({'offset':i,'source_index':case['source_index'],'correct':correct,'parse_failure':value is None,
            'natural_EOS':True,'wire_SHA256':sha(wirefile)})
    n=sum(r['correct']for r in rows);assert n==published['correct']
    scores[suite]={'correct':n,'count':32,'accuracy':n/32,'Wilson95':owner.wilson(n,32),
        'parse_failures':sum(r['parse_failure']for r in rows),'caps':0,'natural_EOS':32,
        'fixture_SHA256':sha(fixture),'rows':rows,'pilot_not_full_benchmark':True}
speeds=[r['backend_emitter_decode_tok_s']for r in metrics if r['backend_emitter_decode_tok_s']is not None]
assert len(speeds)==68 and all(math.isfinite(v)and v>0 for v in speeds)
print(json.dumps({'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'read_only':True,
    'passed':True,'accounting':accounts,'owner_manifest_SHA256':pin,
    'controller_manifest_SHA256':w['controller_manifest_SHA256'],'report_SHA256':sha(P),'watch_SHA256':sha(W),
    'actual_GPU':'910374.1221','CPU_keeper':'910407.636','actual_HTTP':68,'actual_Native':68,
    'phase_hashes':phase_hashes,'multicall':multicalls,'quality':scores,'request_metrics':metrics,
    'backend_emitter_decode_summary':{'n':len(speeds),'min':min(speeds),'median':statistics.median(speeds),
        'max':max(speeds),'strictly_above15':sum(v>15 for v in speeds),'client_incremental_SLO_qualified':False},
    'normal_Native_and_HTTP_shutdown':True,'all8_owner_threads_exited':True,'post':w['post'],
    'original1218_strict_bare_JSON_full_plan_failure_preserved':True,
    'full_quality_no_regression_proven':False,'new_speedup_claim':False}))
