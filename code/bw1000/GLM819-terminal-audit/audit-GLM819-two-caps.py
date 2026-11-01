from pathlib import Path
import hashlib,json,sys
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');S=R/'incoming/GLM-semantic74-owner-r328';label='GLM-latest99-semantic74-Model-r331'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sys.dont_write_bytecode=True;sys.path.insert(0,str(S));from client_quality_r290 import letter
p=R/'results'/('GLM-latest-full-owner-r290-supervisor-'+label+'.json');q=json.loads(p.read_bytes())
suite=q['phases']['quality']['result']['suites']['MMLU'];cases=json.loads((S/'benchmark/mmlu-fixed32.json').read_bytes())['cases']
emitter=R/'results'/('GLM-latest-full-owner-r290-'+label+'-stage3-rank0.json');owner=json.loads(emitter.read_bytes());out=[]
for i,row in enumerate(suite['rows']):
 if not row['cap_hit']:continue
 d=R/'results'/('GLM-latest-full-owner-r290-client-'+label)/'quality/MMLU'/('%02d'%i)
 wire=json.loads((d/'request-001.wire').read_bytes());body=json.loads((d/'request-001.request.json').read_bytes());c=cases[i]
 expected=c['question']+'\n'+'\n'.join('%s. %s'%(chr(65+k),v)for k,v in enumerate(c['choices']))
 assert body['messages'][-1]['content']==expected and body['max_tokens']==4096
 r=owner['service_requests'][36+i];choice=wire['choices'][0];m=choice['message']
 assert r['request_id']==wire['id']and r['actual_generated_tokens']==4096 and r['native_finish_reason']=='length'
 assert choice['finish_reason']=='length'and letter(m.get('content')or'')is None
 out.append({'offset':i,'source_index':c['source_index'],'subject':c.get('subject'),'admitted_index':36+i,
   'wire_SHA256':sha(d/'request-001.wire'),'request_SHA256':sha(d/'request-001.request.json'),
   'Native_tokens':4096,'Native_and_transport_finish':'length','requested_budget':4096,
   'requested_template_flags':body['chat_template_kwargs'],'content_characters':len(m.get('content')or''),
   'reasoning_characters':len(m.get('reasoning_content')or''),'final_answer_parsed':False,
   'actual_prompt_tokens':r['prompt_tokens'],'history_generation_audit':{k:r['admission']['history_render_audit'].get(k)for k in ['generation_thinking','pending_thinking_prefix_open']}})
assert len(out)==2
print(json.dumps({'read_only':True,'actual_GPU':'910376.819','supervisor_SHA256':sha(p),
 'emitter_SHA256':sha(emitter),'caps':out,'scope':'Actual4096tokenbudgetexhaustion;notfinalJSONparserfailure;no8192counterfactualtested','new_jobs':0}))
