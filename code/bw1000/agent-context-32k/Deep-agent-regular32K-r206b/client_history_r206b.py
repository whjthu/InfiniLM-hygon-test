"""Two real continuation requests from this same run's completed thinking flow.

Never fabricates an assistant/tool result, shortens history or changes the
frozen matrix. Summary contains counts/booleans/SHA, never raw RC/nonce.
"""
import argparse,copy,hashlib,importlib.util,json,pathlib,time
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def source_history(regular):
 result=json.loads((regular/'result.json').read_bytes());case=next(c for c in result['cases']if c['name']=='thinking');assert result['passed']and case['passed']and case['real_tool_steps']==8
 request=regular/'thinking/request-009.request.json';decoded=regular/'thinking/request-009.decoded.json';body=json.loads(request.read_bytes());answer=json.loads(decoded.read_bytes());assert body['tool_choice']=='none'and body['chat_template_kwargs']['enable_thinking']is True and answer['finish_reason']=='stop'and not answer['message'].get('tool_calls')
 messages=copy.deepcopy(body['messages'])+[copy.deepcopy(answer['message'])]
 messages.append({'role':'user','content':'Continue this conversation using its actual prior tool results. Reply with the verified nonce and total from the final file, without calling any tool.'})
 expected=json.loads((regular/'thinking/workspace/final.json').read_bytes());assert isinstance(expected['nonce'],str)and type(expected['total'])is int
 rc=[hashlib.sha256(m['reasoning_content'].encode()).hexdigest()for m in messages if m.get('role')=='assistant'and m.get('reasoning_content')];assert rc,'A genuine prior thinking response is required, never synthesize one'
 return messages,expected,rc,{'regular_result_SHA256':sha(regular/'result.json'),'last_actual_request_SHA256':sha(request),'last_actual_response_SHA256':sha(decoded),'verified_file_SHA256':sha(regular/'thinking/workspace/final.json')}
def actual_request(service,user):
 deadline=time.monotonic()+8
 while time.monotonic()<deadline:
  q=json.loads(service.read_bytes());found=[r for r in q['requests']if r.get('admission',{}).get('diagnostic_user')==user]
  if len(found)==1 and found[0].get('final_metadata'):return found[0]
  assert len(found)<=1;time.sleep(.05)
 raise RuntimeError('Actual owner request/audit journal not ready')
def main():
 p=argparse.ArgumentParser()
 for name in ['p0-client','p0-sha256','base-url','model','output-dir','identity','regular-dir','service-report']:p.add_argument('--'+name,required=True)
 p.add_argument('--timeout',type=float,default=7200);x=p.parse_args();out=pathlib.Path(x.output_dir);out.mkdir(exist_ok=False);report={'completed':False,'passed':False,'scope':'Same-residency real chat/thinking continuation from the frozen complete8tool thinking conversation; source-specific RC contribution audit, no universal history claim','cases':[],'fixed_matrix_unchanged':True}
 try:
  assert sha(x.p0_client)==x.p0_sha256;spec=importlib.util.spec_from_file_location('history_real_wire_r206b',x.p0_client);wire=importlib.util.module_from_spec(spec);spec.loader.exec_module(wire)
  identity=json.loads(pathlib.Path(x.identity).read_bytes());assert identity['capacity']==32768 and identity['rows']==1 and identity['Native_SHA256']=='232fb7e2566f7d1961fa433d9e45660bfe760ec726676dcb7f1f406d857c801f';messages,expected,rc,source=source_history(pathlib.Path(x.regular_dir));report.update(source=source,input_RC_fields=len(rc),identity_SHA256=sha(x.identity))
  for thinking,budget,label in [(False,4096,'chat'),(True,8192,'thinking')]:
   folder=out/label;folder.mkdir();client=wire.WireClient(x.base_url,x.model,folder,budget,x.timeout);client.session='P1-history-'+label+'-r206b';before=copy.deepcopy(messages)
   message,finish,usage,event=client.request(messages,'none',False,None,max_tokens=budget,thinking=thinking);assert messages==before
   assert event['HTTP_status']==200 and finish=='stop'and not message.get('tool_calls');content=message.get('content')or'';recall=expected['nonce']in content and str(expected['total'])in content
   owner=actual_request(pathlib.Path(x.service_report),client.session);audit=owner['admission']['history_render_audit'];assert audit['qualified']and audit['source_RC_SHA256_sequence']==rc and audit['input_RC_fields']==audit['rendered_RC_contributions']==len(rc)and audit['RC_one_contribution_per_source_field']and audit['RC_order_preserved']and audit['RC_contributions_nonoverlapping']and audit['official_normalized_render_equals_actual_prompt']
   assert audit['pending_thinking_prefix_open']==thinking and audit['pending_chat_prefix_closed']==(not thinking)and audit['generation_thinking']==thinking
   assert audit['content_one_contribution_each_and_ordered']and audit['all_content_contribution_slices_equal']and audit['input_content_parts']==audit['rendered_content_contributions']
   assert owner['normal_reset_before']and owner['normal_reset_after']and owner['native_finish_reason']=='eos_token'and owner['max_new_tokens']==budget and owner['prompt_tokens']==usage['prompt_tokens']and usage['prompt_tokens']+budget<=32768
   assert sum(r['tokens']for r in owner['Native_continuation']if r['kind']=='prefill')==usage['prompt_tokens']
   row={'mode':label,'passed':recall,'natural_EOS':True,'actual_generated_tokens':owner['actual_generated_tokens'],'prompt_tokens':usage['prompt_tokens'],'output_budget':budget,'verified_nonce_and_total_recalled':recall,'historical_RC_fields':len(rc),'RC_one_contribution_each_and_ordered':True,'content_one_contribution_each_and_ordered':True,'source_specific_RC_audit_qualified':True,'generation_prefix_correct':True,'normal_resets':True,'prepared_prompt_SHA256':audit['actual_prompt_SHA256'],'Native_prompt_token_ids_SHA256':owner['admission']['Native_prompt_token_ids_SHA256'],'Native_committed_all_prompt_tokens':True,'request_SHA256':sha(folder/'request-001.request.json'),'wire_SHA256':event['wire_SHA256'],'message_SHA256':hashlib.sha256(wire.compact(message).encode()).hexdigest(),'new_response_has_reasoning':bool(message.get('reasoning_content'))};report['cases'].append(row)
  report.update(completed=True,passed=len(report['cases'])==2 and all(r['passed']for r in report['cases']),actual_HTTP_requests=2)
 except BaseException as e:report.update(completed=True,passed=False,error_class=type(e).__name__,error_SHA256=hashlib.sha256(str(e).encode()).hexdigest())
 (out/'result.json').write_bytes((json.dumps(report,indent=2)+'\n').encode());print(json.dumps(report,separators=(',',':')));return 0 if report['passed']else 1
if __name__=='__main__':raise SystemExit(main())
