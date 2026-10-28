"""Observe actual prepared prompt with the same official normalizer/renderer.

RC spans are dedicated serialized contributions, not global substring counts.
No Model call, mutation, alternate format or shared monkeypatch is used.
"""
import copy,hashlib,json
sha=lambda text:hashlib.sha256(text.encode()).hexdigest()
def content_parts(message):
 blocks=message.get('content_blocks')if message.get('role')=='user'else None
 if not blocks:return [(message.get('content')or'','','')]
 result=[]
 for block in blocks:
  if block.get('type')=='text':result.append((block.get('text',''),'',''))
  elif block.get('type')=='tool_result':
   value=block.get('content','')
   if not isinstance(value,str):raise ValueError('History audit is qualified for standard text tool results only')
   result.append((value,'<tool_result>','</tool_result>'))
  else:raise ValueError('History audit is qualified for standard text message blocks only')
 return result
def audit(encoder,prepared,mode,effort,prompt,generation=True):
 normalized=encoder.merge_tool_messages(copy.deepcopy(prepared));normalized=encoder.sort_tool_results_by_call_order(normalized)
 fragments=[encoder.render_message(i,normalized,thinking_mode=mode,drop_thinking=False,reasoning_effort=effort,preserve_history_reasoning=True)for i in range(len(normalized))]
 expected=encoder.bos_token+''.join(fragments)
 if not generation:expected=expected.removesuffix(encoder.ASSISTANT_SP_TOKEN+(encoder.thinking_start_token if mode=='thinking'else encoder.thinking_end_token))
 source_rc=[m['reasoning_content']for m in prepared if m.get('role')=='assistant'and m.get('reasoning_content')];normalized_rc=[m['reasoning_content']for m in normalized if m.get('role')=='assistant'and m.get('reasoning_content')];spans=[];offset=len(encoder.bos_token);qualified=True;content_spans=[]
 source_content=[sha(text)for m in prepared for text,_,_ in content_parts(m)if text];normalized_content=[sha(text)for m in normalized for text,_,_ in content_parts(m)if text]
 for i,(m,fragment)in enumerate(zip(normalized,fragments)):
  rc=m.get('reasoning_content')if m.get('role')=='assistant'else None
  if rc:
   opened=i>0 and fragments[i-1].endswith(encoder.ASSISTANT_SP_TOKEN+encoder.thinking_start_token);closed=fragment.startswith(rc+encoder.thinking_end_token);start=offset;end=start+len(rc);same=prompt[start:end]==rc
   qualified=qualified and opened and closed and same
   spans.append({'message_index':i,'start_char':start,'end_char':end,'RC_SHA256':sha(rc),'opening_correct':opened,'closing_correct':closed,'actual_prompt_slice_equal':same})
  lead=len(encoder.REASONING_EFFORT_MAX)if i==0 and mode=='thinking'and effort=='max'else 0;role=m.get('role')
  if role in ['user','developer']:lead+=len(encoder.USER_SP_TOKEN)
  elif role=='latest_reminder':lead+=len(encoder.LATEST_REMINDER_SP_TOKEN)
  elif role=='assistant':
   previous_task=i>0 and normalized[i-1].get('task')is not None
   if(mode=='thinking'or rc)and not previous_task:lead+=len(rc or'')+len(encoder.thinking_end_token)
  parts=content_parts(m)
  for j,(text,prefix,suffix)in enumerate(parts):
   lead+=len(prefix);start=offset+lead;end=start+len(text)
   if text:content_spans.append({'SHA256':sha(text),'start':start,'end':end,'equal':prompt[start:end]==text})
   lead+=len(text)+len(suffix)+(2 if j+1<len(parts)else 0)
  offset+=len(fragment)
 sequence=[sha(x)for x in source_rc];normalized_sequence=[sha(x)for x in normalized_rc];render_sequence=[x['RC_SHA256']for x in spans];order=sequence==normalized_sequence==render_sequence;distinct=all(a['end_char']<=b['start_char']for a,b in zip(spans,spans[1:]));content_order=source_content==normalized_content==[x['SHA256']for x in content_spans];content_equal=all(x['equal']for x in content_spans);qualified=qualified and order and distinct and expected==prompt and content_order and content_equal
 return {'scope':'This actual standard-role request only; dedicated RC/content contribution spans, not arbitrary-history or global text occurrence proof','qualified':qualified,'actual_prompt_SHA256':sha(prompt),'official_normalized_render_equals_actual_prompt':expected==prompt,'input_RC_fields':len(source_rc),'normalized_RC_fields':len(normalized_rc),'rendered_RC_contributions':len(spans),'source_RC_SHA256_sequence':sequence,'RC_order_preserved':order,'RC_contributions_nonoverlapping':distinct,'RC_one_contribution_per_source_field':order and len(spans)==len(source_rc),'RC_spans':spans,'input_content_parts':len(source_content),'rendered_content_contributions':len(content_spans),'content_one_contribution_each_and_ordered':content_order,'all_content_contribution_slices_equal':content_equal,'content_sequence_SHA256':sha(json.dumps(source_content,separators=(',',':'))),'normalized_messages':len(normalized),'generation_thinking':mode=='thinking','pending_chat_prefix_closed':prompt.endswith(encoder.ASSISTANT_SP_TOKEN+encoder.thinking_end_token),'pending_thinking_prefix_open':prompt.endswith(encoder.ASSISTANT_SP_TOKEN+encoder.thinking_start_token)}
