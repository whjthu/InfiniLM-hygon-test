"""CPU geometry and complete-history token accounting; never a model result."""
import hashlib,json,math
from chat_prepare_r173 import prepare
TOOLS=[{'type':'function','function':{'name':'read_file','description':'Read the requested JSON data file.','parameters':{'type':'object','properties':{'path':{'type':'string'}},'required':['path'],'additionalProperties':False}}}]
def geometry(capacity,rows):
 assert capacity in [1024,2048,32768,65536]and rows in [1,2,128]
 pages=lambda slots,block:(slots+block-1)//block
 stride=lambda block:(block*584+575)//576*576
 hcap=(capacity+127)//128;ccap=(capacity+3)//4
 hpages=pages(rows*hcap,2);cpages=pages(rows*ccap,64);ipages=pages(rows*ccap,64);rpages=pages(rows*128,256)
 dependent=20*hpages*stride(2)+21*cpages*stride(64)+21*ipages*8448
 ring=43*rpages*stride(256)
 hstate=20*rows*128*2*512*2;cstate=21*rows*8*4*512*2;istate=21*rows*8*4*128*2
 return dict(capacity=capacity,rows=rows,HCA_compressed_entries_per_row=hcap,CSA_and_index_entries_per_row=ccap,HCA_pages_per_layer=hpages,CSA_pages_per_layer=cpages,index_pages_per_layer=ipages,ring_pages_per_layer=rpages,capacity_dependent_bytes_per_owner=dependent,ring_bytes_per_owner=ring,compressor_states_bytes_per_owner=hstate+cstate+istate,total_listed_persistent_GPU_bytes_per_owner=dependent+ring+hstate+cstate+istate,source_page_limits_admissible=all(0<x<=65536 for x in [hpages,cpages,ipages,rpages])and hcap<=8192,exclusions='Model weights, allocator overhead, small flags/APE/norm constants, temporary projection/attention/indexer/expert tensors and runtime/collective scratch. These are source byte formulas, not observed free memory.',runtime_qualification='UNPROVEN; source admissibility does not establish long-context task correctness or peak memory')
def contiguous_prefill(total,chunk=1024):
 assert type(total)is int and total>0 and 1<=chunk<=4096
 plan=[dict(past=p,tokens=min(chunk,total-p),total=min(p+chunk,total),row=0,emit_sample=(p+chunk>=total))for p in range(0,total,chunk)]
 assert sum(x['tokens']for x in plan)==total and all(x['tokens']<=4096 for x in plan)
 assert [x['past']for x in plan]==[sum(q['tokens']for q in plan[:i])for i in range(len(plan))]
 assert sum(x['emit_sample']for x in plan)==1
 return plan
def history(lines,thinking=False):
 messages=[{'role':'system','content':'Review complete data using tools. Preserve prior results and do not invent observations.'},{'role':'user','content':'Perform eight reads, retain all earlier tool results, then summarize the eight markers and the logged data.'}]
 for i in range(8):
  call={'id':'fixture_call_'+str(i),'type':'function','function':{'name':'read_file','arguments':json.dumps({'path':'input.json'})}}
  result={'marker':'TOOL_HISTORY_MARKER_'+str(i),'iteration':i,'log':lines if i==0 else ['complete-small-result-'+str(i)]}
  messages += [{'role':'assistant','content':'','tool_calls':[call]},{'role':'tool','tool_call_id':call['id'],'content':json.dumps(result,separators=(',',':'))}]
 return messages,{'enable_thinking':thinking,'tools':TOOLS}
def actual_token_count_cases(tokenizer):
 cases=[]
 specs=[('input8K_out4K',8192,32768,4096,False,True),('input16Kplus_out4K',16384,32768,4096,False,True),('input16Kplus_thinking8K',16384,32768,8192,True,True),('input24Kplus_out4K',24576,32768,4096,False,True),('input40Kplus_extension64K_thinking8K',40960,65536,8192,True,True),('input32Kplus_out4K_rejected_at32K',32768,32768,4096,False,False)]
 for label,target,capacity,limit,thinking,admit in specs:
  lines=[];prompt='';ids=[]
  while len(ids)<target:
   begin=len(lines)
   lines.extend('row=%06d value=%s'%(j,hashlib.sha256(('r177-synthetic-tool-data-'+str(j)).encode()).hexdigest())for j in range(begin,begin+32))
   messages,kwargs=history(lines,thinking);prompt,ids=prepare(messages,{},True,kwargs,tokenizer)
  # These are synthetic assistant tool-history fixtures, never model-generated calls.
  assert len([x for x in messages if x['role']=='tool'])==8
  assert all('TOOL_HISTORY_MARKER_'+str(i)in prompt for i in range(8))
  accepted=True
  try:prepare(messages,{},True,kwargs,tokenizer,capacity,limit)
  except ValueError:accepted=False
  assert accepted==admit,(label,len(ids),limit,capacity)
  plan=contiguous_prefill(len(ids));body={'messages':messages,'tools':TOOLS,'max_tokens':limit,'chat_template_kwargs':kwargs}
  cases.append(dict(label=label,synthetic_fixture_only=True,actual_tokenizer_prompt_tokens=len(ids),requested_output_budget=limit,context_capacity=capacity,prompt_plus_output=len(ids)+limit,accepted_by_complete_history_guard=accepted,all8_tool_results_preserved=True,prompt_SHA256=hashlib.sha256(prompt.encode()).hexdigest(),token_ids_SHA256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),JSON_wire_body_bytes=len(json.dumps(body,ensure_ascii=False).encode()),continuous_prefill_plan=plan,per_call_max_tokens=1024,intermediate_chunk_sample_rule='Do not emit/feed sampled IDs from intermediate prefill chunks; emit only final prefill sample, then AR continuation at full prompt past. No reset or token drop between chunks.',HTTP_or_Native_execution='NOT_RUN'))
 return cases
