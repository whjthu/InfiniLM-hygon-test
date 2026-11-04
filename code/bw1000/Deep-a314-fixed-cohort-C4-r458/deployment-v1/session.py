"""New device glue: one real Context, exact six cohorts, append-only audit."""
import ctypes as C
import hashlib,json,os,time
from pathlib import Path
import numpy as np
from fixed_cohort.native_context import FixedCohortContext
from forward_policy0 import make_forward
from precision import select_original
from fixed_cohort.entry import MODES
from fixed_cohort.continuation import run
from scoring import grade
from public_parser.tool_parsers import ToolCallParser


def intp(value):return value.value if isinstance(value,C.c_void_p)else int(value)


class Journal:
 def __init__(self,path):
  self.f=Path(path).open('x');self.writes=0;self.bytes=0;self.seconds=0.
 def put(self,event,value):
  t=time.perf_counter();s=json.dumps(dict(event=event,**value),allow_nan=False,separators=(',',':'))+'\n'
  self.f.write(s);self.f.flush();self.writes+=1;self.bytes+=len(s.encode());self.seconds+=time.perf_counter()-t
 def close(self):self.f.flush();os.fsync(self.f.fileno());self.f.close()


class NativeAudit:
 """Scoped CAPI facade; same actual C function, no shared/global patch."""
 def __init__(self,lib,journal,vocab,memory):self.original=lib;self.journal=journal;self.vocab=vocab;self.attempted=0;self.completed=0;self.memory=memory;self.min_free=[None]*8
 def __getattr__(self,name):return getattr(self.original,name)
 def modern_native_engine_forward(self,ctx,tokens,count,*pointers):
  ps=[intp(p)for p in pointers];tokens=int(tokens);count=int(count)
  rows=list((C.c_int32*count).from_address(ps[0]));past=list((C.c_int32*count).from_address(ps[1]));total=list((C.c_int32*count).from_address(ps[2]));offsets=list((C.c_int32*(count+1)).from_address(ps[3]))
  ids=np.ctypeslib.as_array((C.c_int64*tokens).from_address(ps[4]));positions=np.ctypeslib.as_array((C.c_int64*tokens).from_address(ps[5]))
  before_ids=ids.copy();before_positions=positions.copy()
  identity=dict(index=self.attempted,rows=rows,past=past,total=total,offsets=offsets,tokens=tokens,count=count,
    IDs_SHA256=hashlib.sha256(ids.tobytes()).hexdigest(),positions_SHA256=hashlib.sha256(positions.tobytes()).hexdigest(),
    decode_input_IDs=ids.tolist()if tokens==count else None,position_endpoints=[int(positions[0]),int(positions[-1])])
  self.journal.put('CAPI_attempt',identity);self.attempted+=1;start=time.perf_counter()
  code=self.original.modern_native_engine_forward(ctx,tokens,count,*pointers);end=time.perf_counter();self.completed+=1
  logits=np.ctypeslib.as_array((C.c_uint16*(count*self.vocab)).from_address(ps[6]));sampled=list((C.c_int64*count).from_address(ps[7]))
  counters=[]
  for d in range(8):
   counters.append(dict(device=d,FP32=int(self.original.modern_deep_FP32_calls_r3597(d)),
    HC=[int(self.original.deep_hc_canonical_calls_r7130(d,k))for k in range(3)],
    c4_tile=[int(self.original.modern_deep_c4_tile_calls_r5920(d,k))for k in range(2)],
    layout=[int(self.original.modern_deep_layout_calls_r7052(d,k))for k in range(4)],
    tail=[int(self.original.deep_moe_tail_calls_r7211(d,k))for k in range(5)]))
  # Raw returned status/buffers/counters durable before original forward wrapper
  # applies its strict all8 dispatch assertions. No error is hidden by a counter.
  memory=self.memory();readonly=bool(np.array_equal(before_ids,ids)and np.array_equal(before_positions,positions))
  for i,pair in enumerate(memory):self.min_free[i]=pair[0]if self.min_free[i]is None else min(self.min_free[i],pair[0])
  self.journal.put('CAPI_return',dict(**identity,rc=int(code),CAPI_start_monotonic=start,CAPI_return_monotonic=end,
    CAPI_seconds=end-start,sampled_IDs=sampled,logits_dtype='BF16',logits_shape=[count,self.vocab],
    BF16_nonfinite_count=int(np.count_nonzero((logits&0x7f80)==0x7f80)),
    logits_SHA256=hashlib.sha256(logits.tobytes()).hexdigest(),all8_counters=counters,input_readonly=readonly,GPU_free_total_after_CAPI=memory))
  assert readonly,'Native mutated owned input buffers'
  assert len(memory)==8 and all(f>=16*1024**3 for f,t in memory),'Original all8 postload floor'
  return code


def score_state(record,state,tokenizer):
 ids=state['generated_ids'];natural=state['finish']=='EOS';cap=state['finish']=='cap'
 # Same original bridge convention: EOS omitted from text, special content
 # preserved for the actual public parser; no Unicode/answer cherry picking.
 text=tokenizer.decode([t for t in ids if t!=1],skip_special_tokens=False,clean_up_tokenization_spaces=False)
 parsed=dict(content='',reasoning_content='',tool_calls=[]);error=None
 try:
  parser=ToolCallParser({},thinking=False,choice='none',parallel=False,parser='deepseek_v4')
  for event in parser.feed(text)+parser.feed('',final=True,finish_reason='stop'if natural else'length'):
   parsed['content']+=event.get('content')or'';parsed['reasoning_content']+=event.get('reasoning_content')or'';parsed['tool_calls']+=event.get('tool_calls',[])
 except (ValueError,TypeError,KeyError)as e:error=repr(e)
 score=grade(record['gold'],parsed['content'],natural_EOS=natural,cap_hit=cap,unexpected_tool=bool(parsed['tool_calls']))
 if error:score.update(correct=False,parse_failure=True,parser_error=error)
 visible=[e['monotonic']for e in state['token_events']if e['visible']]
 itl=[b-a for a,b in zip(visible,visible[1:])]
 metric=dict(visible_tokens=len(visible),tokens_including_EOS=len(ids),natural_EOS=natural,cap=cap,
   first_visible_monotonic=visible[0]if visible else None,last_visible_monotonic=visible[-1]if visible else None,
   decode_tok_s=(len(visible)-1)/(visible[-1]-visible[0])if len(visible)>1 and visible[-1]>visible[0]else None,
   ITL=itl,ITL_P95=float(np.quantile(itl,.95))if itl else None,ITL_P99=float(np.quantile(itl,.99))if itl else None)
 return dict(raw_decoded=text,parsed=parsed,score=score,metrics=metric)


def run_session(prepared,config,*,binding,torch,safe_open,tokenizer,output,report,save,deadline_check,canceled=lambda:False):
 output=Path(output);journal=Journal(output/'calls-and-events-private.jsonl');audit=None
 try:
  report['precision_policy']=select_original(binding.lib,prepared['receipt']['library'])
  with FixedCohortContext(binding,prepared['receipt'],config=prepared['config'],cfg=prepared['cfg'],mapping=prepared['mapping'],torch=torch,safe_open=safe_open,report=report,deadline_check=deadline_check)as context:
   assert report['loaded_key_count']==67569
   audit=NativeAudit(binding.lib,journal,prepared['cfg']['vocab_size'],lambda:[list(torch.cuda.mem_get_info(i))for i in range(8)])
   context.forward=make_forward(audit,context.ctx,context.cfg,report,binding.check)
   report.update(actual_rows=4,capacity_per_row=32768,phase='loaded_rows4',fixed_cohort_trials=[]);save()
   for ordinal,mode in enumerate(MODES):
    journal.put('cohort_begin',dict(ordinal=ordinal,mode=mode,warm=ordinal<2))
    trial_start=time.perf_counter();seen_calls=0;seen_tokens=[0]*4
    def progress(phase,value):
     nonlocal seen_calls
     if len(value['calls'])>seen_calls:
      for item in value['calls'][seen_calls:]:journal.put('caller_return',dict(ordinal=ordinal,**item))
      seen_calls=len(value['calls'])
     for row,state in enumerate(value['states']):
      for e in state['token_events'][seen_tokens[row]:]:journal.put('token_emitted',dict(ordinal=ordinal,row=row,**e))
      seen_tokens[row]=len(state['token_events'])
     if phase in ('failed_before_drain','terminal'):journal.put('cohort_boundary',dict(ordinal=ordinal,phase=phase,failed=value['failed'],normal=value.get('normal'),reset_after=value['Native_reset_after'],error=value.get('error')))
     report['progress']=dict(ordinal=ordinal,mode=mode,phase=phase,completed_calls=seen_calls,emitted_tokens=list(seen_tokens))
     if phase in ('cohort_start','terminal','failed_before_drain'):save()
    trial=run(context,prepared['records'],mode=mode,deadline=config['gpu_start_epoch']+5280,save=progress,canceled=canceled)
    trial.update(ordinal=ordinal,warm=ordinal<2,quality=[])
    for record,state in zip(prepared['records'],trial['states']):
     row=score_state(record,state,tokenizer);row.update(case_id=record['case_id'],row=record['row'],suite=record['gold']['suite'])
     row['metrics']['TTFT_s']=None if row['metrics']['first_visible_monotonic']is None else row['metrics']['first_visible_monotonic']-trial['all_arrival_monotonic'];trial['quality'].append(row)
    trial['whole_trial_seconds_including_decode_grade']=time.perf_counter()-trial_start
    trial['useful_tokens']=sum(v['metrics']['visible_tokens']for v in trial['quality'])
    trial['engine_emitter_tok_s']=trial['useful_tokens']/trial['whole_cohort_seconds']
    trial['engine_emitter_tok_s_per_card']=trial['engine_emitter_tok_s']/8
    publish=output/('cohort-%d-private.json'%ordinal)
    with publish.open('x')as f:json.dump(trial,f,allow_nan=False);f.write('\n')
    report['fixed_cohort_trials'].append(dict(ordinal=ordinal,mode=mode,warm=ordinal<2,path=str(publish),SHA256=hashlib.sha256(publish.read_bytes()).hexdigest(),
       whole_cohort_seconds=trial['whole_cohort_seconds'],whole_trial_seconds_including_decode_grade=trial['whole_trial_seconds_including_decode_grade'],
       useful_tokens=trial['useful_tokens'],engine_emitter_tok_s_per_card=trial['engine_emitter_tok_s_per_card'],normal=trial['normal'],quality=[dict(case_id=v['case_id'],suite=v['suite'],score=v['score'],metrics=v['metrics'])for v in trial['quality']]))
    report['completed_cohorts']=len(report['fixed_cohort_trials']);report['logical_requests']=4*report['completed_cohorts'];report['attempted_CAPI']=audit.attempted;report['completed_CAPI']=audit.completed;save()
    if canceled():raise RuntimeError('Original residency canceled; no new cohort')
   report['owner_affinity_after_cohorts']=owner_affinity(binding.lib);report['memory_after_cohorts']=context.floor();save()
  report['all_cohorts_completed_normal']=len(report['fixed_cohort_trials'])==6 and all(v['normal']for v in report['fixed_cohort_trials'])
  report['summary']=summary(report['fixed_cohort_trials']);return report
 finally:
  if audit is not None:report.update(attempted_CAPI=audit.attempted,completed_CAPI=audit.completed,sampled_min_free_per_GPU=audit.min_free,memory_scope='all8 driver free/total after every CAPI; not a continuous inside-kernel peak')
  journal.close();report['journal']=dict(path=str(output/'calls-and-events-private.jsonl'),SHA256=hashlib.sha256((output/'calls-and-events-private.jsonl').read_bytes()).hexdigest(),writes=journal.writes,bytes=journal.bytes,append_flush_seconds=journal.seconds)


def owner_affinity(lib):
 from fixed_cohort.model_affinity_check_r2628 import verify
 return verify(lib)


def summary(trials):
 by={v['ordinal']:v for v in trials};pairs=[]
 for off,on in [(2,3),(5,4)]:
  if off in by and on in by:pairs.append(dict(serial_ordinal=off,cohort_ordinal=on,whole_cohort_speedup=by[off]['whole_cohort_seconds']/by[on]['whole_cohort_seconds'],serial_useful_tokens=by[off]['useful_tokens'],cohort_useful_tokens=by[on]['useful_tokens']))
 return dict(formal_cohorts=sum(not v['warm']for v in trials),formal_logical_requests=4*sum(not v['warm']for v in trials),pairs=pairs,
   serial_endpoint_change=(by[5]['whole_cohort_seconds']/by[2]['whole_cohort_seconds']-1)if 5 in by and 2 in by else None,
   qualitywrong_or_parse_or_cap_not_early_stop=True,online_SLO=False,clock_scope='same-process engine-emitter return clock; not HTTP/clientSSE',Model_quality_noninferiority_proven=False)
