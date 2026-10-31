"""Frozen public GSM8K/MMLU pilots; no question truncation, no answer-conditioned selection."""
import argparse,copy,json,pathlib,re,time,hashlib,math
from fractions import Fraction
from source_pin_r281 import load_wire,verify,sha,D
from client_owner_r281 import bind_identity,row,backend_metrics,wilson

def numeric(text):
 v=re.findall(r'####[ *$]*([-+]?(?:[0-9][0-9,]*(?:[.][0-9]+)?|[.][0-9]+)(?:/[0-9]+)?)',text)
 try:return Fraction(v[-1].replace(',',''))if v else None
 except(ValueError,ZeroDivisionError):return None

def letter(text):
 v=re.findall(r'####\s*([ABCD])\b',text);return ord(v[-1])-65 if v else None

def main():
 p=argparse.ArgumentParser()
 for n in ['base-url','model','output-dir','identity','service-report','manifest-sha','expected-step']:p.add_argument('--'+n,required=True)
 p.add_argument('--deadline-unix',type=float,required=True);p.add_argument('--timeout',default=7200,type=float);args=p.parse_args();out=pathlib.Path(args.output_dir);out.mkdir(exist_ok=False);q={'completed':False,'passed':False,'scope':'Two reused fixed32 public pilots, custom0shot ordinary4096; not full benchmark/no-regression proof','no_retries':True,'max_HTTP':64,'full_benchmark':False,'paired_quality_delta':None,'no_regression_proven':False,'suites':{},'client_incremental_metrics_qualified':False}
 def save():(out/'result.json').write_text(json.dumps(q,indent=2)+chr(10))
 def remaining():
  left=args.deadline_unix-time.time()
  if left<=120:raise TimeoutError('Original residency deadline exhausted')
  return left
 save()
 try:
  remaining();identity=bind_identity(args.identity,args.manifest_sha,args.service_report);assert identity['actual_step']==args.expected_step;wire=load_wire(args.manifest_sha);mf=verify(args.manifest_sha);q['identity_SHA256']=sha(args.identity)
  for suite_index,(name,filename)in enumerate([('GSM8K','gsm8k-fixed32.json'),('MMLU','mmlu-fixed32.json')]):
   f=D/'benchmark'/filename;assert sha(f)==mf['files_SHA256']['benchmark/'+filename];suite=json.loads(f.read_bytes());cases=suite['cases'];assert len(cases)==32 and suite['pilot_not_full_benchmark'];folder=out/name;folder.mkdir();rows=[];q['suites'][name]={'fixture_SHA256':sha(f),'pilot_not_full_benchmark':True,'selection':suite['selection'],'seed':suite['seed'],'minimum_correct_coarse_screen':24,'rows':rows};save()
   for offset,case in enumerate(cases):
    remaining();where=folder/('%02d'%offset);where.mkdir();client=wire.WireClient(args.base_url,args.model,where,4096,min(args.timeout,remaining()-60));client.session='P1-Deep281-'+name+'-%02d'%offset
    if name=='GSM8K':messages=[{'role':'system','content':'Solve this math problem accurately. Explain your work and end with #### followed by your numeric answer.'},{'role':'user','content':case['question']}]
    else:messages=[{'role':'system','content':'Answer the multiple choice question accurately. Explain your choice and end with #### followed by exactly one of A, B, C, D.'},{'role':'user','content':case['question']+chr(10)+chr(10).join('%s. %s'%(chr(65+i),v)for i,v in enumerate(case['choices']))}]
    before=copy.deepcopy(messages);message,finish,usage,event=client.request(messages,'none',False,None,max_tokens=4096,thinking=False);assert messages==before and event['HTTP_status']==200 and finish in ['stop','length']and not message.get('tool_calls');request=json.loads((where/'request-001.request.json').read_bytes());assert request['messages']==messages and request['max_tokens']==4096 and request['chat_template_kwargs']['enable_thinking']is False
    raw=json.loads((where/'request-001.wire').read_bytes());index=38+suite_index*32+offset;r=row(pathlib.Path(args.service_report),args,raw['id'],index,client.session,4096,remaining);assert r['prompt_tokens']==usage['prompt_tokens'];natural=r['native_finish_reason']=='eos_token';assert(finish=='stop')==natural;content=message.get('content')or'';value=numeric(content)if name=='GSM8K'else letter(content);gold=Fraction(case['gold_numeric'].replace(',',''))if name=='GSM8K'else case['answer'];correct=bool(value is not None and value==gold and natural)
    metrics=backend_metrics(r);rows.append(dict(source_index=case['source_index'],question_SHA256=hashlib.sha256(case['question'].encode()).hexdigest(),subject=case.get('subject'),correct=correct,parse_failure=value is None,natural_EOS=natural,cap_hit=not natural,output_budget=4096,generated_tokens=r['actual_generated_tokens'],prompt_tokens=r['prompt_tokens'],request_SHA256=sha(where/'request-001.request.json'),wire_SHA256=sha(where/'request-001.wire'),client_HTTP_wall_s=event['elapsed_s'],**metrics));save()
   n=len(rows);k=sum(v['correct']for v in rows);q['suites'][name].update(completed=True,count=n,correct=k,accuracy=k/n,Wilson95=wilson(k,n),coverage=n/32,natural_EOS=sum(v['natural_EOS']for v in rows),caps=sum(v['cap_hit']for v in rows),parse_failures=sum(v['parse_failure']for v in rows),coarse_screen_pass=k>=24 and all(v['natural_EOS']for v in rows));save()
  q['all64_requests_completed']=True;q['passed']=all(v['coarse_screen_pass']for v in q['suites'].values());q['scores_do_not_prove_no_regression']=True
 except BaseException as e:q['error']=repr(e)
 finally:q['completed']=True;save()
 return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
