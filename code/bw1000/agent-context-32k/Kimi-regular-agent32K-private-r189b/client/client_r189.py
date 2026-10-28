"""One frozen regular suite followed by long-output suite. No retries/cropping."""
import argparse,hashlib,importlib.util,json,pathlib,sys,time
D=pathlib.Path(__file__).resolve().parent;RUNTIME=D.parent/'runtime';sys.path.insert(0,str(RUNTIME))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def load(path,name,expected):
 assert sha(path)==expected
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def main():
 p=argparse.ArgumentParser()
 for x in ['base-url','model','output-dir','identity']:p.add_argument('--'+x,required=True)
 p.add_argument('--timeout',type=float,default=3600);a=p.parse_args();o=pathlib.Path(a.output_dir);o.mkdir(exist_ok=False)
 identity=pathlib.Path(a.identity);report={'completed':False,'passed':False,'identity_SHA256':sha(identity),'no_retries':True,'no_cropping':True,'output_budgets':[4096,4096,8192,4096],'multicall':'NOT_COVERED_BY_THIS_REGULAR_SUITE'}
 def save():(o/'result.json').write_text(json.dumps(report,indent=2)+'\n')
 save()
 try:
  spec=json.loads((D/'client-launch-r189.json').read_bytes())
  p0=load(D/'client_p0_r143.py','r189_p0',spec['p0_SHA256'])
  regular=load(D/'client_regular_agent.py','frozen_regular',spec['regular_SHA256'])
  output=load(D/'client_long_output.py','frozen_long_output',spec['long_output_SHA256'])
  counter=load(RUNTIME/'ordinary_token_counter_r179.py','ordinary_counter',spec['counter_SHA256'])
  profiles=pathlib.Path('/refactor/candidates/Kimi-long-agent32K-r179/regular-agent-profiles-r179/profiles.json')
  receipt=profiles.parent/'CPU-profile-receipt-r179.json';assert sha(receipt)==spec['CPU_profile_receipt_SHA256']
  q=json.loads(receipt.read_bytes());rows=json.loads(profiles.read_bytes());assert rows==q['profiles']
  for x in rows:assert sha(pathlib.Path(x['input_file']))==x['input_SHA256']
  report['regular']=regular.run_checks(p0,a.base_url,a.model,o/'regular',identity,rows,a.timeout);save()
  report['long_output']=output.run_checks(p0,counter.count_tokens,a.base_url,a.model,o/'long-output',identity,a.timeout);save()
  report['passed']=report['regular']['passed']and report['long_output']['passed']
 except BaseException as e:report['error']=repr(e)
 finally:report['completed']=True;save()
 print(json.dumps({k:report.get(k)for k in ['completed','passed','error']}));return 0 if report['passed']else 1
if __name__=='__main__':raise SystemExit(main())
