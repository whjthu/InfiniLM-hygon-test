"""CPU-only P0 supervisor launched by actual emitter after ready.

Never imports Native, starts Slurm, retries a model, or fabricates model output.
The child client owns real HTTP requests; finally requests graceful Native stop.
"""
import argparse,datetime,hashlib,json,os,pathlib,subprocess,sys
CLIENT_SHA256='41dbe048d14ec8f023bd3448426d6e87931f094e5919db6ee72957e0cdf1d77e'
def run_client(client,ready,output,stop,record,expected_handle,runner=subprocess.run):
 sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
 q={'completed':False,'passed':False,'CPU_only':True,'Native_calls':False,'GPU_handle':expected_handle,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'client_SHA256':sha(client),'ready_SHA256':sha(ready)}
 # Validate before taking authority over this service's graceful stop marker.
 assert q['client_SHA256']==CLIENT_SHA256
 identity=json.loads(ready.read_bytes());assert identity['ready']and identity['actual_step']==expected_handle and identity['context']==640 and identity['cards']==16 and identity['model_path']=='/old/models/ZhipuAI-GLM-5.3'and identity['Native_DSO_SHA256']=='ce830c4d77fe193e4a6e82597aa4fb5b50019c82edce542eb6d61217a238bf68'
 assert not output.exists()and not record.exists()and not stop.exists()
 def save():
  temp=record.with_suffix('.tmp');temp.write_text(json.dumps(q,indent=2));temp.replace(record)
 args=[sys.executable,'-B','-u',str(client),'--base-url','http://127.0.0.1:43843/v1','--model','GLM-5.3','--max-tokens','96','--timeout','180','--identity',str(ready),'--output-dir',str(output)]
 q['args']=args;save()
 try:
  process=runner(args,timeout=1800,check=False);q.update(client_exit_code=process.returncode,completed=True)
  result=output/'result.json'
  if result.exists():
   actual=json.loads(result.read_bytes());q.update(result_path=str(result),result_SHA256=sha(result),passed=bool(process.returncode==0 and actual.get('completed')is True and actual.get('passed')is True),actual_HTTP_requests=actual.get('actual_HTTP_requests'))
 except BaseException as error:q.update(error=repr(error),completed=False)
 finally:
  save()
  marker={'normal_stop':True,'reason':'automatic_P0_client_finished_or_failed','GPU_handle':expected_handle,'P0_record':str(record),'P0_record_SHA256':sha(record),'P0_passed':q['passed']}
  with stop.open('x')as f:json.dump(marker,f,indent=2)
 return q
def main():
 a=argparse.ArgumentParser();a.add_argument('--run',required=True);a.add_argument('--handle',required=True);o=a.parse_args()
 assert o.run.startswith('GLM-agent-service-r143c-')and o.run.replace('-','').isalnum()
 assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
 r=pathlib.Path('/refactor');d=pathlib.Path(__file__).resolve().parent
 q=run_client(d/'client_p0_r143.py',r/'results'/('GLM-agent-service-r143c-ready-'+o.run+'.json'),r/'results'/('GLM-agent-P0-r143c-'+o.run),r/'control'/('GLM-agent-service-r143c-stop-'+o.run+'.json'),r/'results'/('GLM-agent-P0-supervisor-r143c-'+o.run+'.json'),o.handle)
 print(json.dumps(q));return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
