"""Execute each frozen original CPU capture/processor staging scope at most once."""
from pathlib import Path
import argparse,base64,datetime,hashlib,json,shlex,subprocess,sys,time,uuid
B=Path(__file__).resolve().parents[2];D=B/'training-gemma-native-inputs-v1';P=D/'package'
def digest(raw):return hashlib.sha256(raw).hexdigest()
def dump(p,v):p.write_bytes((json.dumps(v,indent=2)+'\n').encode())
def main():
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['capture','archive','receive']);args=parser.parse_args();stage=args.stage
 claim_raw=(P/'CLAIM.json').read_bytes();claim=digest(claim_raw);source=json.loads((P/'SOURCE_FILES.json').read_bytes())
 for name,pin in source.items():
  raw=(P/name).read_bytes();assert len(raw)==pin['bytes'] and digest(raw)==pin['SHA256'],name
 request=uuid.uuid4().hex;intent=D/(stage+'-launch-intent.json');assert not intent.exists(),'original scope already attempted; reconcile, never replay'
 dump(intent,dict(request_id=request,claim=claim,stage=stage,local_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),replay_allowed=False))
 A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline')
 ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-26','python3 -B -u -']
 if stage=='capture':
  names=['FRAMEWORK_FILES.json','MODEL_PROCESSOR_FILES.json','ORIGINAL_INPUT_FILES.json','original-launch.json']+['framework/'+n for n in json.loads((P/'FRAMEWORK_FILES.json').read_bytes())['execution_closure']]
  files={n:dict(b64=base64.b64encode((P/n).read_bytes()).decode(),bytes=(P/n).stat().st_size,SHA256=digest((P/n).read_bytes())) for n in names}
  code='CLAIM='+repr(claim)+'\nREQUEST='+repr(request)+'\nFILES='+repr(files)+'\n'+(P/'capture_source_CPU.py').read_text()
  started=time.monotonic()
  try:
   r=subprocess.run(ssh,input=code.encode(),capture_output=True,timeout=320)
   (D/'capture-stdout.private.json').write_bytes(r.stdout);(D/'capture-stderr.private.txt').write_bytes(r.stderr)
   result=json.loads(r.stdout);dump(D/'capture-transport.json',dict(returncode=r.returncode,wall_seconds=time.monotonic()-started,stdout_SHA256=digest(r.stdout),request_id=request));dump(D/'original-CPU-capture.json',result)
   print(json.dumps(dict(stage=stage,returncode=r.returncode,passed=result.get('passed'),error_type=result.get('error_type'),capture_stage=result.get('stage'),rows=result.get('capture',{}).get('rows'),wall_seconds=result.get('wall_seconds'),request_id=request)))
   if r.returncode or not result['passed']:raise SystemExit(1)
  except subprocess.TimeoutExpired:
   dump(D/'capture-transport.json',dict(remote_outcome='UNKNOWN',request_id=request,replay_allowed=False));raise
 elif stage=='archive':
  code='CLAIM='+repr(claim)+'\nREQUEST='+repr(request)+'\nPLAN='+repr((P/'MODEL_PROCESSOR_FILES.json').read_bytes())+'\n'+(P/'produce_processor_archive.py').read_text()
  with (D/'processor-assets.tar').open('xb') as out,(D/'archive-stderr.private.txt').open('xb') as err:
   r=subprocess.run(ssh,input=code.encode(),stdout=out,stderr=err,timeout=200)
  lines=(D/'archive-stderr.private.txt').read_text().splitlines();result=json.loads(next(line for line in reversed(lines) if line.startswith('{')));archive=(D/'processor-assets.tar').read_bytes()
  dump(D/'archive-transport.json',dict(returncode=r.returncode,request_id=request,archive_bytes=len(archive),archive_SHA256=digest(archive)))
  assert r.returncode==0 and result['passed'] and len(archive)==result['archive_bytes'] and digest(archive)==result['archive_SHA256'] and len(archive)<=40500000
  dump(D/'processor-archive-result.json',result);print(json.dumps(dict(stage=stage,passed=True,files=len(result['files']),bytes=result['file_bytes'],archive_bytes=len(archive),request_id=request)))
 else:
  archive=(D/'processor-assets.tar');result=json.loads((D/'processor-archive-result.json').read_bytes());assert result['passed'] and archive.stat().st_size==result['archive_bytes'] and digest(archive.read_bytes())==result['archive_SHA256']
  original=(B/'i03-deep-c4-history-staging-v1/read_source_once.py').read_text(encoding='utf8');ns={'__file__':str(D/'reuse.py')};exec(compile(original.split('REMOTE=',1)[0],'existing_cluster_ssh','exec'),ns);ssh=ns['ssh']
  code=(P/'receive_processor_assets.py').read_bytes();code_sha=digest(code)
  bootstrap="import sys,json,base64,hashlib;h=json.loads(sys.stdin.buffer.readline(1048576));c=base64.b64decode(h['code']);assert hashlib.sha256(c).hexdigest()=="+repr(code_sha)+";exec(compile(c,'frozen_processor_receiver','exec'),{'PAYLOAD':h['payload']})"
  ssh[-1]=ssh[-1].removesuffix(' -')+' -c '+shlex.quote(bootstrap)
  payload=dict(claim=claim,request=request,manifest=json.loads((P/'MODEL_PROCESSOR_FILES.json').read_bytes()),archive_SHA256=result['archive_SHA256'],archive_bytes=result['archive_bytes']);header=json.dumps(dict(code=base64.b64encode(code).decode(),payload=payload)).encode()+b'\n'
  with (D/'receive-stdout.private.json').open('xb') as out,(D/'receive-stderr.private.txt').open('xb') as err:
   process=subprocess.Popen(ssh,stdin=subprocess.PIPE,stdout=out,stderr=err)
   try:
    process.stdin.write(header)
    with archive.open('rb') as stream:
     while True:
      block=stream.read(1024**2)
      if not block:break
      process.stdin.write(block)
    process.stdin.close();rc=process.wait(timeout=200)
   except BaseException:
    process.kill();process.wait();dump(D/'receive-transport.json',dict(remote_outcome='UNKNOWN',request_id=request,replay_allowed=False));raise
  raw=(D/'receive-stdout.private.json').read_bytes();actual=json.loads(raw);dump(D/'receive-transport.json',dict(returncode=rc,request_id=request,stdout_SHA256=digest(raw)));dump(D/'target-processor-stage.json',actual)
  print(json.dumps(dict(stage=stage,returncode=rc,passed=actual.get('passed'),files=len(actual.get('installed_files',[])),file_bytes=actual.get('file_bytes'),request_id=request)))
  if rc or not actual['passed']:raise SystemExit(1)
if __name__=='__main__':main()
