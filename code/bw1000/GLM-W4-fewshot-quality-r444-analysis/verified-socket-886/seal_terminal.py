"""One readonly terminal collection. No wait, startup gate, HTTP or new job."""
from pathlib import Path
import base64,datetime,gzip,hashlib,json,subprocess
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
PREFIX='GLM-r444-verified-ready-socket-v1-verifiedready-20261104-socket-v1-s0'
FOLDER=R/'results/GLM-W4-fewshot-quality-r444/verifiedready-20261104-socket-v1-s0'
def main():
 watch=R/'results'/(PREFIX+'-watch.json');w=json.loads(watch.read_bytes())
 assert (w['actual_GPU'],w['actual_client'],w['CPU_keeper'])==('910376.886','910376.887','910407.950')
 if not w['completed']:print(json.dumps(dict(collected=False,reason='Original watch not terminal; no wait or newpoller')));return
 argv=['sacct','-S','2026-11-04','-j','910376,910407','-n','-P','-o','JobIDRaw,JobName%180,State,ExitCode,ElapsedRaw,NodeList%100']
 p=subprocess.run(argv,capture_output=True,text=True,timeout=30);assert p.returncode==0
 rows=[s for s in p.stdout.splitlines()if PREFIX in s];terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'}
 for step in ['910376.886','910376.887','910407.950']:
  matches=[x.split('|')for x in rows if x.split('|')[0]==step]
  if len(matches)!=1 or matches[0][2].split()[0].rstrip('+')not in terminal:print(json.dumps(dict(collected=False,reason='Original accounting not yet terminal',rows=rows)));return
 client=R/'results'/(PREFIX+'-client');assert Path(w['client_result'])in[client/'result.json',Path('/refactor/results')/(PREFIX+'-client/result.json')]
 files={}
 def add(path):
  raw=path.read_bytes();files[str(path)]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),gzip_base64=base64.b64encode(gzip.compress(raw,mtime=0)).decode())
 add(watch)
 for root in [FOLDER,client]:
  for path in sorted(root.rglob('*')):
   if path.is_file()and path.suffix in ['.json','.jsonl','.wire']:add(path)
 accounting='\n'.join(rows)+'\n';b=accounting.encode();accounting_path=str(R/'results/GLM886-terminal-sacct-original.txt');files[accounting_path]=dict(bytes=len(b),SHA256=hashlib.sha256(b).hexdigest(),base64=base64.b64encode(b).decode())
 archive=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),read_only=True,new_jobs=0,original_sacct_command=argv,accounting_path=accounting_path,files=files)
 output=R/'results/GLM886-terminal-original-sealed-v1.json.gz';raw=gzip.compress((json.dumps(archive,ensure_ascii=False)+'\n').encode(),mtime=0)
 with output.open('xb')as f:f.write(raw)
 print(json.dumps(dict(collected=True,path=str(output),bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),files=len(files),accounting_path=accounting_path,slot_index=dict(slot=0,watch_path=str(watch),client_path=str(client/'result.json'),run_directory=str(FOLDER),accounting_path=accounting_path))))
if __name__=='__main__':main()
