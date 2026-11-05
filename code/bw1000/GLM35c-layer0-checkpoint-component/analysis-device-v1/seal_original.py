"""One terminal-only private seal. Never submit/stop a GPU or loop waiting."""
from pathlib import Path
import argparse,hashlib,io,json,os,socket,subprocess,tarfile,time
SHA=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def seal(root):
 R=Path(root);label='GLM35c-layer0-checkpoint-device-v1';D=R/'incoming/GLM35c-layer0-checkpoint-component/device-v1';watch=R/'results/GLM35c-layer0-device-watch-v1.json'
 w=json.loads(watch.read_bytes())
 if not w.get('completed'):return dict(sealed=False,live=True,actual_step=w.get('actual_step'),reason='Original watch is not terminal; no waiting/collection attempt')
 text=subprocess.check_output(['sacct','-S','2026-11-01','-j','910371,910407','-n','-P','-o','JobIDRaw,JobName%160,State,ExitCode,Elapsed,NodeList%100'],text=True,timeout=25)
 rows=[x.split('|')for x in text.splitlines()if len(x.split('|'))>1 and x.split('|')[1].startswith(label)]
 assert rows and all(x[2].split()[0].rstrip('+')in['COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE']for x in rows),'Exact original accounting remains live/unknown: no seal'
 arc=R/'results/GLM35c-layer0-device-v1-terminal-sealed-v1.tar.gz';intent=arc.with_suffix('.seal-intent.json')
 assert not arc.exists()and not intent.exists(),'Existing collection identity/archive must be reconciled, never resealed'
 identity=dict(HOST=socket.gethostname(),PID=os.getpid(),startticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19],epoch=time.time(),archive=str(arc),original_step=w.get('actual_step'))
 with intent.open('x')as f:json.dump(identity,f,indent=2)
 paths={'watch.json':watch,'GPU_SOURCE_MANIFEST.json':D/'GPU_SOURCE_MANIFEST.json','GPU_RESOURCE_CLAIM.json':D/'GPU_RESOURCE_CLAIM.json','freeze_continuation_inputs.py':D/'freeze_continuation_inputs.py','original-private-plan.json':R/'incoming/GLM952-prefix-continuation/private-original-plan.json','CPU-result.json':R/'candidates/GLM35c-layer0-checkpoint-component-CPU-v3/result.json'}
 step=w.get('actual_step')
 if step:
  assert step.startswith('910371.')and step.split('.')[1].isdigit()
  O=R/'results'/label/step
  if(O/'entry.json').is_file():paths['entry.json']=O/'entry.json'
  for rank in range(8):
   p=O/('rank%d.json'%rank)
   if not p.is_file():continue
   paths[p.name]=p;a=json.loads(p.read_bytes())
   for snap in a.get('snapshots',[]):
    for r in snap['files']:
     p=Path(r['path']);assert p.is_relative_to('/refactor/results/'+label+'/'+step)and'..'not in p.parts
     host=R/p.relative_to('/refactor');assert host.parent==O and host.name.startswith('rank%d-width'%rank)
     paths['raw/'+host.name]=host
  p=R/'logs'/(label+'-'+step+'.log')
  if p.is_file():paths['GPU-log.txt']=p
 for r in rows:
  p=R/'logs'/(r[1]+'-'+r[0]+'.log')
  if p.is_file():paths['logs/'+p.name]=p
 paths.update({'HOST-preflight.json':R/'results/GLM35c-layer0-device-v1-HOST-preflight.json','first-invocation.json':R/'results/GLM35c-layer0-device-v1-first-invocation.json'})
 # Model/checkpoint data are never included, only original token plan and diagnostic raw.
 meta={'GPU_label':label,'sealed_terminal':True,'identity':identity,'files':{}};raw_total=0
 acct=json.dumps(rows,indent=2).encode();extra={'accounting.json':acct}
 with arc.open('xb')as output:
  with tarfile.open(fileobj=output,mode='w:gz',compresslevel=1)as t:
   for name,p in paths.items():
    b=p.read_bytes();h=hashlib.sha256(b).hexdigest();meta['files'][name]={'source_path':str(p),'bytes':len(b),'SHA256':h}
    if name.startswith('raw/'):raw_total+=len(b)
    assert raw_total<=512*1024**2
    m=tarfile.TarInfo(name);m.size=len(b);m.mtime=0;t.addfile(m,io.BytesIO(b))
   for name,b in extra.items():
    meta['files'][name]={'bytes':len(b),'SHA256':hashlib.sha256(b).hexdigest()};m=tarfile.TarInfo(name);m.size=len(b);m.mtime=0;t.addfile(m,io.BytesIO(b))
   meta.update(raw_bytes=raw_total,performance=False,Model_quality=False);b=(json.dumps(meta,indent=2)+'\n').encode();m=tarfile.TarInfo('SEAL.json');m.size=len(b);t.addfile(m,io.BytesIO(b))
 result=dict(sealed=True,archive=str(arc),bytes=arc.stat().st_size,SHA256=SHA(arc),raw_bytes=raw_total,files=len(meta['files']),identity=identity,watch_passed=w.get('passed'),Model_quality=False,performance=False)
 with arc.with_suffix('.receipt.json').open('x')as f:json.dump(result,f,indent=2)
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',default='/public/home/aclsylqidf/hygon-inference-refactor-20261014');a=p.parse_args();print(json.dumps(seal(a.root)),flush=True)
