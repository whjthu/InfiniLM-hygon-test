"""Freeze exactly the original COCO20 rows/files;never export image values."""
from pathlib import Path
import hashlib,json,subprocess,time
D=Path(__file__).resolve().parent;A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline')
code=b'''from pathlib import Path
import csv,datetime,hashlib,io,json,os,signal,time
signal.alarm(60);start=time.monotonic();os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
root=Path('/mnt/data/nvidia-baseline-20261010/inputs/datasets/coco-val2017')
metadata=root/'metadata-zimage-20.csv';manifest=root/'manifest-zimage-20.json'
def stats(p):
 s=p.stat();return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev)
def small(p):
 assert p.is_file() and not p.is_symlink() and p.stat().st_size<1024**2
 raw=p.read_bytes();return dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),utf8=raw.decode())
meta=small(metadata);mf=small(manifest)
rows=list(csv.DictReader(io.StringIO(meta['utf8'])));assert len(rows)==20 and all('image' in r and 'prompt' in r for r in rows)
images={};total=0
for row in rows:
 rawname=row['image'];part=Path(rawname);p=part if part.is_absolute() else root/part
 relative=p.relative_to(root);assert '..' not in relative.parts
 q=root
 for name in relative.parts:q=q/name;assert not q.is_symlink()
 assert p.is_file();before=stats(p);assert 0<before['bytes']<=20*1024**2
 name=relative.as_posix();assert name not in images
 h=hashlib.sha256();read=0
 with p.open('rb') as stream:
  while True:
   assert time.monotonic()-start<55
   block=stream.read(1024**2)
   if not block:break
   h.update(block);read+=len(block)
 assert stats(p)==before and read==before['bytes'];total+=read;assert total<=256*1024**2
 images[name]=dict(**before,SHA256=h.hexdigest())
print(json.dumps(dict(observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),root=str(root),metadata=meta,manifest=mf,
 rows=rows,images=images,image_files=len(images),image_bytes_read=total,image_values_exported=False,model_weights_read=False,GPU_calls=0,inside_seconds=time.monotonic()-start,
 declared_original_dataset_identity='4ddebfd6c9aa4fc37caef3add589d470c1c54e3a20f0ed26092d37b4352b4e4c',
 declaration_matches_manifest_full_SHA256=mf['SHA256']=='4ddebfd6c9aa4fc37caef3add589d470c1c54e3a20f0ed26092d37b4352b4e4c')))
'''
start=time.monotonic();r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-27','python3 -B -u -'],input=code,capture_output=True,timeout=80)
(D/'original-inputs.stdout.private.json').write_bytes(r.stdout);(D/'original-inputs.stderr.private.txt').write_bytes(r.stderr)
v=json.loads(r.stdout) if r.stdout else None
result=dict(transport_rc=r.returncode,RTT_seconds=time.monotonic()-start,stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest(),
 original_input_files=v['image_files'] if v else None,original_image_bytes=v['image_bytes_read'] if v else None,
 declaration_matches_manifest_full_SHA256=v['declaration_matches_manifest_full_SHA256'] if v else None,
 metadata={k:v['metadata'][k] for k in ('bytes','SHA256')} if v else None,manifest={k:v['manifest'][k] for k in ('bytes','SHA256')} if v else None,
 image_values_exported=False,new_GPU_launches=0)
with (D/'original-inputs-result.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
