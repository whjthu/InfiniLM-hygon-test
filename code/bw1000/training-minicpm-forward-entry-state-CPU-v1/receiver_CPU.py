from pathlib import Path
import base64,datetime,hashlib,json,os,subprocess,sys,time
p=PAYLOAD;assert set(p)=={'files','manifest','claim_commit'} and len(p['claim_commit'])==40
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-minicpm-forward-entry-state-CPU-v1');W.mkdir(mode=0o700,exist_ok=False)
start=time.monotonic();epoch=time.time();deadline=epoch+120
(W/'intent.json').write_text(json.dumps(dict(epoch=epoch,deadline=deadline,claim_commit=p['claim_commit'],GPU_cards=0,CPU_test_timeout=90)))
manifest=p['manifest'];assert len(manifest)==13 and set(p['files'])=={x['path'] for x in manifest}
for row in manifest:
 name=row['path'];assert name.startswith('framework/') and '..' not in Path(name).parts
 raw=base64.b64decode(p['files'][name]);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['SHA256'];target=W/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
timeout=min(90,deadline-time.time()-10);assert timeout>0
r=subprocess.run(['/home/whj/nvidia-baseline/env-vlm/bin/python','-B','-u','-X','utf8','run_tests_CPU.py'],cwd=W/'framework',env=env,capture_output=True,timeout=timeout)
(W/'stdout.private.json').write_bytes(r.stdout);(W/'stderr.private.txt').write_bytes(r.stderr)
result=json.loads(r.stdout) if r.stdout else None
out=dict(rc=r.returncode,epoch=epoch,deadline=deadline,wall_seconds=time.monotonic()-start,result=result,stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest(),stderr_bytes=len(r.stderr),source_files=len(manifest),GPU_cards=0,weights_read=False,production_training=False)
(W/'result.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
