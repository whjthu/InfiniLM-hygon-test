"""Owned finite CPU child for one exact archive, with no GPU admission or execution."""
import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
def save(name,v):
    with (ROOT/name).open('x') as f:json.dump(v,f,indent=2)
installed=json.loads((ROOT/'installed.json').read_bytes())
assert not (ROOT/'build-intent.json').exists() and not (ROOT/'archive-receipt.json').exists()
for name,h in installed['files'].items():assert hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest()==h
assert __import__('shutil').disk_usage(ROOT).free>=4*1024**3
start=time.monotonic();args=[sys.executable,'-B',str(ROOT/'src/build_archive.py'),'--plan',str(ROOT/'src/ASSET_PLAN.json'),'--output',str(ROOT),'--claim',installed['claim_commit']]
forced=False
with (ROOT/'build.stdout').open('xb') as out,(ROOT/'build.stderr').open('xb') as err:
    p=subprocess.Popen(args,stdout=out,stderr=err,start_new_session=True)
    ticks=int(Path('/proc',str(p.pid),'stat').read_text().split(') ',1)[1].split()[19])
    save('cpu-owner.json',dict(PID=p.pid,PGID=p.pid,startticks=ticks,claim=installed['claim_commit']))
    try:p.wait(timeout=270)
    except subprocess.TimeoutExpired:
        forced=True
        assert int(Path('/proc',str(p.pid),'stat').read_text().split(') ',1)[1].split()[19])==ticks
        os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=5)
result=dict(returncode=p.returncode,forced_cleanup=forced,wall_seconds=time.monotonic()-start,
            stderr_SHA256=hashlib.sha256((ROOT/'build.stderr').read_bytes()).hexdigest(),
            stdout_SHA256=hashlib.sha256((ROOT/'build.stdout').read_bytes()).hexdigest(),
            GPU_initialized=False,claim_commit=installed['claim_commit'])
if (ROOT/'archive-receipt.json').exists():
    result['archive_receipt']=json.loads((ROOT/'archive-receipt.json').read_bytes())
    result['archive_receipt_SHA256']=hashlib.sha256((ROOT/'archive-receipt.json').read_bytes()).hexdigest()
result['passed']=bool(p.returncode==0 and not forced and result.get('archive_receipt',{}).get('completed'))
save('cpu-outcome.json',result);print(json.dumps(result))
