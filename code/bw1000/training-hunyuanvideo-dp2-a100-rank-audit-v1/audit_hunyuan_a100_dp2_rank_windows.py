"""Audit exact A100 DP2 rank windows, step records, and checkpoint commit markers."""
import hashlib
import json
from pathlib import Path
import subprocess

BASE = Path(__file__).resolve().parent
A100 = Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline')
manifest = (A100/'evidence/remote-node27-r2-manifest.json').read_bytes()
archive = json.loads(manifest)
pins = {row['path']: row for row in archive['files']}
runs = {f'hunyuan-cuda-dp2-baseline-r1-repeat{n}':0 for n in range(1,4)}
runs.update({f'hunyuan-cuda-dp2-fresh-resume{n}-r1':n for n in (10,20)})
selection={}
for run,start in runs.items():
    files = ['result.json', 'launch.json', 'checkpoint-25/COMMITTED.json']
    files += [f'rank-{rank}/{name}' for rank in (0,1) for name in ('summary.json','steps.ndjson')]
    if start==0:
        files += [f'checkpoint-{n}/COMMITTED.json' for n in (10,20)]
    selection[run] = {name:pins[f'node/results/{run}/{name}'] for name in files}
assert sum(p['bytes'] for files in selection.values() for p in files.values()) < 4*1024**2

REMOTE=r'''
from pathlib import Path
import hashlib,json,math,re,statistics,tarfile
home=Path('/home/whj/nvidia-baseline')
archive=Path(ARCHIVE)
assert archive.parent==home
digest=lambda raw:hashlib.sha256(raw).hexdigest()
is_sha=lambda value:isinstance(value,str) and bool(re.fullmatch('[a-f0-9]{64}',value))
tar=None
def read(pin):
    global tar
    rel=Path(pin['path'])
    assert rel.parts[:2]==('node','results') and 0<pin['bytes']<=256*1024
    for p in (home/rel,home.joinpath(*rel.parts[1:])):
        if p.is_file() and not p.is_symlink() and p.stat().st_size==pin['bytes']:
            raw=p.read_bytes()
            if digest(raw)==pin['sha256']:return raw
    if tar is None:tar=tarfile.open(archive,'r:gz')
    member=tar.getmember(rel.as_posix())
    assert member.isfile() and member.size==pin['bytes']
    raw=tar.extractfile(member).read(pin['bytes']+1)
    assert len(raw)==pin['bytes'] and digest(raw)==pin['sha256']
    return raw

def require(condition, label):
    if not condition:raise ValueError(label)

def audit(run, start, files):
    raw={n:read(pin) for n,pin in files.items()}
    doc={n:json.loads(b) for n,b in raw.items() if n.endswith('.json')}
    results=doc['result.json']
    metrics=results['metrics']
    cfg=doc['launch.json']['config']
    require(cfg['world_size']==2 and cfg['sp']==1 and cfg['warmup_steps']==5 and cfg['max_steps']==20,'launch geometry')
    require(metrics['world_size']==2 and metrics['performance_accepted'] is False,'aggregate acceptance')
    count=25-max(5,start)
    rows=[]; windows=[]
    for rank in (0,1):
        summary=doc[f'rank-{rank}/summary.json']
        require(summary['rank']==rank and summary['completed_global_step_at_start']==start,'rank resume cursor')
        require(summary['measurement_steps']==count,'rank window denominator')
        sec=summary['measurement_wall_seconds']
        require(math.isfinite(sec) and sec>0,'rank duration')
        windows.append(dict(rank=rank,seconds=sec,steps=count,boundary=summary['timing_boundary'],
                            adapter_before=summary['adapter_before'],adapter_after=summary['adapter_after']))
        steps=[json.loads(line) for line in raw[f'rank-{rank}/steps.ndjson'].splitlines()]
        require([r['global_step'] for r in steps]==list(range(start+1,26)),'rank step coverage')
        require(all(r['rank']==rank and math.isfinite(r['loss']) and math.isfinite(r['grad_norm_before_clip'])
                    and r['grad_norm_before_clip']>0 and all(is_sha(r[k]) for k in
                    ('replica_sha256','rank_state_sha256','rng_after_sha256','flow_input_sha256')) for r in steps),'rank step record')
        rows.append(steps)
    require(windows[0]['boundary']==windows[1]['boundary'],'unequal rank boundaries')
    require(all(a['replica_sha256']==b['replica_sha256'] for a,b in zip(*rows)),'replica divergence')
    require(metrics['measurement_steps']==count and math.isclose(metrics['measurement_wall_seconds'],
        max(w['seconds'] for w in windows),rel_tol=1e-10),'aggregate differs from slowest rank')
    markers={}
    for n in doc:
        if not n.endswith('/COMMITTED.json'):continue
        marker=doc[n]
        step=int(n.split('/')[0].split('-')[1])
        require(marker['global_step']==step and [p['rank'] for p in marker['ranks']]==[0,1],'checkpoint rank set')
        require(all(is_sha(p['sha256']) and is_sha(p['replica_sha256']) for p in marker['ranks']),'checkpoint digest')
        replicas={p['replica_sha256'] for p in marker['ranks']}
        require(replicas=={rows[0][step-start-1]['replica_sha256']},'checkpoint state differs from step')
        markers[str(step)]=dict(marker_SHA256=files[n]['sha256'],ranks=marker['ranks'],schema=marker['schema'])
    return dict(run=run,start=start,windows=windows,measurement_updates_per_second=count/max(w['seconds'] for w in windows),
        performance_accepted=False,checkpoints=markers,rank_steps=rows,
        configured_resume=cfg.get('resume_from_checkpoint'),original_artifacts=files)

audited={}
try:
    for run,files in SELECTION.items():
        audited[run]=audit(run,RUNS[run],files)
finally:
    if tar:tar.close()
baseline=audited['hunyuan-cuda-dp2-baseline-r1-repeat1']
recoveries={}
for run,result in audited.items():
    start=result['start']
    if start==0:continue
    for a,b in zip(baseline['rank_steps'],result['rank_steps']):
        require(a[start:]==b,'resumed rank trajectory differs from continuous run')
    require(result['checkpoints']['25']==baseline['checkpoints']['25'],'resumed final commit differs')
    recoveries[str(start)]=dict(replayed_steps_per_rank=25-start,
        resume_config_path=result['configured_resume'],matched_repeat1_step_records=True,
        matching_final_checkpoint_commit=True)
for record in audited.values():del record['rank_steps']
print(json.dumps(dict(passed=True,runs=audited,recoveries=recoveries,
    baseline_median_updates_per_second=statistics.median(v['measurement_updates_per_second']
        for v in audited.values() if v['start']==0),checkpoint_payloads_read=False,GPUs_launched=0)))
'''
wire = (f'ARCHIVE={archive["archive"]!r}\nSELECTION={selection!r}\nRUNS={runs!r}\n'+REMOTE).encode()
args=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A100/'ssh_config'),
      'a100-27','timeout 75 python3 -B -u -']
r=subprocess.run(args,input=wire,capture_output=True,timeout=88)
receipt=dict(rc=r.returncode,manifest_SHA256=hashlib.sha256(manifest).hexdigest(),
             stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())
if r.returncode==0:receipt['value']=json.loads(r.stdout)
else:
    tail=r.stderr.decode(errors='replace').splitlines()
    receipt['diagnostic']=[s[:160] for s in tail if s.startswith(('ValueError:', 'AssertionError', 'KeyError:', 'FileNotFoundError:'))][-2:]
    receipt['failure_types']=[name for name in ('Connection closed','Permission denied') if name.encode() in r.stderr]
with (BASE/'hunyuan-dp2-A100-rank-and-recovery-audit-v1.json').open('x',encoding='utf8') as f:
    json.dump(receipt,f,indent=2)
v=receipt.get('value',{})
print(json.dumps(dict(rc=r.returncode,diagnostic=receipt.get('diagnostic'),runs=len(v.get('runs',{})),
    recoveries=v.get('recoveries'),baseline_median=v.get('baseline_median_updates_per_second'))))
