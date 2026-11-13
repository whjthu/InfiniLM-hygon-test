"""Read the manifest-pinned HunyuanVideo A100 launch and result JSONs in place."""
from pathlib import Path
import hashlib,json,subprocess

B=Path(__file__).resolve().parent
A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline')
groups={'26':{'T09.sft':'hunyuan-cuda-sft-baseline-r1',
              'T09.sage-grpo':'hunyuan-cuda-rl-baseline-r1'},
        '27':{'T09.dp2':'hunyuan-cuda-dp2-baseline-r1'}}
launch_pins={'T09.sft':'3fdd29e7ea1a09fc54872c044699199cd5ef16bd73af6e7d522fe2ff1b32662f',
             'T09.dp2':'181b1ca706a7d62741298192c3391a0336acf0c60624aa9f2645e90a55934a6f',
             'T09.sage-grpo':'6b230bb6ac909a4599faa37fc9a20e85d425c97e8c643ca111b3f124ef16c5a3'}
remote=r'''
from pathlib import Path
import hashlib,json,tarfile
home=Path('/home/whj/nvidia-baseline')
archive=Path(ARCHIVE)
assert archive.parent==home and archive.name.startswith('evidence-export-app-')
result={}
archive_reader=None
try:
    for case,repeats in CASES.items():
        records=[]
        for pins in repeats:
            item={}
            for role,pin in pins.items():
                rel=Path(pin['path'])
                assert rel.parts[:2]==('node','results') and len(rel.parts)==4
                assert 0<pin['bytes']<=262144
                candidates=[home/rel,home.joinpath(*rel.parts[1:])]
                found=[p for p in candidates if p.is_file() and not p.is_symlink()]
                raw=None
                for p in found:
                    if p.stat().st_size!=pin['bytes']:continue
                    data=p.read_bytes()
                    if hashlib.sha256(data).hexdigest()==pin['sha256']:
                        raw=data
                        break
                if raw is None and archive.is_file() and not archive.is_symlink():
                    if archive_reader is None:archive_reader=tarfile.open(archive,mode='r:gz')
                    try:member=archive_reader.getmember(rel.as_posix())
                    except KeyError:member=None
                    if member and member.isfile() and member.size==pin['bytes']:
                        raw=archive_reader.extractfile(member).read(pin['bytes']+1)
                if raw is None:
                    item[role]=dict(present=False,pin=pin)
                    continue
                assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['sha256']
                doc=json.loads(raw)
                if role=='launch':
                    config=doc['config']
                    keys=('seed','world_size','sp','max_steps','checkpoint_steps','warmup_steps',
                          'training_cache','training_cache_sha','transformer_version','lora_rank','lora_alpha',
                          'lora_target_modules','actor_devices','vae_device','num_rollouts','num_inference_steps')
                    values=dict(config={k:config[k] for k in keys if k in config},config_keys=sorted(config),
                                weights=doc.get('weights'),source=doc.get('source'))
                else:
                    names=('status','passed','ok','world_size','sp','timed_steps','warmup_steps','max_steps',
                           'completed_steps','train_wall_s','walltime_seconds','elapsed_seconds','metrics','timing',
                           'timings','throughput','steps_per_second','checkpoint_steps','checkpoint_step_basis',
                           'checkpoint_io_policy','total_rollouts','stage_timings','checks','invocation_seconds')
                    def bounded(v):
                        if isinstance(v,(int,float,bool,str,type(None))):return v
                        if isinstance(v,dict):return {k:bounded(x) for k,x in v.items() if isinstance(x,(int,float,bool,type(None)))}
                        return None
                    values=dict(fields={k:bounded(doc[k]) for k in names if k in doc},keys=sorted(doc))
                item[role]=dict(present=True,pin=pin,summary=values)
            records.append(item)
        result[case]=records
finally:
    if archive_reader:archive_reader.close()
print(json.dumps(result))
'''
full={}
for host,cases in groups.items():
    if host != "27":continue
    manifest=A/f'evidence/remote-node{host}-r2-manifest.json'
    raw=manifest.read_bytes()
    doc=json.loads(raw)
    members={row['path']:row for row in doc['files']}
    selected={}
    for case,prefix in cases.items():
        selected[case]=[]
        for repeat in range(1,4):
            stem=f'node/results/{prefix}-repeat{repeat}/'
            pins={n:members[stem+n+'.json'] for n in ('launch','result')}
            assert pins['launch']['sha256']==launch_pins[case]
            selected[case].append(pins)
    program=f'CASES={selected!r}\nARCHIVE={doc["archive"]!r}\n'+remote
    argv=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),
          'a100-'+host,'timeout 40 python3 -B -u -']
    try:
        r=subprocess.run(argv,input=program.encode(),capture_output=True,timeout=55)
        val=dict(rc=r.returncode,manifest_SHA256=hashlib.sha256(raw).hexdigest(),
                 stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())
        if r.returncode==0:val['cases']=json.loads(r.stdout)
        else:val['failure_types']=[x for x in ('Connection closed','AssertionError','Permission denied','KeyError') if x.encode() in r.stderr]
    except subprocess.TimeoutExpired:
        val=dict(rc=None,error='observation timed out')
    full[host]=val
with (B/'hunyuan-original-a100-dp2-reference-check-v2.json').open('x',encoding='utf8') as f:json.dump(full,f,indent=2)
print(json.dumps({h:dict(rc=v['rc'],counts={c:sum(a['present'] for row in rows for a in row.values())
    for c,rows in v.get('cases',{}).items()},failure_types=v.get('failure_types')) for h,v in full.items()}))
