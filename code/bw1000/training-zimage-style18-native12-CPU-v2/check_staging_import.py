"""Clean -I -S target-shaped scheduler import with delivered members only."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def pin(raw):return {'bytes':len(raw),'SHA256':hashlib.sha256(raw).hexdigest()}


CHILD = r'''
import hashlib,json,pathlib,subprocess,sys
stage=pathlib.Path(sys.argv[1]).resolve(strict=True)
sys.path.insert(0,str(stage))
calls=[]
def no_process(*args,**kwargs):
 calls.append(True)
 raise RuntimeError('No process/scheduler execution permitted in import-only check')
subprocess.Popen=no_process
subprocess.run=no_process
subprocess.check_output=no_process
import pilot
scheduler=pilot.load_scheduler(stage)
assert scheduler.node_map(json.loads((stage/'ledger.json').read_bytes()))['f11r4n14']==('1087656','T07')
providers={}
for name in ('pilot','sync_dispatch','qualified_sampler','admission'):
 module=sys.modules[name]
 path=pathlib.Path(module.__file__).resolve(strict=True)
 assert path==stage/(name+'.py')
 raw=path.read_bytes()
 providers[name]={'bytes':len(raw),'SHA256':hashlib.sha256(raw).hexdigest()}
assert not calls and not any(n in sys.modules for n in ('torch','diffsynth','sitecustomize'))
assert all(not p or 'site-packages' not in p for p in sys.path)
print(json.dumps({'passed':True,'providers':providers,'scheduler_or_process_calls':len(calls),
 'actual_node_map_used':'1087656/f11r4n14/T07','I_flag':sys.flags.isolated,'S_flag':sys.flags.no_site,
 'only_staged_project_modules':True,'synthetic_root_adoption_for_import_only':True},sort_keys=True))
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--CSV',type=Path,required=True)
    args=p.parse_args()
    package=Path(__file__).resolve().parent
    args.output.mkdir(mode=0o700)
    stage=args.output/'target-incoming-fixture'
    stage.mkdir()
    manifest=json.loads((package/'SOURCE_FILES.json').read_bytes())
    for name,expected in manifest['files'].items():
        raw=(package/name).read_bytes()
        if pin(raw)!=expected:raise ValueError('Actual delivered source manifest differs')
        path=stage/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f:f.write(raw)
    # Mirrors only the future root source adoption. No production claim changes.
    claim=json.loads((stage/'CLAIM.json').read_bytes())
    assert claim['root_GO'] is claim['CPU_GO'] is False
    claim.update(root_GO=True,CPU_GO=True,state='SYNTHETIC_LOCAL_IMPORT_ONLY_ADOPTION')
    raw=(json.dumps(claim,indent=2,sort_keys=True)+'\n').encode()
    (stage/'CLAIM.json').write_bytes(raw)
    manifest['files']['CLAIM.json']=pin(raw)
    (stage/'SOURCE_FILES.json').write_text(json.dumps(manifest,sort_keys=True)+'\n',encoding='utf8')
    raw=args.CSV.read_bytes()
    assert len(raw)==3601 and hashlib.sha256(raw).hexdigest()=='f71b51feb39af32e1d79b7750c9c4c437d930a35f69b60e52f80389ba6084350'
    (stage/'metadata.csv').write_bytes(raw)
    body=CHILD.encode()
    (args.output/'child-stdlib-source.py').write_bytes(body)
    reply=subprocess.run([sys.executable,'-I','-S','-B','-u','-c',CHILD,str(stage.resolve())],
                         stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                         timeout=15)
    (args.output/'stdout.private').write_bytes(reply.stdout)
    (args.output/'stderr.private').write_bytes(reply.stderr)
    result={'state':'CLEAN_DELIVERED_ONLY_SCHEDULER_IMPORT_PASS' if reply.returncode==0 else 'CLEAN_IMPORT_FAILED',
            'passed':reply.returncode==0,'returncode':reply.returncode,
            'original_package_manifest':pin((package/'SOURCE_FILES.json').read_bytes()),
            'stdout':pin(reply.stdout),'stderr':pin(reply.stderr),'child_source':pin(body),
            'planned_native_rows':12,'actual_native_calls':0,'remote_or_Slurm_calls':0,
            'original_request_replayed':False,'production_claim_changed':False,'training_READY':False}
    if reply.returncode==0:result['observed']=json.loads(reply.stdout)
    raw=(json.dumps(result,indent=2,sort_keys=True)+'\n').encode()
    (args.output/'result.json').write_bytes(raw)
    print(json.dumps({'passed':result['passed'],'result':pin(raw)},sort_keys=True))
    raise SystemExit(0 if result['passed'] else 1)


if __name__=='__main__':main()
