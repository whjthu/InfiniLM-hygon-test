"""Prepare and prove the 17 read-only InternVL model binds on the verified host."""

import json
import os
from pathlib import Path
import socket
import subprocess
import sys


ROOT = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
INPUT = ROOT / 'incoming/training-internvl-modelroot17-preflight-635df739726b41c29a71aceff6569076'
OUTPUT = ROOT / 'results/training-internvl-modelroot17-current-mount-CPU-v1'
REHASH = ROOT / 'results/training-internvl-original14-current-mount-rehash-CPU-v1/outcome.json'
REQUEST = '94a3020790c84091836a492a596a70f9'
ASSEMBLER_SHA = 'ebff2c2df8665cde980758aa0a8bfb06e5edf65cd71d1346bde2fc432315d3d6'
AUTH_SHA = '8aea495d35a53a311b40a5d2b09379b8e76ee7d5cca68557496a6eb18dca5a49'
REHASH_SHA = '668a0262b516d73cb00cd73ad09e36b3396d28983059e7594061af0b2bdaab62'


def load_assembler():
    import hashlib
    import importlib.util

    path = INPUT / 'assemble.py'
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != ASSEMBLER_SHA:
        raise ValueError('existing assembler differs from the published source')
    spec = importlib.util.spec_from_file_location('internvl_assemble', path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(data, str(path), 'exec'), module.__dict__)
    return module


def mount_plan(a):
    assets, _ = a.document(INPUT/'ASSETS.json', a.ASSETS_SHA, 65536)
    authority, _ = a.document(INPUT/'three-original-seals.json', AUTH_SHA, 65536)
    current, _ = a.document(REHASH, REHASH_SHA, 65536)
    old, _ = a.document(a.ORIGINAL_361, a.OUTCOME_SHA, a.OUTCOME_BYTES, extent=a.OUTCOME_BYTES)
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    a.require(current['state']=='CURRENT_MOUNT_ALL14_WHOLE_SHA_MATCHED' and current['failure_type'] is None
              and current['actual_verified_members']==14 and current['actual_verified_body_bytes']==4922323586
              and current['authority_SHA256']==AUTH_SHA and current['old361_outcome_SHA256']==a.OUTCOME_SHA
              and current['host']==socket.gethostname() and current['boot_id']==boot,
              'old14 qualification only applies on its observed host and boot')
    baseline = a.old_rows(old, assets['old14'])
    verified = current['verified_members']
    a.require(set(verified)==set(baseline), 'original old14 coverage differs')
    rows = a.source_receipts(assets, authority)
    for name, entry in assets['old14'].items():
        stamp, _ = a.fd_info(entry['source'])
        witness = verified[name]
        a.require(name not in rows and set(stamp)==set(baseline[name])
                  and {k for k in stamp if stamp[k]!=baseline[name][k]}=={'st_dev'}
                  and witness==dict(bytes=entry['bytes'],SHA256=entry['SHA256'],current_target_stat=stamp),
                  'old14 member differs from its current mount hash receipt')
        rows[name] = dict(source=entry['source'],bytes=entry['bytes'],SHA256=entry['SHA256'],stamp=stamp)
    a.require(len(rows)==17 and all(a.TOKEN.fullmatch(leaf) for leaf in rows)
              and sum(row['bytes'] for row in rows.values())==17072712250, 'expected all 17 original members')
    index = rows['model.safetensors.index.json']
    value, stamp = a.document(index['source'],index['SHA256'],131072,extent=index['bytes'])
    a.require(stamp==index['stamp'] and set(value['weight_map'].values())==
              {f'model-0000{i}-of-00004.safetensors' for i in range(1,5)}, 'original index shard coverage differs')
    return dict(request=a.REQUEST, qualification_request=REQUEST, prepare_allowed=True, members=rows,
                modelroot=str(OUTPUT/'modelroot'), source_hash_receipt=REHASH_SHA, three_seals_receipt=AUTH_SHA,
                verified_host_boot=dict(host=socket.gethostname(),boot=boot), training_READY=False)


def main():
    a = load_assembler()
    a.require(sys.argv[1:]==[] and os.getuid()==a.OWNER, 'fixed project-owned inspection required')
    prepared = a.prepare(mount_plan(a), OUTPUT)
    plan, _ = a.document(prepared['plan'],prepared['plan_SHA256'],65536)
    args = ['/public/software/apps/apptainer/1.3.4/bin/apptainer','exec','--contain','--no-home','--pwd','/']
    for bind in [*prepared['binds'],f'{INPUT}:/audit:ro',f'{prepared["plan"]}:/audit-plan.json:ro']:
        args += ['--bind',bind]
    argv = [*args,str(ROOT/'containers/dtk26-sandbox-r1'),'/usr/bin/python3','-I','-S','-B',
            '/audit/assemble.py','verify-mounted','--plan','/audit-plan.json','--plan-sha256',prepared['plan_SHA256']]
    try:
        child = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True, timeout=35)
        limited = child.stdout[-65536:]
        a.write_once(OUTPUT/'mount-verifier.stdout',limited)
        a.write_once(OUTPUT/'mount-verifier.stderr',child.stderr[-65536:])
        result = dict(request=REQUEST,plan_SHA256=prepared['plan_SHA256'],container_rc=child.returncode,
                      source_hash_receipt=REHASH_SHA,three_seals_receipt=AUTH_SHA,
                      source_weight_body_bytes=0,actual_GPU_steps=0,training_READY=False,
                      container_stdout_SHA256=a.digest(limited),container_stderr_SHA256=a.digest(child.stderr))
        if child.returncode==0:
            proof = json.loads(limited)
            a.require(proof['request']==a.REQUEST and proof['members']==17 and
                      proof['state']=='VERIFIED_ALL17_READONLY_REGULAR_MOUNTS', 'unexpected mounted verifier result')
            a.verify_originals(plan)
            result.update(state='VERIFIED_ALL17_READONLY_ON_REHASH_HOST',same_host_only=True,
                          readonly_members=17,logical_model_bytes=proof['logical_model_bytes'],
                          original_members_unchanged_after_container=True)
        else:
            result.update(state='HOST_PLAN_PREPARED_CONTAINER_FAILED',same_host_only=True)
        body = a.serialized(result)
        a.write_once(OUTPUT/'outcome.json',body)
        print(json.dumps(result|{'outcome_SHA256':a.digest(body)}))
        return int(child.returncode!=0)
    except subprocess.TimeoutExpired:
        a.write_once(OUTPUT/'timeout.json',a.serialized(dict(request=REQUEST,plan_SHA256=prepared['plan_SHA256'],
                     state='CONTAINER_INSPECTION_CLIENT_TIMEOUT',training_READY=False)))
        raise


if __name__=='__main__':
    raise SystemExit(main())
