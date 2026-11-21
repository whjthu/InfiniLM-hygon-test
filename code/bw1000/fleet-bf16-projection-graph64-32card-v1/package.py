"""Prepare one sealed controller payload; callers own admission and transport."""

import base64
import json
from pathlib import Path
import sys
import zlib


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0,str(HERE))
from controller import read,sha,source_identity,source_verify
from scope import REQUEST,validate_plan


QUALIFIED_PILOT = Path('results/bw1000/2026-11-22/O01-BF16-graph64-original674-eight-card-paired-measurement.json')
PILOT_SHA = 'ad5cf2658a85fb25cd003ee01d78e5d094474aa6de7777892b282db763bba4ab'


def source(repo=REPO):
    index=HERE.relative_to(REPO)/'SOURCE_FILES.json'
    raw=read(repo/index,32768)
    references={name:Path(pin['path']) for name,pin in json.loads(raw)['files'].items()}
    need=all(not path.is_absolute() and '..' not in path.parts for path in references.values())
    if not need:
        raise ValueError('source paths leave published tree')
    files={name:read(repo/path,32768) for name,path in references.items()}
    files['SOURCE_FILES.json']=raw
    source_verify(files)
    references['SOURCE_FILES.json']=index
    return references,files


def prepare(publication, campaign_epoch, selected, before_snapshot_SHA256, *, repo=REPO):
    """Return the fixed plan, finite-code bytes and exact source paths. No execution."""
    raw=read(repo/QUALIFIED_PILOT,32768)
    record=json.loads(raw)
    if (sha(raw)!=PILOT_SHA or record['state']!='COMPLETED_DEVICE_COMPONENT_AND_POST_CLEAN'
            or record['slurm'][:4]!=['1103606.674','O01-resident-BF16-graph64-pilot-v1','COMPLETED','0:0']
            or record['independent_post_probe']!='1103606.675'
            or (record['valid_cards'],record['completed_arms'],record['observed_driver_busy_samples'])!=(8,32,128)):
        raise ValueError('original .674 and .675 qualification differs')
    members,files=source(repo)
    campaign_epoch=float(campaign_epoch)
    plan=dict(request=REQUEST,campaign_epoch=campaign_epoch,shared_start_epoch=campaign_epoch+60,
              deadline_epoch=campaign_epoch+125,publication=publication,pilot_SHA256=PILOT_SHA,
              source_SHA256=source_identity(files),before_snapshot_SHA256=before_snapshot_SHA256,selected=selected)
    validate_plan(plan)
    payload=dict(plan=plan,files={name:[sha(raw),base64.b64encode(zlib.compress(raw)).decode('ascii')]
                                for name,raw in sorted(files.items())})
    script=('PAYLOAD='+repr(payload)+'\n').encode('ascii')+files['controller.py']
    return plan,script,[QUALIFIED_PILOT.as_posix(),Path(__file__).resolve().relative_to(REPO).as_posix(),
                        *(path.as_posix() for path in members.values())]
