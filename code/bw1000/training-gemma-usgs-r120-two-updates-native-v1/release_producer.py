"""Compose R120's independent read-only release evidence from an owner physical proof."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from contract import NODE, PARENT, REQUEST
from r120_holder import ORIGINAL, holder_gate, step_number, MINIMUM_LATEST_HOLDER
from utc_accounting import collect_exact, read_pinned, validate


def attach(physical, physical_sha, accounting, now):
    required={'request_id','node','parent','independent','released','same_boot_and_pci','open_kfd_holders',
              'scope_owned_processes_remaining','holder_step','post_step','original_release_receipt_SHA256'}
    if set(physical)!=required or not re.fullmatch(r'[0-9a-f]{64}',physical_sha):
        raise ValueError('R120_EXACT_INDEPENDENT_PHYSICAL_INPUT')
    if physical['request_id']!=REQUEST or physical['node']!=NODE or physical['parent']!=PARENT:
        raise ValueError('R120_RELEASE_WRONG_SCOPE')
    holder=physical['holder_step']; post=physical['post_step']
    if step_number(holder)<MINIMUM_LATEST_HOLDER or step_number(post)<=step_number(holder):
        raise ValueError('R120_CURRENT_HOLDER_AND_POST_REQUIRED')
    envelope=validate(accounting)
    if envelope['requested_steps']!=[*ORIGINAL,holder,post]:
        raise ValueError('R120_EXACT_RELEASE_QUERY_STEP_ORDER')
    output={k:v for k,v in physical.items() if k not in ('holder_step','post_step')}
    output.update(UTC_accounting=envelope,physical_projection_SHA256=physical_sha,
                  holder_accounting=envelope['rows'][-2],post_accounting=envelope['rows'][-1])
    holder_gate(output,envelope['rows'],lambda host:{host},now)
    return output


def produce(physical, physical_sha, now, collect=collect_exact):
    holder,post=physical['holder_step'],physical['post_step']
    if step_number(holder)<MINIMUM_LATEST_HOLDER or step_number(post)<=step_number(holder):
        raise ValueError('R120_CURRENT_HOLDER_AND_POST_REQUIRED')
    return attach(physical,physical_sha,collect([*ORIGINAL,holder,post]),now)


def main():
    import time
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--physical',type=Path,required=True)
    parser.add_argument('--physical-sha256',required=True)
    args=parser.parse_args()
    physical=read_pinned(args.physical,args.physical_sha256)
    proof=produce(physical,args.physical_sha256,time.time())
    print(json.dumps(proof,sort_keys=True,allow_nan=False))


if __name__=='__main__':
    main()
