"""Unified entry for a bound original native recording or fresh resume phase."""
from pathlib import Path
import argparse
import json
import re
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'generation/src'))
from gencuda.zimage_native_record import run


def fixed_first_main(argv, *, generation_accelerator):
    """Registered T07.sft fixed-input diagnostic using one native first update."""
    p = argparse.ArgumentParser(description=fixed_first_main.__doc__)
    p.add_argument('--binding',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--reference-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epoch',type=float,required=True)
    p.add_argument('--deadline',type=float,required=True)
    a = p.parse_args(argv)
    if not all(path.is_absolute() for path in (a.binding,a.reference,a.output)):
        p.error('binding, reference and output must use absolute paths')
    if re.fullmatch(r'[0-9a-f]{64}',a.reference_sha256) is None:
        p.error('reference-sha256 must identify the original progress record')
    with a.binding.open('rb') as stream:
        raw = stream.read(2**20+1)
    if not 0 < len(raw) <= 2**20:
        p.error('invalid bounded native binding')
    binding = json.loads(raw)
    if binding.get('case') != 'T07.sft' or binding.get('platform') != generation_accelerator:
        p.error('fixed first update requires the registered T07.sft accelerator')
    result = run(binding,a.output,mode='first-fixed',epoch=a.epoch,deadline=a.deadline,
                 first_reference=a.reference,first_reference_sha256=a.reference_sha256)
    print(json.dumps(dict(passed=result['passed'],mode=result['mode'],
                          actual_optimizer_updates=result['actual_optimizer_updates'],
                          formal_performance=False,Model_GO=False)))
    return int(not result['passed'])

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--binding',required=True);p.add_argument('--output',required=True)
    p.add_argument('--epoch',type=float,required=True);p.add_argument('--deadline',type=float,required=True)
    p.add_argument('--mode',choices=('continuous','resume20'),required=True);p.add_argument('--checkpoint')
    p.add_argument('--first-resumed-update-only',action='store_true')
    p.add_argument('--warmup-before-restored-update',action='store_true')
    p.add_argument('--profile-resumed-update',action='store_true')
    p.add_argument('--diagnostic-parent-checkpoint',action='store_true')
    a=p.parse_args()
    result=run(json.loads(Path(a.binding).read_bytes()),a.output,mode=a.mode,epoch=a.epoch,deadline=a.deadline,checkpoint=a.checkpoint,
               first_resumed_update_only=a.first_resumed_update_only,warmup_before_restored_update=a.warmup_before_restored_update,profile_resumed_update=a.profile_resumed_update,
               diagnostic_parent_checkpoint=a.diagnostic_parent_checkpoint)
    print(json.dumps(dict(passed=result['passed'],mode=a.mode,actual_optimizer_updates=result['actual_optimizer_updates'],formal_performance=False,Model_GO=False)))

if __name__=='__main__':main()
