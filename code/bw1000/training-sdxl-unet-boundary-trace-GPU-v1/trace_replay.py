"""Use the pinned original CUDA/HIP replay to observe the same native boundaries."""
from pathlib import Path
import argparse
import json
import os

from boundary_probe import BoundaryProbe
import hygon_sdxl_first_step as frozen_replay


def main():
    parser = argparse.ArgumentParser()
    for name in ('reference','policy','weights','output'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--accelerator', choices=('cuda','hip'), required=True)
    args = parser.parse_args()
    from diffusers import UNet2DConditionModel

    binding = frozen_replay.verify_reference(args.reference,args.policy)
    args.output.mkdir(mode=0o700, exist_ok=False)
    with BoundaryProbe(UNet2DConditionModel) as trace:
        frozen_replay.replay(args.reference,args.weights,args.output/'replay',binding,
                             accelerator=args.accelerator)
        evidence = trace.report()
    data = json.loads((args.output/'replay/summary.json').read_bytes())
    evidence.update(platform=args.accelerator, original_replay_evidence=data,
                    source_replay_SHA256=frozen_replay.sha(Path(frozen_replay.__file__)),
                    worker_PID=os.getpid(), tensor_values_serialized_by_observer=False)
    with (args.output/'boundary-trace.json').open('x',encoding='utf-8') as f:
        json.dump(evidence,f,indent=2,allow_nan=False)
        f.write('\n')


if __name__=='__main__':
    main()
