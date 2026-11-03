"""Set the real step/original epoch once, then exec the fixed diagnostic entry."""
import json
import os
from pathlib import Path
import sys
import time
from support import D, R, RUN, publish, sha, verify


def main():
    m = verify(os.environ['DEEP438_DIAG_CONTROLLER_SHA'], require_ready=True)
    cfg = json.loads((D / 'config.json').read_bytes())
    epoch = float(os.environ['DEEP438_DIAG_START'])
    assert time.time() + 120 < epoch + 2700
    cfg.update(gpu_start_epoch=epoch,
               actual_step=os.environ['SLURM_JOB_ID'] + '.' + os.environ['SLURM_STEP_ID'])
    assert cfg['actual_step'].startswith('910374.')
    output = R / 'results' / ('Deep447-boundary-' + RUN + '-actual-config.json')
    publish(output, cfg)
    # Same absolute path is mounted in the container by the retained wrapper.
    entry = Path('/refactor/incoming/Deep-layer2-3-boundary-diagnostic-r447/entry-v2/entry.py')
    os.execv(sys.executable, [sys.executable, '-B', '-u', str(entry),
             '--config', str(output), '--config-sha', sha(output),
             '--pin', str(D / 'binding-pin.json'), '--pin-sha', sha(D / 'binding-pin.json'),
             '--source-sha', m['entry_manifest_SHA256'], '--execute'])


if __name__ == '__main__':
    main()
