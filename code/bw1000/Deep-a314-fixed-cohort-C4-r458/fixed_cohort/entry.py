"""Explicit fixed-cohort entry; no HTTP, default public factory or scheduler."""
from pathlib import Path
import hashlib
import json
from .inputs import load_four
from .profile import PROFILE, binding_descriptor, config as make_config
from .checkpoint_loader import CONFIG_SHA, INDEX_SHA, bind_library
from .native_context import FixedCohortContext
from .continuation import run

MODES = ('serial', 'cohort', 'serial', 'cohort', 'cohort', 'serial')


def prepare(config_path):
    value = json.loads(Path(config_path).read_bytes())
    assert value['profile'] == PROFILE and value['rows'] == 4
    assert value['capacity_per_row'] == 32768 and value['cohort_modes'] == list(MODES)
    assert value['fullpre'] == value['r366'] == value['HC_mode'] == 0
    # This exact parent binding performs source/receipt/50objects/dependency
    # verification only. It does not dlopen Native or construct a Runtime.
    from .native_binding import binding
    parent, receipt, normalized = binding()
    descriptor = binding_descriptor(parent)
    model = Path(value['model_path'])
    raw = (model / 'config.json').read_bytes()
    index = (model / 'model.safetensors.index.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == CONFIG_SHA
    assert hashlib.sha256(index).hexdigest() == INDEX_SHA
    records = load_four(value['encoding'], value['gold'])
    return dict(config=make_config(value['model_path']), cfg=json.loads(raw),
                mapping=json.loads(index)['weight_map'], records=records,
                descriptor=descriptor, receipt=receipt, normalized=normalized)


def execute(prepared, *, torch, safe_open, report, deadline_check, deadline,
            save, canceled=lambda: False):
    """Future actual device caller. Current CPU claim never invokes this."""
    receipt = prepared['receipt']
    binding = bind_library(receipt['library'], prepared['config']['Native_SHA256'])
    with FixedCohortContext(binding, receipt, config=prepared['config'],
            cfg=prepared['cfg'], mapping=prepared['mapping'], torch=torch,
            safe_open=safe_open, report=report, deadline_check=deadline_check) as context:
        report.update(profile=PROFILE, actual_rows=4, capacity_per_row=32768,
                      standard_public_C1=False, fixed_cohort_trials=[])
        for ordinal, mode in enumerate(MODES):
            context.check_thread()
            trial = run(context, prepared['records'], mode=mode, deadline=deadline,
                        save=lambda phase, value: save(ordinal, phase, value), canceled=canceled)
            trial.update(ordinal=ordinal, warm=ordinal < 2)
            report['fixed_cohort_trials'].append(trial)
            if canceled():
                break
    return report
