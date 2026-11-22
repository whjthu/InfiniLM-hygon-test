"""Layer a single generator change onto the frozen native first-forward plugin."""

from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from types import ModuleType

import torch
from swift.callbacks import callbacks_map


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:
    from rng_treatment import ARMS, apply_rng, describe_inputs, need, observed_model, owner
finally:
    sys.path.pop(0)

KEY = 'minicpm_usgs_resume_witness'
OLD_OBSERVER = ('native_observer.py', 3904, '2d875c5b46123482bf69cb3833458a37c3c5dd91ef7b6648ffed92caeeea600c')
OLD_PROBE = ('probe.py', 10995, '1d340abb7659f30cb625fd21164f47168b16267fccf86f9cb8da73edcc6bfdc4')


def exact(directory, identity):
    name, size, expected = identity
    path = directory / name
    need(path.resolve(strict=True) == path, 'aliased immutable observer source')
    with path.open('rb') as stream:
        raw = stream.read(size + 1)
    need(len(raw) == size and sha256(raw).hexdigest() == expected, 'unexpected .1189 observer byte identity')
    return path, raw


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def bind():
    directory = Path(os.environ['MINICPM_FIRST_FORWARD_ORIGINAL_OBSERVER']).resolve(strict=True)
    path, raw = exact(directory, OLD_OBSERVER)
    exact(directory, OLD_PROBE)
    arm = os.environ['MINICPM_RNG_TREATMENT']
    need(arm in ARMS and KEY not in callbacks_map, 'RNG arm or original callback namespace differs')
    old_module = ModuleType('minicpm_original_1189_observer')
    old_module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), old_module.__dict__)
    original = callbacks_map[KEY]

    class NativeRNGArm(original):
        def on_train_begin(self, args, state, control, **kwargs):
            result = super().on_train_begin(args, state, control, **kwargs)
            old_compute = self.observer.bound_call
            need(self.trainer.compute_loss is old_compute, 'original first-forward observer is absent')
            witness = self.witness

            def intervene(*call, **options):
                self.trainer.compute_loss = old_compute
                handles = []
                input_before = None
                input_unchanged = None

                def before(_module, argv, values):
                    nonlocal input_before
                    need(input_before is None, 'unexpected additional model forward')
                    input_before = describe_inputs((argv, values), torch)

                def after(_module, argv, values, _output):
                    nonlocal input_unchanged
                    input_unchanged = input_before == describe_inputs((argv, values), torch)

                try:
                    self.check_time()
                    need(len(witness.rng_loads) == 1 and witness.rng_loads[0]['saved_equal'] is True
                         and not witness.inputs and not witness.steps and self.trainer.state.global_step == 20,
                         'RNG intervention must follow the original native cursor and RNG load')
                    # Imported globals of the actual native witness. Use its exact
                    # structured full-state fingerprint, as the archived probe did.
                    core = witness.before.__func__.__globals__
                    rng_values = core['rng_values']
                    rng_fingerprint = core['rng_fingerprint']
                    parameters, model_before = observed_model(torch, self.model, self.check_time)
                    handles.append(self.model.register_forward_pre_hook(before, prepend=True, with_kwargs=True))
                    handles.append(self.model.register_forward_hook(after, with_kwargs=True))
                    treatment = apply_rng(torch, rng_values, rng_fingerprint, witness.rng_loads[0]['actual'], self.rank, arm)
                    actual_step = os.environ['SLURM_JOB_ID'] + '.' + os.environ['SLURM_STEP_ID']
                    run = dict(owner(), arm=arm, rank=self.rank, world_size=8, checkpoint_step=20, actual_step=actual_step,
                               config_SHA256=os.environ['MINICPM_USGS_CONFIG_SHA256'])
                    witness.write('rng-treatment.json', dict(run=run, treatment=treatment))
                    try:
                        old_compute(*call, **options)
                    except SystemExit as finished:
                        need(type(finished.code) is int and finished.code == 0, 'unexpected original observer exit')
                        self.check_time()
                        current, model_after = observed_model(torch, self.model, self.check_time)
                        need(parameters == current and input_unchanged is True and not witness.steps
                             and self.trainer.state.global_step == 20, 'parameters, root input or step mutated')
                        witness.write('rng-state-audit.json', dict(
                            run=run, parameter_identity_and_versions_unchanged=True, root_input_identity_and_versions_unchanged=True,
                            before=digest(model_before), after=digest(model_after),
                            training_modes_unchanged=model_before['training'] == model_after['training'],
                            observed_buffer_rotary_changed=model_before['observed_values'] != model_after['observed_values'],
                            opaque_cache_attrs_before=model_before['opaque_cache_attrs'],
                            opaque_cache_attrs_after=model_after['opaque_cache_attrs'],
                            observed_bytes_before=model_before['observed_bytes'], observed_bytes_after=model_after['observed_bytes'],
                            backward_calls=0, optimizer_updates=0))
                        raise
                    raise RuntimeError('original observer returned beyond the zero-update boundary')
                except Exception as failure:
                    self.observer.detach()
                    if not witness.failed:
                        witness.fail(failure)
                    raise
                finally:
                    for handle in handles:
                        handle.remove()

            self.trainer.compute_loss = intervene
            return result

    callbacks_map[KEY] = NativeRNGArm


bind()
