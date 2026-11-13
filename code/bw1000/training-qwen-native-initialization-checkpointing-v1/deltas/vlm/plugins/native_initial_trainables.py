"""Restore original trainables before the native recorder and baseline digest."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from swift.callbacks import TrainerCallback, callbacks_map

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from vlm.runtime.first_update_capture import remaining
from vlm.runtime.initial_trainables import restore_initial_trainables, validate_initialization_callback_order
from vlm.runtime.native_recipe_contract import validate_native_recipe

KEY = 'native_initial_trainables'


class NativeInitialTrainables(TrainerCallback):
    def __init__(self, swift_args, trainer):
        super().__init__(swift_args, trainer)
        self.trainer = trainer

    def on_train_begin(self, args, state, control, **kw):
        epoch = float(os.environ['FIRST_UPDATE_EPOCH'])
        deadline = float(os.environ['FIRST_UPDATE_DEADLINE'])
        maximum = float(os.environ['FIRST_UPDATE_MAX_SECONDS'])
        remaining(epoch, deadline, time.time(), maximum)
        path = Path(os.environ['FIRST_UPDATE_BINDING'])
        if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= 2 * 1024**2:
            raise ValueError('initialization binding path or length differs')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != os.environ['FIRST_UPDATE_BINDING_SHA256']:
            raise ValueError('initialization binding SHA differs')
        binding = json.loads(raw)
        initial = binding['initial_state']
        if not isinstance(initial, dict) or set(initial) != {'path', 'bytes', 'SHA256'}:
            raise ValueError('expected exact original initial-state file binding')
        rank = int(os.environ['RANK'])
        if int(os.environ['WORLD_SIZE']) != 8 or not 0 <= rank < 8:
            raise ValueError('expected original eight-rank initialization')
        validate_initialization_callback_order(self.trainer.callback_handler.callbacks, self,
                                              recorder_name='NativeFirstUpdates')
        model = kw.get('model', self.trainer.model)
        checkpointing_method = getattr(self.trainer, '_prepare_gradient_checkpointing', None)
        named, _ = validate_native_recipe(
            model, args, binding, checkpointing_method=checkpointing_method)
        # The recorder owns its exclusive capture directory. Initialization uses
        # a sibling directory so it cannot pre-create or overwrite that output.
        output = Path(os.environ['FIRST_UPDATE_CAPTURE']).with_name(
            Path(os.environ['FIRST_UPDATE_CAPTURE']).name + '-initialization') / ('rank-%02d.json' % rank)
        output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        # Reserve the receipt before modifying parameters. Existing receipts are
        # a duplicate-run error, including a failed prior initialization.
        with output.open('x', encoding='utf8') as stream:
            try:
                proof = restore_initial_trainables(named, self.trainer.optimizer,
                    global_step=state.global_step, path=initial['path'], file_bytes=initial['bytes'],
                    file_SHA256=initial['SHA256'], named_SHA256=binding['initial_trainables_SHA256'])
                remaining(epoch, deadline, time.time(), maximum)
                proof.update(passed=True, rank=rank, world_size=8,
                             binding_SHA256=hashlib.sha256(raw).hexdigest(),
                             scheduler_restored=False, data_cursor_restored=False,
                             numerical_alignment_qualified=False)
                json.dump(proof, stream, indent=2)
            except BaseException as error:
                json.dump(dict(passed=False, rank=rank, error_type=type(error).__name__,
                               numerical_alignment_qualified=False), stream)
                raise
        return control


if KEY in callbacks_map:
    raise ValueError('duplicate native initial-trainable callback')
callbacks_map[KEY] = NativeInitialTrainables
