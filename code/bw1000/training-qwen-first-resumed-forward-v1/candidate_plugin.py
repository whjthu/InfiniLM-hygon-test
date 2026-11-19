"""A rank-local text-only SDPA math route for the Qwen recovery experiment."""
from collections import Counter
from functools import wraps
import json
import os
from pathlib import Path

from swift.callbacks import TrainerCallback, callbacks_map
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen

from first_forward_trace import install

KEY = 'qwen_math_sdpa_recovery'
install()


class QwenMathSdpaRecovery(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        rank = int(os.environ['RANK'])
        phase = os.environ['QWEN_PAIR_PHASE']
        expected_step = {'continuous': 0, 'resumed': 20}[phase]
        if not (torch.version.hip and torch.cuda.current_device() == rank
                and 0 <= rank < 8 and os.environ['WORLD_SIZE'] == '8'
                and state.global_step == expected_step and args.max_steps == 25):
            raise ValueError('Unexpected native Qwen rank or step')
        registry = qwen.ALL_ATTENTION_FUNCTIONS
        original = registry['sdpa']
        if original is not sdpa_attention_forward:
            raise ValueError('Unexpected original Qwen SDPA provider')
        self.initial_deterministic = torch.are_deterministic_algorithms_enabled()
        self.initial_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(True, warn_only=False)
        self.assert_policy()
        self.original = original
        self.masks = Counter()
        self.layers = set()
        self.longest = 0

        @wraps(original)
        def route(module, query, key, value, attention_mask, *args, **kw):
            if not isinstance(module, qwen.Qwen3VLTextAttention):
                return original(module, query, key, value, attention_mask, *args, **kw)
            if query.dtype != torch.bfloat16 or query.device.type != 'cuda':
                raise ValueError('Expected BF16 Qwen text attention on DCU')
            self.assert_policy()
            self.masks['implicit' if attention_mask is None else str(attention_mask.dtype)] += 1
            self.longest = max(self.longest, query.shape[-2])
            self.layers.add(module.layer_idx)
            with sdpa_kernel(SDPBackend.MATH):
                if (not torch.backends.cuda.math_sdp_enabled()
                        or torch.backends.cuda.flash_sdp_enabled()
                        or torch.backends.cuda.mem_efficient_sdp_enabled()
                        or torch.backends.cuda.cudnn_sdp_enabled()):
                    raise ValueError('Only the requested SDPA backend may be enabled')
                return original(module, query, key, value, attention_mask, *args, **kw)

        self.wrapper = route
        registry.register('sdpa', route)
        if registry['sdpa'] is not route:
            raise ValueError('Qwen SDPA registry did not adopt the bound route')
        self.rank = rank
        self.phase = phase
        return control

    @staticmethod
    def assert_policy():
        if (not torch.are_deterministic_algorithms_enabled()
                or torch.is_deterministic_algorithms_warn_only_enabled()):
            raise ValueError('Strict deterministic torch policy was not effective')

    def on_train_end(self, args, state, control, **kwargs):
        self.assert_policy()
        registry = qwen.ALL_ATTENTION_FUNCTIONS
        if registry['sdpa'] is not self.wrapper:
            raise ValueError('Qwen SDPA registry changed during training')
        registry.register('sdpa', self.original)
        if state.global_step != 25 or self.layers != set(range(36)) or not self.masks:
            raise ValueError('Missing model text attention coverage')
        # Counts include activation-checkpoint recomputation when invoked.
        out = Path(args.output_dir) / f'math-sdpa-rank-{self.rank:02d}.json'
        with out.open('x', encoding='utf-8') as handle:
            json.dump(dict(rank=self.rank, phase=self.phase, completed_step=state.global_step,
                           mode='MATH', deterministic=True, warn_only=False,
                           previously_deterministic=self.initial_deterministic, previous_warn_only=self.initial_warn_only,
                           mask_calls=self.masks, text_layers=len(self.layers),
                           longest_query=self.longest, native_provider=self.original.__module__),
                      handle, allow_nan=False)
        torch.use_deterministic_algorithms(self.initial_deterministic, warn_only=self.initial_warn_only)
        return control


if KEY in callbacks_map:
    raise ValueError('Qwen math SDPA callback is already registered')
callbacks_map[KEY] = QwenMathSdpaRecovery
