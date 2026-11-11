"""Frozen native recipe interpretation keeps the original Gemma schedule."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vlm.runtime.native_recipe_contract import native_recipe_spec


def launch():
    options = dict(max_steps='25', gradient_accumulation_steps='4', per_device_train_batch_size='1',
        save_steps='10', save_total_limit='3', learning_rate='1e-4', weight_decay='.1',
        adam_beta1='.9', adam_beta2='.95', adam_epsilon='1e-8', warmup_ratio='.03',
        optim='adamw_torch', lr_scheduler_type='cosine', torch_dtype='bfloat16',
        gradient_checkpointing='false', freeze_vit='true', target_modules='all-linear',
        lora_rank='16', lora_alpha='32', lora_dropout='.05', attn_impl='sdpa',
        template='gemma3_vision', max_length='512')
    command = ['swift', 'sft']
    for key, value in options.items():
        command.extend(['--' + key, value])
    return dict(case='T02.sft', command=command, a100_launch_sha256='a' * 64)


class NativeRecipeSpecTest(unittest.TestCase):
    def test_original_ratio_warmup_is_not_the_minicpm_five_step_schedule(self):
        value = native_recipe_spec(launch())
        self.assertEqual(value['effective_warmup_steps'], 1)
        self.assertEqual(value['training']['warmup_ratio'], .03)
        self.assertEqual(value['frozen_paths'], ['model.vision_tower', 'model.multi_modal_projector'])
        self.assertFalse(value['training']['gradient_checkpointing'])
        self.assertEqual(value['original_launch_SHA256'], 'a' * 64)

    def test_ambiguous_missing_changed_and_unqualified_recipes_rejected(self):
        for kind in ('duplicate', 'missing', 'rank', 'length', 'steps', 'case'):
            value = launch()
            command = value['command']
            if kind == 'duplicate':
                command.extend(['--max_steps', '25'])
            elif kind == 'missing':
                i = command.index('--warmup_ratio')
                del command[i:i + 2]
            elif kind == 'case':
                value['case'] = 'T05.sft'
            else:
                key, changed = dict(rank=('--lora_rank', '4'), length=('--max_length', '511'),
                                    steps=('--max_steps', '2'))[kind]
                command[command.index(key) + 1] = changed
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                native_recipe_spec(value)


if __name__ == '__main__':
    unittest.main()
