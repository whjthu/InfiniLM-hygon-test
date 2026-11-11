"""Keep full paired denominators and reject incomplete native witness records."""
import copy
import hashlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'vlm/runtime'))
from resume_pair import compare_resume_pair


def identity(value):
    return dict(SHA256=hashlib.sha256(str(value).encode()).hexdigest(), finite=True)


def record(start):
    return dict(rank=0, world_size=8, native_resume_window_complete=True,
        plan=dict(checkpoint_global_step=start, unchanged_max_steps=25,
                  gradient_accumulation_steps=4, world_size_to_verify=8, per_device_batch=1,
                  dataloader_batches_per_rank_to_verify=8, effective_batch=32),
        saved_checkpoints=[20, 25] if start == 10 else [25], RNG_loads=[dict(saved_equal=True)],
        inputs=[dict(ordinal=i + 1, global_step=start + i // 4,
                     full_input=identity(('input', start * 4 + i))) for i in range((25 - start) * 4)],
        outputs=[dict(ordinal=i + 1, loss=identity(('loss', start * 4 + i)))
                 for i in range((25 - start) * 4)],
        steps=[dict(global_step=step, state={name: identity((step, name))
               for name in ('adapter', 'optimizer', 'scheduler')}) for step in range(start + 1, 26)])


class ResumePairTest(unittest.TestCase):
    def test_complete_tail_comparison_is_not_whole_recovery_acceptance(self):
        row = compare_resume_pair(record(10), record(20))
        self.assertEqual(row['input_equal_forwards'], 20)
        self.assertEqual(row['exact_loss_forwards'], 20)
        self.assertEqual(row['exact_full_state_updates'], 5)
        self.assertFalse(row['recovery_qualified'])

    def test_bad_denominators_or_saved_boundary_rejected(self):
        for mutation in ('input', 'checkpoint', 'state', 'RNG', 'ordinal'):
            control = record(10)
            if mutation == 'input':
                control['inputs'].pop()
            elif mutation == 'checkpoint':
                control['saved_checkpoints'] = [25]
            elif mutation == 'state':
                control['steps'][0]['state'].pop('optimizer')
            elif mutation == 'RNG':
                control['RNG_loads'] = []
            else:
                control['inputs'][-1]['ordinal'] -= 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                compare_resume_pair(control, record(20))

    def test_mismatches_stay_in_the_complete_twenty_forward_denominator(self):
        control, resumed = record(10), record(20)
        resumed['inputs'][0]['full_input'] = identity('different input')
        resumed['outputs'][1]['loss'] = identity('different loss')
        resumed['steps'][-1]['state']['optimizer'] = identity('different optimizer')
        before = copy.deepcopy(resumed)
        row = compare_resume_pair(control, resumed)
        self.assertEqual(resumed, before)
        self.assertEqual(row['compared_tail_forwards'], 20)
        self.assertEqual(row['input_equal_forwards'], 19)
        self.assertEqual(row['exact_loss_forwards'], 19)
        self.assertEqual(row['exact_full_state_updates'], 4)


if __name__ == '__main__':
    unittest.main()
