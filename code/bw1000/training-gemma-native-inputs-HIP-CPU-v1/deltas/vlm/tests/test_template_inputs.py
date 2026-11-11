"""Do not accept incomplete, reordered or mismatched native input evidence."""
import copy
import hashlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vlm.runtime.template_inputs import compare_input_identities, settings_from_launch


def records():
    return [dict(row_ordinal=i, fields={k: dict(SHA256=hashlib.sha256((k + str(i)).encode()).hexdigest(),
                finite=True, tensors=0, elements=0) for k in ('input_ids', 'labels', 'pixel_values')})
            for i in range(1, 65)]


class TemplateInputTest(unittest.TestCase):
    def test_complete_identity_is_not_first_update_acceptance(self):
        value = compare_input_identities(records(), records())
        self.assertEqual(value['exact_rows'], 64)
        self.assertFalse(value['first_update_numerics_qualified'])
        self.assertFalse(value['model_loaded'])

    def test_missing_and_reordered_rows_rejected(self):
        for mutation in ('missing', 'reordered'):
            changed = records()
            if mutation == 'missing':
                changed.pop()
            else:
                changed[0], changed[1] = changed[1], changed[0]
            with self.assertRaises(ValueError):
                compare_input_identities(records(), changed)

    def test_changed_label_pixel_and_missing_field_remain_visible(self):
        changed = records()
        changed[0]['fields']['labels']['SHA256'] = '0' * 64
        changed[1]['fields']['pixel_values']['SHA256'] = '1' * 64
        del changed[2]['fields']['pixel_values']
        original = copy.deepcopy(changed)
        result = compare_input_identities(records(), changed)
        self.assertEqual(result['compared_rows'], 64)
        self.assertEqual(result['differing_rows'], 3)
        self.assertEqual(changed, original)

    def test_missing_labels_and_invalid_identity_rejected(self):
        for mutation in ('labels', 'hash', 'finite'):
            changed = records()
            if mutation == 'labels':
                del changed[0]['fields']['labels']
            elif mutation == 'hash':
                changed[0]['fields']['labels']['SHA256'] = 'g' * 64
            else:
                changed[0]['fields']['labels']['finite'] = False
            with self.assertRaises(ValueError):
                compare_input_identities(records(), changed)

    def test_original_template_length_and_seed_are_read_without_defaults(self):
        args = ['swift', 'sft', '--model_type', 'gemma3_vision', '--template', 'gemma3_vision',
                '--max_length', '512', '--max_pixels', '262144', '--padding_side', 'right',
                '--padding_free', 'false', '--seed', '42', '--data_seed', '42',
                '--torch_dtype', 'bfloat16', '--attn_impl', 'sdpa']
        value = settings_from_launch(dict(command=args, case='T02.sft', a100_launch_sha256='0' * 64))
        self.assertEqual((value['max_length'], value['template_type'], value['seed']),
                         (512, 'gemma3_vision', 42))
        with self.assertRaises(ValueError):
            settings_from_launch(dict(command=args + ['--seed', '43'], case='T02.sft', a100_launch_sha256='0' * 64))


if __name__ == '__main__':
    unittest.main()
