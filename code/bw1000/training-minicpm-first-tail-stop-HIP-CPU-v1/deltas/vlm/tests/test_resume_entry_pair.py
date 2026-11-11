"""Retain first-tail denominators and reject unbound or incomplete snapshots."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vlm.runtime.resume_entry_pair import compare_forward_entry_pair
from vlm.tests.test_resume_pair import record, identity


def pair_record(start):
    value = record(start)
    value.update(native_resume_window_complete=False, declared_observation_window_complete=True,
                 diagnostic_stop_requested=True, diagnostic_stop_step=21, completed_global_step=21,
                 entry_state_step=20, entry_state_captured=True,
                 saved_checkpoints=[20] if start == 10 else [])
    value['inputs'] = value['inputs'][:(21 - start) * 4]
    value['outputs'] = value['outputs'][:(21 - start) * 4]
    value['steps'] = [r for r in value['steps'] if r['global_step'] <= 21]
    value['plan'].update(declared_observation_micro_forwards=(21 - start) * 4,
                         planned_micro_forwards_per_rank=(25 - start) * 4)
    return value


def snapshot(start):
    rows = [dict(kind='parameter', name='base', present=True, requires_grad=False,
                 SHA256=identity('frozen')['SHA256'])]
    return dict(schema=1, global_step=20, forward_ordinal=(20 - start) * 4 + 1,
                all_registered_parameters_and_buffers_observed=True, observer_RNG_unchanged=True,
                registered_state=rows, RNG=identity('RNG'), context={}, input_layouts={}, input_alignments={},
                selected_module_runtime={}, loader_rng_types=['generator'], accelerator_rng_types=['generator'],
                runtime_cursor={})


def bind(record, entry):
    entry['registered_state_SHA256'] = hashlib.sha256(json.dumps(entry['registered_state'], sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    raw = json.dumps(entry).encode()
    record['entry_state_SHA256'] = hashlib.sha256(raw).hexdigest()
    return raw


class EntryPairTest(unittest.TestCase):
    def compare(self, a, b, x, y):
        return compare_forward_entry_pair(a, b, bind(a, x), bind(b, y))

    def test_equal_metadata_never_qualifies_full_recovery(self):
        row = self.compare(pair_record(10), pair_record(20), snapshot(10), snapshot(20))
        self.assertEqual(row['input_equal_forwards'], 4)
        self.assertEqual(row['exact_loss_forwards'], 4)
        self.assertTrue(row['registered_state_equal'])
        self.assertTrue(all(row['entry_fields'].values()))
        self.assertFalse(row['full25_native_recovery_compared'])
        self.assertFalse(row['recovery_qualified'])
        self.assertFalse(row['numerical_cause_attributed'])

    def test_entry_and_loss_mismatches_preserve_all_four_occurrences(self):
        a, b, x, y = pair_record(10), pair_record(20), snapshot(10), snapshot(20)
        y['RNG'] = identity('different RNG')
        y['registered_state'][0]['SHA256'] = identity('different frozen')['SHA256']
        b['outputs'][0]['loss'] = identity('different loss')
        row = self.compare(a, b, x, y)
        self.assertEqual(row['compared_tail_forwards'], 4)
        self.assertEqual(row['exact_loss_forwards'], 3)
        self.assertFalse(row['entry_fields']['RNG_equal'])
        self.assertEqual(row['changed_registrations'], [dict(kind='parameter', name='base')])

    def test_bad_window_or_stop_request_is_rejected(self):
        for field in ('inputs', 'steps', 'saved_checkpoints', 'diagnostic_stop_requested'):
            a, b, x, y = pair_record(10), pair_record(20), snapshot(10), snapshot(20)
            if field in ('inputs', 'steps'):
                a[field].pop()
            elif field == 'saved_checkpoints':
                a[field] = []
            else:
                a[field] = False
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.compare(a, b, x, y)

    def test_unbound_snapshot_bytes_are_rejected(self):
        a, b, x, y = pair_record(10), pair_record(20), snapshot(10), snapshot(20)
        raw_a, raw_b = bind(a, x), bind(b, y)
        with self.assertRaisesRegex(ValueError, 'bytes differ'):
            compare_forward_entry_pair(a, b, raw_a + b' ', raw_b)

    def test_missing_runtime_and_duplicate_registration_are_rejected(self):
        for mutation in ('runtime', 'duplicate', 'rank'):
            a, b, x, y = pair_record(10), pair_record(20), snapshot(10), snapshot(20)
            if mutation == 'runtime':
                del y['loader_rng_types']
            elif mutation == 'duplicate':
                y['registered_state'].append(copy.deepcopy(y['registered_state'][0]))
            else:
                b['rank'] = 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.compare(a, b, x, y)


if __name__ == '__main__':
    unittest.main()
