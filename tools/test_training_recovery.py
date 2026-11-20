"""Recovery acceptance tests using the original DP2 step layout."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training_recovery import STATE_FIELDS, verify_recovery
from verify_training_acceptance import verify as verify_matrix


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


class RecoveryGateTest(unittest.TestCase):
    def setUp(self):
        self.attachments = {}
        self.framework = 'f' * 40
        self.original_launch = 'e' * 64
        self.case = {
            'case': 'T09.dp2',
            'original_cards': 2,
            'launch_SHA256': self.original_launch,
            'recovery_plan': {
                'original_launch_SHA256': self.original_launch,
                'checkpoint_steps': [10, 20, 25],
                'resume_steps': [10, 20],
                'training_ranks': 2,
                'comparison_limits': {'loss_absolute': 0, 'parameter_relative_l2': 0},
                'original_references': [self.pin({'case': 'T09.dp2'})],
            },
        }
        self.continuous = self.receipt(0)
        self.continuous['checkpoints'] = [
            dict(global_step=step, manifest_SHA256=digest(f'checkpoint/step/{step}'),
                 state=self.states(step), manifest=self.checkpoint_inventory(step)) for step in (10, 20, 25)
        ]
        self.resumed = [self.receipt(step) for step in (10, 20)]
        for item in self.resumed:
            step = item['start_global_step']
            item.update(source_checkpoint_SHA256=digest(f'checkpoint/step/{step}'),
                        restored_state=self.states(step))

    def pin(self, data):
        value = digest(json.dumps(data, sort_keys=True))
        self.attachments[value] = copy.deepcopy(data)
        return {'SHA256': value}

    def states(self, step):
        return [dict(rank=rank, accumulation_boundary=0,
                     **{field: digest(f'{field}/{rank}/{step}') for field in STATE_FIELDS}) for rank in range(2)]

    def binding(self):
        return dict(case=self.case['case'], framework_commit=self.framework, platform='Hygon',
                    original_launch_SHA256=self.original_launch)

    def checkpoint_inventory(self, step):
        return self.pin(dict(self.binding(), run_id=self.continuous['run_id'], global_step=step,
                             original_checkpoint_SHA256=digest(f'checkpoint/step/{step}'), state=self.states(step),
                             files=[dict(relative_path=f'rank-{rank}.pt', bytes=512,
                                         SHA256=digest(f'checkpoint/file/{step}/{rank}')) for rank in range(2)]))

    def receipt(self, first):
        run_id = digest(f'run/{first}')
        step = f'1087656.{first+512}'
        terminal = self.pin(dict(self.binding(), run_id=run_id, start_global_step=first, final_global_step=25,
                                 accounting_columns=['JobIDRaw', 'State%40', 'ExitCode', 'NodeList', 'AllocTRES%160'],
                                 accounting_row=[step, 'COMPLETED', '0:0', 'node1', 'cpu=8,gres/dcu=2']))
        return {
            'case': 'T09.dp2', 'framework_commit': self.framework, 'platform': 'Hygon',
            'original_launch_SHA256': self.original_launch, 'terminal_status': 'COMPLETED', 'exit_code': 0,
            'terminal_receipt_SHA256': terminal['SHA256'], 'terminal_evidence': terminal,
            'slurm_step': step, 'run_id': run_id,
            'start_global_step': first, 'final_global_step': 25, 'final_state': self.states(25),
            'rank_processes': [dict(rank=r, process_identity_SHA256=digest(f'host/pid/start/{first}/{r}'))
                               for r in range(2)],
            'updates': [
                dict(global_step=step,
                     ranks=[dict(rank=r, input_SHA256=digest(f'microbatches/{step}/{r}'),
                                 loss=(26-step)/13) for r in range(2)])
                for step in range(first+1, 26)
            ],
        }

    def evidence(self):
        return {'gate': 'recovery', 'passed': True, 'continuous': self.pin(self.continuous),
                'resumed': [self.pin(receipt) for receipt in self.resumed]}

    def verify(self, evidence=None):
        return verify_recovery(self.case, self.framework, evidence or self.evidence(),
                               lambda ref: self.attachments[ref['SHA256']])

    def test_both_dp2_checkpoints_recover_with_two_fresh_ranks(self):
        results = self.verify()
        self.assertEqual([(r['resume_step'], r['reproduced_updates'], r['training_ranks'])
                          for r in results], [(10, 15, 2), (20, 5, 2)])

    def test_self_asserted_pass_does_not_supply_recovery(self):
        with self.assertRaisesRegex(ValueError, 'requires continuous and resumed receipts'):
            self.verify({'gate': 'recovery', 'passed': True})

    def test_rejects_missing_foreign_or_unsuccessful_accounting(self):
        resumed = self.resumed[0]
        original = resumed['terminal_evidence']
        resumed.pop('terminal_evidence')
        with self.assertRaisesRegex(ValueError, 'independently pinned terminal evidence'):
            self.verify()
        actual = self.attachments[original['SHA256']]
        for replacement, reason in (
            (dict(actual, run_id=self.continuous['run_id']), 'different execution'),
            (dict(actual, accounting_row=['1087656.888', *actual['accounting_row'][1:]]), 'no successful Slurm step'),
            (dict(actual, accounting_row=[resumed['slurm_step'], 'RUNNING', *actual['accounting_row'][2:]]),
             'no successful Slurm step'),
            (dict(actual, accounting_row=[*actual['accounting_row'][:-1], 'cpu=8,node=1']), 'registered DCU allocation'),
            (dict(actual, accounting_row=[*actual['accounting_row'][:-1], 'cpu=8,gres/dcu=1']), 'registered DCU allocation'),
        ):
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                terminal = self.pin(replacement)
                resumed.update(terminal_evidence=terminal, terminal_receipt_SHA256=terminal['SHA256'])
                self.verify()

    def test_rejects_checkpoint_inventory_drift_and_final_control_state(self):
        checkpoint = self.continuous['checkpoints'][0]
        original = checkpoint['manifest']
        manifest = self.attachments[original['SHA256']]
        for replacement, reason in (
            (dict(manifest, global_step=20), 'different execution or global step'),
            (dict(manifest, files=[]), 'no persisted files'),
            (dict(manifest, files=manifest['files'] * 2), 'duplicated checkpoint file'),
            (dict(manifest, state=self.states(20)), 'disagrees with recorded rank state'),
        ):
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                checkpoint['manifest'] = self.pin(replacement)
                self.verify()
        checkpoint['manifest'] = original
        self.resumed[0]['final_state'][1]['scheduler_SHA256'] = digest('different final scheduler')
        with self.assertRaisesRegex(ValueError, 'final scheduler, data cursor or accumulation boundary differs'):
            self.verify()

    def test_matrix_entry_rejects_a_recovery_flag_without_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(strict=True)

            def file(path, content):
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                data = content if isinstance(content, str) else json.dumps(content)
                target.write_bytes(data.encode())
                return {'path': path, 'SHA256': digest(data)}

            case = copy.deepcopy(self.case)
            case['recovery_plan']['original_references'] = [file('original.json', {'case': case['case']})]
            case['formal_acceptance'] = {'gate_evidence': {
                gate: file(f'{gate}.json', dict(case=case['case'], framework_commit=self.framework,
                                               gate=gate, passed=True))
                for gate in ('function', 'numerics', 'recovery')
            }}
            groups = {f'T{i:02}': {'cases': [f'T{i:02}.sft']} for i in range(1, 13)}
            groups['T06']['cases'].append('T06.ddpo')
            groups['T07']['cases'].append('T07.flow-grpo')
            groups['T08']['cases'].append('T08.motion-rft')
            groups['T09']['cases'] += ['T09.dp2', 'T09.sage-grpo']
            ids = [name for group in groups.values() for name in group['cases']]
            cases = [dict(case if name == case['case'] else {'case': name}, stage3_accepted=False,
                          reference_evidence=None, hygon_evidence=None) for name in ids]
            matrix = file('matrix.json', dict(framework_commit=self.framework, case_count=17, cases=cases,
                          evidence_SHA256={}, node_ledger=file('ledger.json', {'groups': groups}),
                          stage3={'minimum_A100_throughput_ratio': .9,
                                  'maximum_relative_effect_regression': .05}))
            file('status/bw1000/latest.json', {'unified_training_acceptance_matrix': {
                'matrix': matrix['path'], 'matrix_SHA256': matrix['SHA256'],
                'cases': 17, 'stage3_passed': 0, 'framework_commit': self.framework,
            }})
            with self.assertRaisesRegex(ValueError, 'requires the registered case-specific comparison plan'):
                verify_matrix(root)
            with patch('verify_training_acceptance.verify_numerics') as numerics:
                with self.assertRaisesRegex(ValueError, 'requires continuous and resumed receipts'):
                    verify_matrix(root)
                numerics.assert_called_once()

    def test_accepts_nonexact_checkpoint_with_an_original_limit_and_bound_comparison(self):
        self.case['recovery_plan']['comparison_limits']['parameter_relative_l2'] = .01
        final = self.resumed[0]['final_state']
        final[0]['parameters_SHA256'] = digest('measured recovered parameters')
        evidence = self.evidence()
        evidence['parameter_comparisons'] = {
            '10': self.pin(dict(case=self.case['case'], framework_commit=self.framework, platform='Hygon',
                                original_launch_SHA256=self.original_launch, global_step=25,
                                continuous_run_SHA256=evidence['continuous']['SHA256'],
                                resumed_run_SHA256=evidence['resumed'][0]['SHA256'],
                                continuous_parameters_SHA256=[row['parameters_SHA256']
                                                              for row in self.continuous['final_state']],
                                resumed_parameters_SHA256=[row['parameters_SHA256'] for row in final],
                                maximum_rank_relative_l2=.006))
        }
        self.assertEqual(self.verify(evidence)[0]['max_parameter_relative_l2'], .006)
        comparison = copy.deepcopy(self.attachments[evidence['parameter_comparisons']['10']['SHA256']])
        comparison['resumed_run_SHA256'] = evidence['resumed'][1]['SHA256']
        evidence['parameter_comparisons']['10'] = self.pin(comparison)
        with self.assertRaisesRegex(ValueError, 'parameter comparison differs'):
            self.verify(evidence)

    def test_rejects_process_checkpoint_rank_input_and_loss_regressions(self):
        base = copy.deepcopy(self.resumed)
        edits = [
            ('distinct process', lambda arms: arms[0]['rank_processes'][1].update(
                process_identity_SHA256=self.continuous['rank_processes'][1]['process_identity_SHA256'])),
            ('missing or extra', lambda arms: arms.pop()),
            ('wrong.*global steps|differs.*global steps', lambda arms: arms[0].update(final_global_step=24)),
            ('unordered', lambda arms: arms[1]['updates'].pop()),
            ('incomplete recovery rank', lambda arms: arms[1]['restored_state'].pop()),
            ('restored checkpoint state', lambda arms: arms[0].update(source_checkpoint_SHA256=digest('wrong'))),
            ('different input', lambda arms: arms[1]['updates'][0]['ranks'][1].update(input_SHA256=digest('wrong'))),
            ('loss exceeds', lambda arms: arms[1]['updates'][0]['ranks'][1].update(loss=12)),
        ]
        for key in ('rng_SHA256', 'optimizer_SHA256', 'scheduler_SHA256',
                    'parameters_SHA256', 'data_cursor_SHA256', 'accumulation_boundary'):
            value = 1 if key == 'accumulation_boundary' else digest('wrong state')
            edits.append(('restored checkpoint state', lambda arms, k=key, v=value:
                          arms[1]['restored_state'][1].update({k: v})))
        for message, mutate in edits:
            with self.subTest(message=message):
                self.resumed = copy.deepcopy(base)
                mutate(self.resumed)
                with self.assertRaisesRegex(ValueError, message):
                    self.verify()

    def test_rejects_missing_scheduled_continuous_checkpoint(self):
        self.continuous['checkpoints'].pop(1)
        with self.assertRaisesRegex(ValueError, 'incomplete continuous checkpoint'):
            self.verify()


if __name__ == '__main__':
    unittest.main()
