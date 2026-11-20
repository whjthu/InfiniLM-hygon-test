"""Check that only a real execution of the verified shared case entry can qualify."""
import unittest

from training_function import verify_function


class FunctionalExecutionTest(unittest.TestCase):
    def setUp(self):
        self.case = dict(case='T09.dp2', launch_SHA256='1'*64, original_cards=2)
        self.commit, self.tree = '2'*40, '3'*40
        self.entry = 'tools/hygon_training_recipe.py'
        self.registry = dict(source=dict(framework_commit=self.commit, framework_tree=self.tree,
                                         manifest_SHA256='4'*64), entry_SHA256='5'*64)
        self.binding = dict(case=self.case['case'], original_launch_SHA256=self.case['launch_SHA256'],
                            framework_commit=self.commit, platform='Hygon', run_id='6'*64)
        self.step = '1087656.100'
        self.launch = dict(self.binding, framework_tree=self.tree, entry_path=self.entry,
                           source_manifest_SHA256='4'*64, entry_SHA256='5'*64, slurm_step=self.step,
                           selected_case=self.case['case'], run_requested=True, dry_run=False)
        self.terminal = dict(self.binding, launch_receipt_SHA256='7'*64,
                             accounting_columns=['JobIDRaw', 'State%40', 'ExitCode', 'AllocTRES%160'],
                             accounting_row=[self.step, 'COMPLETED', '0:0', 'cpu=8,node=1,gres/dcu=2'])
        self.numeric = dict(self.binding, slurm_step=self.step, terminal_status='COMPLETED', exit_code=0,
                            cards=2, optimizer_update_counts={'zero_forward': 0, 'first_update': 1},
                            terminal_receipt_SHA256='8'*64)
        self.pin = {'path': 'numeric.json', 'SHA256': '9'*64}
        self.evidence = dict(hygon_execution=self.pin,
                             launch_evidence={'path': 'launch.json', 'SHA256': '7'*64},
                             terminal_evidence={'path': 'terminal.json', 'SHA256': '8'*64})
        self.objects = {
            ('numeric.json', '9'*64): self.numeric,
            ('launch.json', '7'*64): self.launch,
            ('terminal.json', '8'*64): self.terminal,
        }

    def verify(self):
        return verify_function(self.case, self.commit, self.entry, self.registry, self.evidence,
                               self.pin, lambda pin: self.objects[pin['path'], pin['SHA256']])

    def test_registry_launch_and_numerics_share_one_execution(self):
        self.assertEqual(self.verify(), {'passed': True, 'slurm_step': self.step, 'cards': 2})

    def test_old_registry_does_not_qualify_the_final_commit(self):
        self.registry['source']['framework_commit'] = 'a'*40
        with self.assertRaisesRegex(ValueError, 'exact framework commit'):
            self.verify()

    def test_rejects_different_entry_plan_or_source(self):
        for field, value in (('dry_run', True), ('run_requested', False), ('selected_case', 'T09.sft'),
                             ('framework_tree', 'a'*40), ('entry_SHA256', 'a'*64),
                             ('entry_path', 'tools/different_entry.py')):
            with self.subTest(field=field):
                old = self.launch[field]
                self.launch[field] = value
                with self.assertRaisesRegex(ValueError, 'verified case entry'):
                    self.verify()
                self.launch[field] = old

    def test_rejects_registry_only_and_separate_numeric_receipt(self):
        self.evidence['hygon_execution'] = {'path': 'different.json', 'SHA256': '9'*64}
        with self.assertRaisesRegex(ValueError, 'same Hygon execution'):
            self.verify()
        self.evidence['hygon_execution'] = self.pin
        self.numeric['optimizer_update_counts']['first_update'] = 0
        with self.assertRaisesRegex(ValueError, 'first optimizer update'):
            self.verify()

    def test_rejects_nonterminal_different_step_or_extra_cards(self):
        for column, value, message in ((0, '1087656.999', 'successful Slurm step'),
                                       (1, 'RUNNING', 'successful Slurm step'),
                                       (3, 'cpu=8,node=1,gres/dcu=8', 'exact DCU count')):
            with self.subTest(column=column, value=value):
                old = self.terminal['accounting_row'][column]
                self.terminal['accounting_row'][column] = value
                with self.assertRaisesRegex(ValueError, message):
                    self.verify()
                self.terminal['accounting_row'][column] = old

    def test_terminal_must_bind_launcher_in_addition_to_numeric_run(self):
        self.terminal['launch_receipt_SHA256'] = 'a'*64
        with self.assertRaisesRegex(ValueError, 'bind the launcher'):
            self.verify()


if __name__ == '__main__':
    unittest.main()
