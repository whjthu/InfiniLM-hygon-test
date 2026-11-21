import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import reconcile_qwen_usgs_dev80_original as reader


PARENT = b'1103606|original-parent|RUNNING|0:0|10|node-set|cpu=2560,node=40,gres/dcu=320'


class OriginalReconcileTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.base = Path(directory.name).resolve()
        self.coverage = dict.fromkeys(reader.NODES.values(),'NO_NUMBERED_STEP')
        self.label = reader.LABEL+'-base-gpu'

    def rows(self, state='RUNNING', handle='1103606.801', node='f13r1n10'):
        return PARENT+b'\n'+f'{handle}|{self.label}|{state}|0:0|4|{node}|cpu=4,gres/dcu=1'.encode()

    def reconcile(self, first=PARENT, last=PARENT):
        accounts=iter((first,last))
        return reader.reconcile(reader.REPO,self.base,account=lambda:next(accounts),
                                membership=lambda _repo:dict(observed_epoch=12,node_membership=self.coverage))

    def test_exact_absence_reuses_both_nodes_only_after_actual_observations(self):
        result=self.reconcile()
        self.assertEqual(result['state'],'ABSENCE_VERIFIED_AFTER_PRE_EXEC_AUTH_FAILURE')
        self.assertTrue(result['original_hold_eligible_for_removal'])
        self.assertFalse(result['original_replay_allowed'])
        self.assertFalse(result['current_pci_KFD_port_admission'])
        self.assertFalse(result['training_READY'])

    def test_late_or_disappearing_rows_never_release_an_original_hold(self):
        for first,last in ((PARENT,self.rows()),(self.rows(),PARENT)):
            result=self.reconcile(first,last)
            self.assertEqual(result['state'],'ORIGINAL_ACCOUNTING_NOT_TERMINAL')
            self.assertFalse(result['original_hold_eligible_for_removal'])

    def test_terminal_accounting_still_requires_independent_clean_post(self):
        result=self.reconcile(self.rows('COMPLETED'),self.rows('COMPLETED'))
        self.assertEqual(result['state'],'ORIGINAL_TERMINAL_ROWS_REQUIRE_INDEPENDENT_CLEANUP')
        self.assertFalse(result['original_hold_eligible_for_removal'])

    def test_namespace_without_accounting_remains_unknown(self):
        original=self.base/reader.LABEL
        original.mkdir()
        (original/'intent.json').write_text(json.dumps(dict(request=reader.REQUEST,publication=reader.PUBLICATION)),encoding='utf-8')
        result=self.reconcile()
        self.assertTrue(result['first']['namespace']['original_intent_observed'])
        self.assertEqual(result['state'],'ORIGINAL_NAMESPACE_PRESENT_ACCOUNTING_UNKNOWN')
        self.assertFalse(result['original_hold_eligible_for_removal'])

    def test_live_other_step_keeps_nodes_reserved(self):
        self.coverage['f13r1n10']='PARTIALLY_ASSIGNED'
        result=self.reconcile()
        self.assertTrue(result['original_not_submitted_verified'])
        self.assertFalse(result['original_hold_eligible_for_removal'])

    def test_foreign_parent_and_wrong_node_do_not_pass(self):
        with self.assertRaisesRegex(ValueError,'not uniquely covered'):
            self.reconcile(PARENT.replace(b'1103606',b'1087656'))
        with self.assertRaisesRegex(ValueError,'duplicate or foreign original Slurm phase'):
            self.reconcile(self.rows(node='other-host'))

    def test_membership_checks_actual_parent_owner_even_with_qualified_inventory(self):
        parents = {p:dict(state='RUNNING',ledger_exact=True,step_locations_exact=True,owned_by_observer=(p!='1103606'))
                   for p in ('1087656','1103606')}
        with patch.object(reader,'snapshot',return_value=dict(distinct_nodes=80,allocated_DCU=640,
                membership_qualified=True,parents=parents)), patch.object(reader.os,'getuid',return_value=1001,create=True):
            with self.assertRaisesRegex(ValueError,'allocation is not verified'):
                reader.live_membership(reader.REPO)


if __name__ == '__main__':
    unittest.main()
