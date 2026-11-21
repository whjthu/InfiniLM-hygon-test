import json
from hashlib import sha256
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import sample_training80_owned_busy as sampler
from tuning_queue import TRAINING80_LEDGER


class ActiveSamplerTest(unittest.TestCase):
    def setUp(self):
        self.owners = sampler.node_map(json.loads(TRAINING80_LEDGER.read_bytes()))
        self.pins = {node:[f'0000:{i:02x}:00.0' for i in range(8)] for node in self.owners}
        self.plan = {'request':'a'*32, 'scheduled_epoch':100.0, 'bindings':{}}
        index = 0
        for i,(node,(parent,_)) in enumerate(self.owners.items()):
            parts = (self.pins[node][:4],self.pins[node][4:]) if i < 40 else (self.pins[node],)
            for cards in parts:
                index += 1
                self.plan['bindings'][f'{index:064x}'] = dict(
                    request='a'*32,scheduled_epoch=100.0,node=node,step=f'{parent}.{index}',
                    boot_id='01234567-89ab-cdef-0123-456789abcdef',pci=cards,owner_receipt_SHA256='b'*64,
                    mode='workload' if i < 40 else 'idle_gres_sample')

    @staticmethod
    def recorded_points(drm, cards, scheduled, value=81):
        return [dict(point=p,opened_epoch=scheduled+p*2,closed_epoch=scheduled+p*2+.25,
                     busy={pci:value for pci in cards}) for p in range(3)]

    def reports(self, value):
        return {digest:sampler.sample(binding,digest,binding['node'],binding['step'],binding['boot_id'],clock=lambda:95,
                     read=lambda *args:self.recorded_points(*args,value=value))
                for digest,binding in self.plan['bindings'].items()}

    def test_complete_window_combines_distinct_live_step_pci_and_idle_nodes(self):
        for value in (80,81):
            with self.subTest(value=value):
                result=sampler.aggregate(self.owners,self.pins,self.plan,self.reports(value))
                self.assertEqual(result['observed_DCU'],640)
                self.assertEqual(result['sampled_workload_DCU'],320)
                self.assertEqual(result['observed_samples'],1920)
                self.assertEqual(result['fleet_mean_busy_percent'],value)
                self.assertEqual(result['strictly_above_80_in_this_window'],value>80)

    def test_permission_failure_and_missing_report_preserve_the_fleet_denominator(self):
        reports=self.reports(99)
        first, second = list(reports)[:2]
        reports[first].update(error='PermissionError',rows=[])
        del reports[second]
        result=sampler.aggregate(self.owners,self.pins,self.plan,reports)
        self.assertEqual(result['observed_DCU'],632)
        self.assertEqual(result['expected_samples'],1920)
        self.assertEqual(len(result['missing_cards']),8)
        self.assertEqual(result['failed_sources'],{first:'PermissionError'})
        self.assertIsNone(result['fleet_mean_busy_percent'])
        self.assertIsNone(result['strictly_above_80_in_this_window'])

    def test_node_report_separates_observed_zero_active_and_missing_cards(self):
        reports = self.reports(99)
        first, second, _, fourth = list(reports)[:4]
        node = reports[first]['node']
        other = reports[fourth]['node']
        cards = self.pins[node]
        self.plan['bindings'][second]['mode'] = reports[second]['mode'] = 'idle_gres_sample'
        for row in reports[first]['rows']:
            row['busy'].update({cards[0]: 0, cards[1]: 0})
        reports[first]['rows'][2]['busy'][cards[1]] = 1
        for row in reports[second]['rows']:
            row['busy'][cards[5]] = 0
        del reports[fourth]
        result = sampler.aggregate(self.owners, self.pins, self.plan, reports)
        parent, group = self.owners[node]
        self.assertEqual(result['node_observations'][node], dict(
            parent=parent, group=group, expected_cards=8, zero_at_every_point=[cards[0], cards[5]],
            nonzero_at_some_point=[cards[i] for i in (1, 2, 3, 4, 6, 7)], missing=[],
            sampled_inside_workload=cards[:4], sampled_inside_idle_probe=cards[4:],
        ))
        self.assertEqual(result['node_observations'][other]['missing'], self.pins[other][4:])
        self.assertEqual(result['node_observations'][other]['zero_at_every_point'], [])
        self.assertIsNone(result['fleet_mean_busy_percent'])

    def test_conflicting_assignments_and_clock_mismatch_reject_samples(self):
        reports=self.reports(99)
        first,second=list(reports)[:2]
        reports[first]['rows'][1]['closed_epoch']=104
        with self.assertRaisesRegex(ValueError,'common time window'):
            sampler.aggregate(self.owners,self.pins,self.plan,reports)
        self.plan['bindings'][second]['pci']=self.plan['bindings'][first]['pci']
        with self.assertRaisesRegex(ValueError,'overlapping PCI assignment'):
            sampler.aggregate(self.owners,self.pins,self.plan,{})

    def test_wrong_numbered_step_or_boot_cannot_read_sysfs(self):
        digest,binding=next(iter(self.plan['bindings'].items()))
        def unexpected(*_):
            raise AssertionError('device read must not begin')
        for step,boot in (('1103606.999',binding['boot_id']),(binding['step'],'other-boot')):
            with self.assertRaises(ValueError):
                sampler.sample(binding,digest,binding['node'],step,boot,read=unexpected)

    def test_read_points_reads_only_assigned_card_counters(self):
        card, extra = '0000:01:00.0','0000:02:00.0'
        paths = {}
        for pci in (card,extra):
            entry=MagicMock()
            device=entry.__truediv__.return_value
            device.resolve.return_value.name=pci
            device.__truediv__.return_value.read_text.return_value='87'
            paths[pci]=(entry,device.__truediv__.return_value.read_text)
        drm=MagicMock()
        drm.glob.return_value=[paths[card][0],paths[extra][0]]
        with patch('builtins.sorted',side_effect=lambda seq:list(seq)):
            clock=iter([100,100,100.1,102,102,102.1,104,104,104.1])
            points=sampler.read_points(drm,[card],100,clock=lambda:next(clock))
        self.assertEqual([row['busy'] for row in points],[{card:87}]*3)
        self.assertEqual(paths[card][1].call_count,3)
        paths[extra][1].assert_not_called()

    def test_aggregate_cli_pins_inventory_and_writes_a_complete_result(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            ledger, pins, plan = (folder / name for name in ('ledger.json','pins.json','plan.json'))
            output = folder/'summary.json'
            ledger.write_bytes(TRAINING80_LEDGER.read_bytes())
            pins.write_text(json.dumps(self.pins),encoding='utf-8')
            self.plan.update(ledger_SHA256=sha256(ledger.read_bytes()).hexdigest(),
                             pci_pins_SHA256=sha256(pins.read_bytes()).hexdigest())
            plan.write_text(json.dumps(self.plan),encoding='utf-8')
            for digest, report in self.reports(81).items():
                (folder/(digest+'.json')).write_text(json.dumps(report),encoding='utf-8')
            argv = ['sampler','aggregate',*(str(path) for path in (ledger,pins,plan,folder,output)),
                    sha256(plan.read_bytes()).hexdigest()]
            with patch('sys.argv',argv):
                self.assertEqual(sampler.main(),0)
            self.assertEqual(json.loads(output.read_bytes())['observed_samples'],1920)
            pins.write_text('{}',encoding='utf-8')
            with patch('sys.argv',argv), self.assertRaisesRegex(ValueError,'metadata SHA-256 differs'):
                sampler.main()


if __name__ == '__main__':
    unittest.main()
