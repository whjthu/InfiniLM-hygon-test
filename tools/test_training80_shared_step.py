"""Test scheduler/resource and device identity gates without a Slurm or HIP connection."""
import unittest

from training80_shared_step import guard, srun_args, srun_env


class SharedStepTest(unittest.TestCase):
    def setUp(self):
        self.parent, self.node = '1087656', 'f11r4n11'
        self.pcis = [f'0000:{bus:02x}:00.0' for bus in range(8)]
        self.request = '1'*32
        self.args = dict(parent=self.parent, node=self.node, request_id=self.request, cards=3,
                         node_PCIs=self.pcis, workload_scope='engineering', deadline_epoch=1090,
                         duration_limit_s=120, peer_step_PCIs={self.parent+'.1': self.pcis[0:2]})
        self.lease = self.row(2, 3, label='training-share-'+self.request)
        self.raw = [self.row(1, 2), self.lease]
        self.job = f'JobId={self.parent} JobState=RUNNING UserId=owner(1200)'
        self.visible = self.pcis[2:5]
        self.render, self.primary = set(self.visible), set()
        self.reads = 0

    def row(self, step, cards, *, label='existing', host=None, nodes=1):
        return (f'StepId={self.parent}.{step} Name={label} State=RUNNING NodeList={host or self.node}'
                f' Nodes={nodes} TRES=cpu=4,mem=8G,node={nodes},gres/dcu={cards}')

    def query(self, *args):
        if args[:3] == ('scontrol', 'show', 'job'):
            return [self.job]
        if args[:3] == ('scontrol', 'show', 'hostnames'):
            return args[-1].split(',')
        if args[:3] == ('scontrol', 'show', 'step'):
            self.reads += 1
            return self.raw
        raise AssertionError(args)

    def run_guard(self, run_query=None, **changes):
        args = dict(environ={'SLURM_JOB_ID': self.parent, 'SLURM_STEP_ID': '2'}, observer_uid=1200,
                    hostname=lambda: self.node+'.domain', run_query=run_query or self.query,
                    visible_devices=lambda: self.visible, drm_devices=lambda: (self.pcis, self.render, self.primary),
                    clock=lambda: 1000)
        return guard(self.args, **(args | changes))

    def test_accepts_three_isolated_cards_next_to_two_card_step(self):
        report = self.run_guard()
        self.assertEqual(report['admission_step'], self.parent+'.2')
        self.assertEqual(report['assigned_PCIs'], self.visible)
        self.assertEqual(self.reads, 2)

    def test_cpu_only_peer_can_share_scheduler_host_budget(self):
        self.raw[0] = self.row(1, 0).replace(',gres/dcu=0', '')
        self.args['peer_step_PCIs'][self.parent+'.1'] = []
        self.assertEqual(self.run_guard()['assigned_PCIs'], self.visible)

    def test_ignores_known_other_node_without_reserving_it(self):
        self.raw.append(self.row(9, 8, host='f11r4n12'))
        self.assertEqual(self.run_guard()['checked_peer_steps'], [self.parent+'.1'])

    def test_count_without_proven_local_pci_is_rejected(self):
        self.raw.append(self.row(9, 1))
        with self.assertRaisesRegex(ValueError, 'colocated step counts'):
            self.run_guard()

    def test_confirmed_peer_does_not_license_sharing_its_device(self):
        self.visible = self.pcis[1:4]
        self.render = set(self.visible)
        with self.assertRaisesRegex(ValueError, 'HIP/DRM'):
            self.run_guard()

    def test_visible_runtime_mask_alone_does_not_prove_isolation(self):
        for field in ('render', 'primary'):
            value = getattr(self, field)
            value.add(self.pcis[-1])
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'HIP/DRM'):
                self.run_guard()
            value.clear()
            if field == 'render':
                value.update(self.visible)

    def test_new_peer_during_physical_check_invalidates_admission(self):
        def query(*args):
            rows = self.query(*args)
            if args[:3] == ('scontrol', 'show', 'step') and self.reads == 2:
                return rows + [self.row(9, 1)]
            return rows

        with self.assertRaisesRegex(ValueError, 'changed during admission'):
            self.run_guard(query)

    def test_multi_node_peer_with_aggregate_count_is_rejected(self):
        self.raw[0] = self.row(1, 4, host=self.node+',f11r4n12', nodes=2)
        with self.assertRaisesRegex(ValueError, 'stable single-node'):
            self.run_guard()

    def test_rejects_wrong_parent_owner_or_an_expired_step(self):
        with self.assertRaisesRegex(ValueError, 'original budget'):
            self.run_guard(clock=lambda: 1091)
        with self.assertRaisesRegex(ValueError, 'current owner'):
            self.run_guard(observer_uid=3000)

    def test_srun_request_scopes_node_cards_cpu_memory_and_no_overlap(self):
        args = srun_args(self.args, ['/usr/bin/python3', 'qualified_worker.py'], cpus=6, memory_mib=16384, minutes=2)
        self.assertTrue({'--exact', '--immediate=5', '--gres=dcu:3', '--cpus-per-task=6', '--mem=16384M',
                         '--nodelist='+self.node} <= set(args))
        self.assertFalse(any(arg.startswith('--overlap') for arg in args))
        self.assertEqual(srun_env({'SLURM_OVERLAP': '1', 'HIP_VISIBLE_DEVICES': '0,1', 'PATH': '/bin'}), {'PATH': '/bin'})
        with self.assertRaisesRegex(ValueError, 'original budget'):
            srun_args(self.args, ['/usr/bin/python3'], cpus=6, memory_mib=16384, minutes=3)

    def test_refuses_formal_benchmark_and_whole_node_workload(self):
        for changes in (dict(cards=8), dict(workload_scope='formal')):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'engineering workload'):
                guard(self.args | changes)


if __name__ == '__main__':
    unittest.main()
