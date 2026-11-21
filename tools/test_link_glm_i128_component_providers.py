from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import link_glm_i128_component_providers as link


class GLMProviderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.auditor = link.load_auditor()

    def fixtures(self):
        records = {}
        for index, arm in enumerate(link.ROLES, 1):
            plan, rows, _, _ = self.auditor.load(arm)
            records[arm] = dict(arm=arm, passed=True, outcome='TWO_TARGET_COMPILER_OBJECTS_VERIFIED_WITHOUT_GPU_EXECUTION',
                accounting=[f'1103606.{index}', plan['label'], 'COMPLETED', '0:0', '20', plan['node'], 'cpu=4,node=1,mem=8G'],
                original_request=str(index)*32, publication='a'*40, launch_SHA256='b'*64, build_SHA256='c'*64,
                device_numerical_qualified=False, verified_objects=[dict(rows=row,
                    object=f'/refactor/candidates/{plan["label"]}/rows{row}/routed_i128.o',
                    object_SHA256=sha256(f'{arm}:{row}'.encode()).hexdigest(), compiled_bytes=123,
                    candidate_id=plan['variants'][str(row)]['id'], source_SHA256=plan['variants'][str(row)]['source_SHA256'])
                    for row in rows])
        return records

    def test_published_claims_preserve_all_four_combinations(self):
        objects = link.adopted_objects(self.fixtures(), self.auditor)
        self.assertEqual(set(objects), {'ORIG', 'G1', 'G2', 'D2', 'D4'})
        self.assertEqual(objects['G2']['rows'], 2)
        self.assertEqual(objects['D4']['rows'], 4)
        self.assertNotEqual(objects['G1']['CPU_step'], objects['D2']['CPU_step'])

    def test_rejects_wrong_or_missing_original_compile_and_role(self):
        cases = [lambda a: a['gate'].update(passed=False), lambda a: a['down']['verified_objects'].pop(),
                 lambda a: a['down']['verified_objects'][0].update(object='/not-original'),
                 lambda a: a['down'].update(original_request=a['gate']['original_request'])]
        for change in cases:
            base = self.fixtures()
            change(base)
            with self.assertRaises(ValueError):
                link.adopted_objects(base, self.auditor)

    def test_link_command_isolated_one_object_and_passive_graph(self):
        command = link.shared_command('a.o', Path('G1.so'), Path('only-i128.map'))
        self.assertEqual(command[-2:], ['a.o', link.RUNTIME])
        self.assertIn('-Wl,-z,defs,-Bsymbolic', command)
        self.assertIn('-Wl,-soname,G1.so', command)
        libraries = {link.RUNTIME: 'f'*64}
        good = f'libcudart.so.12 => {link.RUNTIME} (0x00)\n'
        self.assertEqual(link.bound_runtime_graph(good, libraries), libraries)
        with self.assertRaises(ValueError):
            link.bound_runtime_graph(good + 'libcuda.so.1 => not found\n', libraries)
        with self.assertRaises(ValueError):
            link.bound_runtime_graph(good + 'libforeign.so => /usr/lib/foreign.so (0x01)\n', libraries)

    def test_binding_requires_real_terminal_and_pins_each_dso(self):
        accounting = f'1103606.55|{link.LABEL}|COMPLETED|0:0|11|{link.NODE}|cpu=4,node=1,mem=8G\n'.encode()
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name).resolve()
            roles = {}
            for role in ('ORIG', 'G1', 'G2', 'D2', 'D4'):
                path = folder / (role + '.so')
                path.write_bytes(role.encode())
                roles[role] = link.stream_pin(path)
            hip = folder / 'hip.so'
            hip.write_bytes(b'test hip metadata fixture')
            report = dict(state='ISOLATED_HOST_LINK_PASS', request_scope=link.LABEL, actual_CPU_step='1103606.55',
                          epoch=1000, deadline=1300, actual_target_GPU_steps=0, GPU_GO=False,
                          verified_roles=roles, hip=link.stream_pin(hip))
            with patch.object(link, 'OUTPUT', folder), patch.object(link, 'HIP', str(hip)):
                result = link.caller_binding(report, accounting, '0000:0a:00.0', 1301, 1481, self.auditor)
                self.assertEqual(result['schema'], 'glm_i128_one_gpu_v1')
                self.assertEqual(set(result['providers']), set(roles))
                with self.assertRaises(ValueError):
                    link.caller_binding(report, accounting.replace(b'COMPLETED', b'FAILED'),
                                        '0000:0a:00.0', 1301, 1481, self.auditor)
                (folder / 'G1.so').write_bytes(b'changed')
                with self.assertRaises(ValueError):
                    link.caller_binding(report, accounting, '0000:0a:00.0', 1301, 1481, self.auditor)

    def test_original_model_runtime_pins_are_present(self):
        path = link.PACKAGES / 'attention-detail-CPU-v1/BUILD_PLAN.json'
        providers = json.loads(link.read_pin(path, link.PROVIDER_PLAN_SHA))['providers']
        self.assertTrue(link.check_sha(providers[link.HIP]))
        self.assertTrue(link.check_sha(providers[link.RUNTIME]))


if __name__ == '__main__':
    unittest.main()
