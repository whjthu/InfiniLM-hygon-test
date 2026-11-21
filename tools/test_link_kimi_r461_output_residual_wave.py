from copy import deepcopy
import re
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

import link_kimi_r461_output_residual_wave as linker
import prepare_kimi_r461_output_residual_score_capture as capture_source


REPO = Path(__file__).resolve().parents[1]
RENDER = REPO/'code/bw1000/Kimi-vocab-parallel-embedding-r461/output-residual-wave-compile-CPU-v1/prepare.py'


class ResidualWaveLinkTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name).resolve()
        self.source = self.workspace/'generated'
        self.assertEqual(runpy.run_path(str(RENDER))['prepare'](self.source)['manifest_SHA256'], linker.MANIFEST)
        self.pins = linker.checked_sources(self.source)
        self.scores = self.workspace/'scores'
        self.output = self.workspace/'linked'
        self.output.mkdir()
        self.receipt = dict(completed=True, passed=True, manifest_SHA256=linker.MANIFEST,
                            all_fresh_M_before_compile=True, actual_device_TUs=2, actual_GPU_steps=0,
                            runtime_calls=0, original_step='1103606.700', node='f13r1n01',
                            epoch=1700000000.0, deadline=1700000300.0, arms={})
        for arm in linker.ARMS:
            work = self.scores/arm
            work.mkdir(parents=True)
            obj = work/'fused_scores.o'
            pre = work/'fused_scores.before.d'
            md = work/'fused_scores.d'
            for path in (obj, pre, md):
                path.write_text(arm+path.name, encoding='utf-8')
            self.receipt['arms'][arm] = dict(object=str(obj), object_SHA256=linker.digest(obj), exports=[linker.HIDDEN],
                                             fresh_M_SHA256=linker.digest(pre), actual_MD_SHA256=linker.digest(md))
        self.accounting = dict(columns=['JobIDRaw', 'State', 'ExitCode', 'ElapsedRaw', 'NodeList', 'AllocTRES'],
                               row=['1103606.700', 'COMPLETED', '0:0', '76', 'f13r1n01', 'cpu=4,node=1,mem=8G'])

    def qualified(self, receipt=None, accounting=None):
        return linker.qualified_scores(self.receipt if receipt is None else receipt,
                                       self.accounting if accounting is None else accounting, output=self.scores)

    def test_real_render_keeps_identical_host_bytes_and_invalidates_changed_source(self):
        for item in linker.HOST_FILES:
            self.assertEqual((self.source/'full-tree'/item).read_bytes(), (self.source/'wave32'/item).read_bytes())
        (self.source/'wave32/fused_scores.cu').write_bytes(b'drift')
        with self.assertRaisesRegex(ValueError, 'rendered source bytes differ'):
            linker.checked_sources(self.source)

    def test_compile_objects_require_actual_terminal_zero_dcu_and_both_dependency_receipts(self):
        self.assertEqual(set(self.qualified()), set(linker.ARMS))
        for index, value in ((0, '1103606.799'), (1, 'RUNNING'), (3, '301'), (4, 'other-node'),
                             (5, 'cpu=4,node=1,mem=8G,gres/dcu=1')):
            altered = deepcopy(self.accounting)
            altered['row'][index] = value
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.qualified(accounting=altered)
        (self.scores/'wave32/fused_scores.d').write_bytes(b'other evidence')
        with self.assertRaisesRegex(ValueError, 'qualified score object differs'):
            self.qualified()

    def test_compiler_must_match_the_original_bounded_producer(self):
        tool = self.workspace/'compiler'
        tool.write_bytes(b'original compiler bytes')
        cpu330 = dict(commands=[[str(tool), '-fPIC', '-MD', '-c']])
        cpu987 = dict(actual_CPU='910407.987', passed=True, completed=True,
                      compiler_current_container=dict(canonical=str(tool), SHA256=linker.digest(tool)))
        linker.compiler_pin(cpu330, cpu987)
        tool.write_bytes(b'different compiler bytes')
        with self.assertRaisesRegex(ValueError, 'HOST compiler differs'):
            linker.compiler_pin(cpu330, cpu987)

    def select_capture(self):
        self.source = self.workspace/'capture'
        self.assertEqual(capture_source.prepare(self.source)['manifest_SHA256'], linker.CAPTURE_MANIFEST)
        self.pins = linker.checked_sources(self.source, manifest=linker.CAPTURE_MANIFEST)

    def test_diagnostic_changes_only_the_common_host_and_copies_after_dispatch(self):
        original = self.pins
        self.select_capture()
        self.assertEqual({name for name, pin in original.items() if self.pins[name] != pin},
                         {f'{arm}/component.cpp' for arm in linker.ARMS})
        for arm in linker.ARMS:
            cpp = (self.source/arm/'component.cpp').read_text(encoding='utf-8')
            dispatch = cpp.index('const auto dispatch = Clock::now()')
            done = cpp.index('const auto done = Clock::now()', dispatch)
            copy = cpp.index('probe.score_output_host->copy_from(runtime, scores)', done)
            self.assertIn('probe.output = probe.apply', cpp[dispatch:done])
            self.assertNotIn('capture', cpp[dispatch:done])
            self.assertLess(cpp.index('invalid Kimi fused score output buffer'), cpp.index('cudaGetDeviceCount'))
            self.assertIn('const auto &scores = probe.fused_score_hold', cpp[done:copy])
            self.assertRegex(cpp[copy:], re.compile(r'copy_from[^\n]+\n\s*runtime.syncStream\(\);'))
        with self.assertRaisesRegex(ValueError, 'metadata identity differs'):
            linker.checked_sources(self.source)

    def fixture_build(self, *, unexpected_provider=False, missing_symbol=False, missing_dependency=False,
                      capture=False, capture_export=False):
        shared = self.workspace/'shared.hpp'
        shared.write_bytes(b'accepted original header')
        providers = {str(Path('/runtime')/name): f'{index:064x}' for index, name in enumerate(linker.REQUIRED_PROVIDERS, 1)}
        cuda = str(Path(linker.CUDA))
        known = {**providers, cuda: linker.CUDA_SHA}
        original = {f'/refactor/incoming/output-residual-stage-diagnostic-CPU-v1/{name}': 'c'*64
                    for name in linker.HOST_FILES | {'stage.hpp'}}
        original[str(shared)] = linker.digest(shared)
        build330 = dict(actual_CPU='1103606.330', completed=True, passed=True, actual_passive_providers=known,
                        commands=[['/usr/bin/g++', '-fPIC', '-MD', '-c'], ['/usr/bin/g++', '-shared', '-lstdc++']])
        build335 = dict(actual_CPU='1103606.335', completed=True, passed=True, dependency_SHA256=original,
                        passive_providers=sorted(providers))
        scores = self.qualified()
        expected = {str(shared), *(str(self.source/'full-tree'/member) for member in linker.HOST_FILES)}
        phases = []
        paths = {Path('/runtime')/name for name in linker.REQUIRED_PROVIDERS} | {Path(linker.CUDA), Path('/foreign/lib.so')}
        real_digest, real_resolve = linker.digest, Path.resolve

        def digest(path):
            return known[str(path)] if str(path) in known else real_digest(path)

        def resolve(path, *args, **kwargs):
            return path if path in paths else real_resolve(path, *args, **kwargs)

        def run(phase, argv, cap):
            phases.append(phase)
            if phase == 'HOST-compile-once':
                Path(argv[-1]).write_bytes(b'shared host object')
            if phase.endswith('-link'):
                arm = phase.removesuffix('-link')
                self.assertIn(str(scores[arm]), argv)
                self.assertNotIn(str(scores[next(a for a in linker.ARMS if a != arm)]), argv)
                self.assertLess(argv.index(str(scores[arm])), argv.index(linker.CUDA))
                self.assertIn('-Wl,--no-undefined', argv)
                Path(argv[-1]).write_bytes(arm.encode())
            if phase.endswith('-exports'):
                reported = (*linker.VISIBLE, linker.CAPTURE_EXPORT) if capture_export else linker.VISIBLE
                return '\n'.join('0000 T '+name for name in reported)
            if phase.endswith('-symtab'):
                return '' if missing_symbol else f'  3: 000 1 FUNC LOCAL DEFAULT 1 {linker.HIDDEN}'
            if phase.endswith('-passive'):
                labels = [f'{name} => /runtime/{name} (0x0000)' for name in linker.REQUIRED_PROVIDERS]
                if unexpected_provider:
                    labels.append('lib.so => /foreign/lib.so (0x0000)')
                return '\n'.join([*labels, f'cudart => {linker.CUDA} (0x0000)'])
            return ''

        with patch.object(linker, 'dep_paths', return_value=expected-{str(shared)} if missing_dependency else expected), \
                patch.object(Path, 'resolve', resolve), patch.object(linker, 'digest', digest):
            result = linker.component_build(self.source, self.output, self.pins, scores, build330, build335, run,
                                            capture=capture)
        return phases, result

    def test_one_host_compile_links_both_arms_separately(self):
        phases, result = self.fixture_build()
        self.assertEqual(phases.count('HOST-compile-once'), 1)
        self.assertEqual([phase for phase in phases if phase.endswith('-link')], ['full-tree-link', 'wave32-link'])
        self.assertNotEqual(result['arms']['full-tree']['SHA256'], result['arms']['wave32']['SHA256'])
        self.assertEqual(result['arms']['full-tree']['providers'], result['arms']['wave32']['providers'])

    def test_refuses_missing_host_dependency_hidden_definition_and_foreign_provider(self):
        for option, reason in (('missing_dependency', 'fresh HOST dependencies'),
                               ('missing_symbol', 'own score definition'),
                               ('unexpected_provider', 'provider identity differs')):
            with self.subTest(option=option), self.assertRaisesRegex(ValueError, reason):
                self.fixture_build(**{option: True})

    def test_linker_requires_export_to_match_the_selected_build(self):
        with self.assertRaisesRegex(ValueError, 'score capture export disagrees'):
            self.fixture_build(capture_export=True)
        self.select_capture()
        with self.assertRaisesRegex(ValueError, 'score capture export disagrees'):
            self.fixture_build(capture=True)
        _, result = self.fixture_build(capture=True, capture_export=True)
        self.assertTrue(all(linker.CAPTURE_EXPORT in arm['exports'] for arm in result['arms'].values()))


if __name__ == '__main__':
    unittest.main()
