from copy import deepcopy
import json
import posixpath
import unittest

import prepare_glm474_fresh_kv_native_build as builder


class FreshKVNativeBuildTest(unittest.TestCase):
    def setUp(self):
        self.sources, self.pins = builder.caller_sources()
        plan = json.loads((builder.CALLER.parent/'depth-stratified-observer-CPU-v1/PLAN.json').read_bytes())
        binding = json.loads((builder.CALLER.parent/'depth-stratified-observer-CPU-v1/BASELINE_BINDING.json').read_bytes())
        library = str(builder.ORIGIN/'B/libGLM_depth_observer_B.so')
        build = dict(passed=True, step='1103606.308', library=dict(path=library, SHA256=builder.ORIGIN_LIBRARY_SHA),
                     actual=dict(objects=32), objects={}, fresh_M={}, commands=[])
        self.plan = plan
        self.kernel = dict(path=str(builder.KERNEL/'fresh_kv_pack.o'), SHA256='b'*64, slurm_step='1103606.700')
        remap = {}
        for unit in plan['units']:
            name = unit['name']
            output = (str(builder.ORIGIN/'B/objects'/(name+'.o')) if name in plan['affected_units']
                      else binding['units'][name]['original_object'])
            build['objects'][name] = dict(path=output, SHA256='b'*64, bytes=32)
            build['fresh_M'][name] = dict(pins_file=str(builder.ORIGIN/'B'/(name+'.dependency-pins.json')), pins_SHA256='f'*64)
            remap[unit['baseline169_object']] = output
            if name in builder.CHANGED:
                source = str(builder.ORIGIN/'B/project-mirror') + posixpath.normpath(unit['source'])[len('/refactor'):]
                build['commands'].append(dict(label='B-'+name+'-compile', rc=0, owned_child_reaped=True,
                    argv=['/usr/bin/g++', *plan['new_common_flags']['HOST'], '-MD', '-MF', '/owned.d', '-MT', output,
                          '-c', source, '-o', output]))
        argv = [remap.get(arg, arg) for arg in plan['original169']['link_argv']]
        argv[argv.index('-o')+1] = library
        build['commands'].append(dict(label='B-link', rc=0, owned_child_reaped=True, argv=argv))
        self.build = build

    def test_published_caller_and_actual_ordered_link_recipe_shape(self):
        result = builder.make_recipe(self.build, self.pins, self.kernel)
        self.assertEqual(result['planned'], dict(fresh_M=2, HOST_compile=2, GPU_compile=0,
                                                 reused_objects=30, link_objects=33, links=1))
        objects = [arg for arg in result['link'] if arg.endswith('.o')]
        expected = [str(builder.NATIVE/(name+'.o')) if name in builder.CHANGED else self.build['objects'][name]['path']
                    for name in builder.OBJECT_NAMES]
        self.assertEqual(objects, [*expected, self.kernel['path']])
        self.assertLess(result['link'].index(self.kernel['path']), result['link'].index('-lcudart'))
        self.assertEqual(set(result['host']), set(builder.CHANGED))
        for name, unit in result['host'].items():
            self.assertEqual(unit['preprocess'][0], '/usr/bin/g++')
            self.assertIn('-iquote', unit['compile'])
            self.assertEqual(unit['compile'][-1], str(builder.NATIVE/(name+'.o')))
        self.assertTrue(result['linked_library'].endswith('libGLM_fresh_kv_pack.so'))

    def test_object_order_and_unqualified_origin_rejected(self):
        for field, mutation in (
                ('link', lambda build: build['commands'][-1]['argv'].append('/bad/other.o')),
                ('receipt', lambda build: build['commands'][0].update(owned_child_reaped=False)),
                ('object', lambda build: build['objects'].pop('u02'))):
            build = deepcopy(self.build)
            mutation(build)
            with self.subTest(field=field), self.assertRaises(ValueError):
                builder.make_recipe(build, self.pins, self.kernel)

    def test_kernel_requires_actual_original_terminal_and_zero_gres_scope(self):
        row = [self.kernel['slurm_step'], builder.KERNEL_LABEL, 'COMPLETED', '0:0', '11', 'f12r2n13',
               'cpu=4,mem=8G,node=1']
        worker = dict(passed=True, completed=True, original_step=row[0], manifest_SHA256=builder.KERNEL_MANIFEST,
                      exports=builder.EXPORTS, object=self.kernel['path'], object_SHA256=self.kernel['SHA256'],
                      actual_device_TUs=1, actual_GPU_steps=0)
        proof = dict(request=builder.KERNEL_REQUEST, state='COMPILE_VERIFIED', independent_observation=True,
                     actual_GPU_steps=0, row=row, object_SHA256=self.kernel['SHA256'])
        self.assertEqual(builder.compiled_kernel(proof, worker), self.kernel)
        for index, replacement in ((2, 'RUNNING'), (6, 'cpu=4,mem=8G,node=1,gres/dcu=1'), (5, 'other-node')):
            other = deepcopy(proof)
            other['row'][index] = replacement
            with self.subTest(index=index), self.assertRaises(ValueError):
                builder.compiled_kernel(other, worker)

    def test_dependencies_preserve_exact_old_headers_and_added_generated_includes(self):
        recipe = builder.make_recipe(self.build, self.pins, self.kernel)
        unit = recipe['host']['u17']
        inherited = '/old/header.hpp'
        prior = {unit['original_source']: '1', inherited: '2'}
        system = {'/usr/include/array': '3'}
        actual = {unit['generated_source']: unit['generated_source_SHA256'], inherited: '2', **system}
        actual.update({str(builder.NATIVE/'src'/name): digest for name,digest in unit['required_generated_includes'].items()})
        builder.check_host_graph(prior, actual, unit, system)
        with self.assertRaisesRegex(ValueError, 'dependency differs'):
            builder.check_host_graph(prior, dict(actual, **{inherited: 'drift'}), unit, system)
        with self.assertRaisesRegex(ValueError, 'unregistered'):
            builder.check_host_graph(prior, dict(actual, **{'/unregistered.hpp': 'added'}), unit, system)
        deps = builder.dependencies(b'/candidate.o: /one.h \\\n /two.h\n', '/candidate.o')
        self.assertEqual(deps, {'/one.h','/two.h'})

    def test_link_qualification_preserves_all_existing_exports_and_providers(self):
        reference = {f'old_{n}': 'T' for n in range(281)}
        current = {**reference, **dict.fromkeys(builder.EXPORTS, 'T')}
        platforms = {f'/opt/lib{n}.so': 'b'*64 for n in range(71)}
        libraries = ['libc.so', 'libcudart.so']
        builder.linked_scope(reference, current, libraries, libraries, platforms, platforms)
        with self.assertRaisesRegex(ValueError, 'strong export'):
            builder.linked_scope(reference, dict(current, extra='T'), libraries, libraries, platforms, platforms)
        with self.assertRaisesRegex(ValueError, 'NEEDED'):
            builder.linked_scope(reference, current, libraries, libraries[:1], platforms, platforms)
        with self.assertRaisesRegex(ValueError, 'provider identity'):
            builder.linked_scope(reference, current, libraries, libraries, platforms, dict(platforms, other='a'))
        with self.assertRaisesRegex(ValueError, 'is unresolved'):
            builder.provider_paths('libcudart.so => not found')


if __name__ == '__main__':
    unittest.main()
