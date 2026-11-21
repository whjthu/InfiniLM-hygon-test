"""Synthetic pre-seal links simulate inode views; no real bind/model qualification."""
from copy import deepcopy
from hashlib import sha256
import ast
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest

import stage_internvl_weight_segment as legacy
import verify_zimage_sealed_runtime_view as view


class SealedView(unittest.TestCase):
    def test_readonly_metadata_and_plan_preserve_all_sealed_inodes(self):
        repo=Path(__file__).resolve().parents[1]
        original=json.loads((repo/view.MANIFEST).read_bytes())
        fixture=deepcopy(original)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();stage=root/'stage';stage.mkdir();consumer=root/'preseal-fixture-view'
            for row in fixture['assets']:
                raw=('explicit synthetic fixture asset%02d\n'%row['asset_id']).encode()
                row.update(bytes=len(raw),SHA256=sha256(raw).hexdigest())
                path=stage/('asset-%02d'%row['asset_id']);spec=view.spec(row)
                legacy.initialize(path,spec)
                from io import BytesIO
                legacy.append(path,spec,0,sha256(raw).hexdigest(),BytesIO(raw))
                dest=consumer/row['target_relative_path'];dest.parent.mkdir(parents=True,exist_ok=True)
                # This is fixture setup before seal. Production helper never calls link().
                os.link(path/'payload.partial',dest)
                legacy.seal(path,spec)
            before={r['asset_id']:view.canonical_file(stage/('asset-%02d'%r['asset_id'])/r['filename'])
                    for r in fixture['assets']}
            observed=view.verify_runtime(fixture,stage)
            self.assertTrue(observed['all20_sealed_metadata_consistent'])
            self.assertEqual(observed['verified_sealed_assets'],20)
            self.assertFalse(observed['independent_payload_SHA_verified'])
            partial=view.verify_runtime(fixture,stage,[2])
            self.assertEqual((partial['planned_assets'],partial['verified_sealed_assets']),(20,1))
            with self.assertRaisesRegex(ValueError,'Full20'):
                view.file_bind_plan(fixture,partial,stage,root/'partial-must-not-exist')
            plan=view.file_bind_plan(fixture,observed,stage,root/'new-skeleton')
            self.assertFalse((root/'new-skeleton').exists())
            self.assertEqual(len(plan['placeholders']),20)
            self.assertEqual(plan['apptainer_bind_arguments'].count('--bind'),22)
            self.assertTrue(all(v.endswith(':ro') for v in plan['apptainer_bind_arguments'][1::2]))
            self.assertEqual(plan['source_link_or_copy_operations'],0)
            simulated=view.verify_runtime(fixture,stage,mounted_root=consumer,require_readonly=False)
            self.assertTrue(simulated['mounted_view_metadata_verified'])
            self.assertFalse(simulated['readonly_mount_verified'])
            # Real original backend function hashes this canonical regular-file
            # fixture tree. This is source-policy evidence, not an OS bind test.
            with tarfile.open(repo/'code/bw1000/training-framework-bd24ec3-v1/FRAMEWORK.tar.gz','r:gz') as archive:
                backend=archive.extractfile('framework/generation/src/gencuda/backends/zimage_training.py').read()
            self.assertEqual(sha256(backend).hexdigest(),'6526be513a7b55977ede0dd176757da1530fffedaefa0080639e4191b1f49d31')
            function=next(n for n in ast.parse(backend).body if isinstance(n,ast.FunctionDef) and n.name=='_pinned_tree_sha256')
            import hashlib
            ns={'Path':Path,'hashlib':hashlib}
            exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),
                         'original_tree_identity_function_AST_only','exec'),ns)
            self.assertEqual(len(ns['_pinned_tree_sha256'](consumer/'base','model')),64)
            after={r['asset_id']:view.canonical_file(stage/('asset-%02d'%r['asset_id'])/r['filename'])
                   for r in fixture['assets']}
            self.assertEqual(before,after)
            # A source timestamp change after sealing must never be forgiven.
            row=fixture['assets'][0];leaf=stage/'asset-01'/row['filename']
            current=leaf.stat();os.utime(leaf,ns=(current.st_atime_ns,current.st_mtime_ns+1000000000))
            rejected=view.verify_leaf(row,stage)
            self.assertFalse(rejected['verified'])
            self.assertEqual(rejected['state'],'SEAL_METADATA_REJECTED')


if __name__=='__main__':unittest.main()
