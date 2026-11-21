import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import preflight_vggt_tum_dev24_backfill as probe


class Dev24BackfillPreflightTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.repo = Path(directory.name).resolve(strict=True)
        self.results = self.repo / 'private-results'
        self.stage = self.results / probe.STAGE_NAME
        self.accounting = f'{probe.PARENT}|fixture-parent|RUNNING|0:0|600|{probe.NODE}|gres/dcu=320\n'.encode()

    def put(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = value if isinstance(value, bytes) else json.dumps(value).encode()
        path.write_bytes(raw)
        return probe.digest(raw)

    def staged(self):
        manifest = {path: self.put(self.repo / path, b'fixture source ' + path.encode()) for path in probe.SOURCE_FILES}
        manifest_sha = self.put(self.repo / probe.MANIFEST_PATH, manifest)
        input_sha = self.put(self.stage / 'inputs' / probe.MANIFEST, b'fixture original inputs')
        heads = {arm: dict(gpu_step=gpu, request=request, clean_post=clean, training_receipt_SHA256=prior,
                          global_step=25, actual_checkpoint_bytes_read=0, file_SHA256='a' * 64,
                          module_SHA256='b' * 64) for arm, (gpu, request, clean, prior) in probe.ARMS.items()}
        authority_sha = self.put(self.stage / 'head-authority.json', heads)
        original_sha = self.put(self.stage / 'original-training-accounting.psv', b'fixture original training rows')
        self.stage_sha = self.put(self.stage / 'stage-complete.json', dict(
            state='INPUTS_STAGED_ORIGINAL_TRAINING_RECEIPTS_MATCHED', new_slurm_steps=0, optimizer_updates=0,
            actual_checkpoint_bytes_read=0, GPU_device_admitted=False, scorer_permission=False,
            input_staging=dict(inputs_SHA256=input_sha, bundle_SHA256=probe.BUNDLE_SHA, input_pairs=22,
                               planned_windows=24, verified_members=23), accounting_SHA256=original_sha,
            head_authority_SHA256=authority_sha))
        patches = patch.multiple(probe, MANIFEST_SHA=manifest_sha, INPUTS_SHA=input_sha)
        patches.start()
        self.addCleanup(patches.stop)

    def run_probe(self, *, accounting=None, node='NO_NUMBERED_STEP'):
        replies = iter([self.accounting] * 2 if accounting is None else accounting)
        return probe.preflight(self.repo, self.results, lambda: next(replies), lambda: node)

    def test_missing_and_partial_stage_remain_distinct(self):
        self.assertEqual(self.run_probe()['state'], 'TARGET_STAGING_REQUIRED')
        self.stage.mkdir(parents=True)
        self.assertEqual(self.run_probe()['state'], 'RECONCILE_ORIGINAL_STAGING')

    def test_valid_stage_is_bound_but_does_not_inflate_training_or_admission(self):
        self.staged()
        result = self.run_probe()
        self.assertEqual(result['state'], 'ORIGINAL_STAGING_VERIFIED_DEVICE_AND_HOLDS_PENDING')
        self.assertEqual(result['stage_SHA256'], self.stage_sha)
        self.assertEqual(len(result['original_observations']), 2)
        self.assertEqual(result['original_heads'], 2)
        self.assertEqual((result['requested_DCU'], result['planned_windows'], result['optimizer_updates']), (1, 24, 0))
        self.assertFalse(result['counts_toward_training_reserve'])
        self.assertFalse(result['device_admitted'])
        self.assertFalse(result['approved_for_dispatch'])
        self.assertEqual(result['new_slurm_steps'], 0)

    def test_late_exact_scoring_step_forces_original_reconciliation(self):
        self.staged()
        late = f'{probe.PARENT}.999|{probe.LABEL}-gpu|RUNNING|0:0|4|{probe.NODE}|gres/dcu=1\n'.encode()
        result = self.run_probe(accounting=[self.accounting, self.accounting + late])
        self.assertEqual(result['state'], 'RECONCILE_ORIGINAL_SCORING')
        self.assertTrue(result['original_observations'][1]['rows'])
        self.put(self.stage / 'score-intent.json', b'transport outcome unknown')
        result = self.run_probe()
        self.assertEqual(result['state'], 'RECONCILE_ORIGINAL_SCORING')
        self.assertEqual(result['original_observations'][0]['markers'], ['score-intent.json'])

    def test_no_parent_accounting_cannot_establish_original_absence(self):
        with self.assertRaisesRegex(ValueError, 'parent missing'):
            self.run_probe(accounting=[b'', b''])

    def test_clean_assets_with_active_numbered_step_still_blocked(self):
        self.staged()
        result = self.run_probe(node='CPU_STEP_ONLY')
        self.assertEqual(result['state'], 'WAITING_FOR_NUMBERED_STEP_CLEARANCE')

    def test_staged_inputs_and_controller_bytes_are_bound(self):
        self.staged()
        path = self.stage / 'inputs' / probe.MANIFEST
        path.write_bytes(b'other stage')
        with self.assertRaisesRegex(ValueError, 'input digest differs'):
            self.run_probe()

    def test_current_frozen_controller_covers_exact_members(self):
        probe.verify_controller_source(probe.REPO)


if __name__ == '__main__':
    unittest.main()
