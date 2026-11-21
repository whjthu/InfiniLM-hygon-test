from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from training_sample_artifacts import verifier


class ArtifactTest(TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        root = Path(folder.name)
        self.repo = root / "repo"
        self.private = root / "lab"
        self.repo.mkdir()
        self.private.mkdir()

    def record(self, location, raw, sample="q", status="scored"):
        return dict(sample_id=sample, status=status,
                    artifact=dict(location=str(location), SHA256=sha256(raw).hexdigest()))

    def text_predictions(self):
        lines = [b'{"question_id":2,"answer":"\xe6\x98\xaf"}\r\n',
                 b'{"question_id":3,"status":"failed","reason":"timeout"}']
        path = self.private / "text.jsonl"
        path.write_bytes(b"".join(lines))
        samples = [self.record(f"{path}#line={i}", raw, str(i+1), "scored" if i == 1 else "failed")
                   for i, raw in enumerate(lines, 1)]
        return path, samples, sha256(b"".join(lines)).hexdigest()

    def test_text_lines_match_private_root_and_original_bytes(self):
        _, samples, digest = self.text_predictions()
        with self.assertRaisesRegex(ValueError, "outside the declared roots"):
            verifier(self.repo)(samples, "TextVQA", digest)
        verifier(self.repo, (self.private,))(list(reversed(samples)), "TextVQA", digest)

    def test_wrong_sample_or_status_is_rejected_even_with_matching_hash(self):
        _, samples, digest = self.text_predictions()
        check = verifier(self.repo, (self.private,))
        for change in (dict(sample_id="3"), dict(status="failed")):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "different sample or status"):
                check([dict(samples[0], **change), samples[1]], "TextVQA", digest)

    def test_rejects_extra_unplanned_record_with_repinned_file(self):
        path, samples, _ = self.text_predictions()
        raw = path.read_bytes() + b'\n{"question_id":9,"answer":"yes"}'
        path.write_bytes(raw)
        # Keep the two planned line hashes current. The extra record is still unaccounted for.
        records = raw.splitlines(keepends=True)
        samples[1]["artifact"]["SHA256"] = sha256(records[1]).hexdigest()
        with self.assertRaisesRegex(ValueError, "rows outside the planned"):
            verifier(self.repo, (self.private,))(samples, "TextVQA", sha256(raw).hexdigest())

    def test_co3d_keeps_header_and_checks_body(self):
        path = self.repo / "pose.jsonl"
        header = b'{"checkpoint":"example"}\n'
        row = b'{"sample_id":"pair","status":"scored","error":12}'
        path.write_bytes(header + row)
        sample = self.record("pose.jsonl#line=2", row, "pair")
        check = verifier(self.repo)
        digest = sha256(header + row).hexdigest()
        check([sample], "AUC@30", digest)
        path.write_bytes(header.replace(b"example", b"spoofed") + row)
        with self.assertRaisesRegex(ValueError, "prediction file hash differs"):
            check([sample], "AUC@30", digest)

    def test_whole_output_and_opaque_failure_require_original_content(self):
        path = self.repo / "frame"
        raw = b"\x00" * (2**20 + 3)
        path.write_bytes(raw)
        failure = b"device error"
        (self.repo / "fail").write_bytes(failure)
        samples = [self.record("frame", raw), self.record("fail", failure, "f", "failed")]
        check = verifier(self.repo)
        check(samples, "LPIPS")
        path.write_bytes(raw + b"\x01")
        with self.assertRaisesRegex(ValueError, "artifact body hash differs"):
            check(samples, "LPIPS")

    def test_missing_line_and_reused_file_are_rejected(self):
        path = self.repo / "outputs"
        raw = b"bytes"
        path.write_bytes(raw)
        check = verifier(self.repo)
        with self.assertRaisesRegex(ValueError, "line is missing"):
            check([self.record("outputs#line=2", raw)], "FID")
        with self.assertRaisesRegex(ValueError, "locator reused"):
            check([self.record(path, raw), self.record(path, raw, "second")], "FID")
