from pathlib import Path, PurePosixPath
import tempfile
import unittest

from install_gemma_domain_framework import (
    BASE_MANIFEST, base_files, candidate_files, digest, exact_source, git_object, write_tree,
)
from prepare_gemma_domain_pair import candidate


def pin(data, mode=0o644):
    return dict(bytes=len(data), SHA256=digest(data), git_blob=git_object("blob", data), mode=mode)


class GemmaFrameworkInstallation(unittest.TestCase):
    def test_real_base_and_overlay_derive_one_complete_tree_identity(self):
        repo = Path(__file__).resolve().parents[1]
        base = base_files((repo / BASE_MANIFEST[0]).read_bytes())
        target = PurePosixPath("/refactor/local-install-fixture")
        overlay, _ = candidate(repo, target, target / "models", target / "plugins", 29920)
        new, oid = candidate_files(base, overlay)
        self.assertEqual(len(base), 393)
        self.assertEqual(len(new), 396)
        self.assertEqual(set(new) - set(base), set(overlay) - set(base))
        self.assertEqual(len(oid), 40)
        for key in base.keys() - overlay.keys():
            self.assertEqual(new[key], base[key])

    def test_copies_real_files_with_binary_identity_and_executable_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source, dest = root / "source", root / "destination"
            source.mkdir()
            original, replacement, extra = b"original\n", b"replacement\n", b"new\xff\0"
            (source / "tool.py").write_bytes(original)
            (source / "unchanged.py").write_bytes(original)
            base = {"tool.py": pin(original, 0o755), "unchanged.py": pin(original)}
            overlay = {"tool.py": replacement, "runtime/new.py": extra}
            new, _ = candidate_files(base, overlay)
            write_tree(source, dest, base, overlay, new)
            self.assertEqual(exact_source(dest, "tool.py", new["tool.py"]), replacement)
            self.assertEqual(exact_source(dest, "runtime/new.py", new["runtime/new.py"]), extra)
            self.assertEqual(exact_source(dest, "unchanged.py", new["unchanged.py"]), original)
            self.assertEqual(new["tool.py"]["mode"], 0o755)
            self.assertEqual((source / "tool.py").read_bytes(), original)

    def test_rejects_changed_base_even_if_overlay_would_replace_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = root / "source"
            source.mkdir()
            (source / "tool.py").write_bytes(b"tampered")
            base = {"tool.py": pin(b"original")}
            overlay = {"tool.py": b"replacement"}
            new, _ = candidate_files(base, overlay)
            dest = root / "destination"
            with self.assertRaisesRegex(ValueError, "bytes differ"):
                write_tree(source, dest, base, overlay, new)
            self.assertFalse((dest / "tool.py").exists())


if __name__ == "__main__":
    unittest.main()
