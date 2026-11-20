from dataclasses import replace
from pathlib import Path
import shutil
import struct
import tempfile
import unittest

from verify_deep_c4_direct_history_link import (CANDIDATE_NATIVE, PLAN, PLAN_SHA, REPO, Elf,
    expected_probe, link_argv, parse_candidate_receipt, passive_elf, read_pinned_json, verify_relinked)


def elf_fixture():
    """An ELF64 with dynamic metadata only, for an actual readelf parser round trip."""
    strings = b'\0libbefore.so\0audited\0libfixture.so\0'
    names = b'\0.dynstr\0.dynsym\0.dynamic\0.shstrtab\0'
    symbols = bytes(24) + struct.pack('<IBBHQQ', strings.index(b'audited'), 0x12, 0, 1, 0, 0)
    tags = b''.join(struct.pack('<qQ', tag, value) for tag, value in (
        (1, 1), (14, strings.index(b'libfixture.so')), (16, 0), (0, 0)))
    contents = (strings, symbols, tags, names)
    binary = bytearray(64 + 56)
    locations = []
    for contents_for_section in contents:
        binary.extend(bytes(-len(binary) % 8))
        locations.append(len(binary))
        binary.extend(contents_for_section)
    binary.extend(bytes(-len(binary) % 8))
    table = len(binary)
    binary.extend(bytes(64))
    for index, (name, kind, link, info, align, entsize) in enumerate((
        (b'.dynstr', 3, 0, 0, 1, 0), (b'.dynsym', 11, 1, 1, 8, 24),
        (b'.dynamic', 6, 1, 0, 8, 16), (b'.shstrtab', 3, 0, 0, 1, 0),
    )):
        binary.extend(struct.pack('<IIQQQQIIQQ', names.index(name), kind, 2, locations[index], locations[index],
                                  len(contents[index]), link, info, align, entsize))
    ident = b'\x7fELF\x02\x01\x01' + bytes(9)
    struct.pack_into('<16sHHIQQQIHHHHHH', binary, 0, ident, 3, 62, 1, 0, 64, table, 0, 64, 56, 1, 64, 5, 4)
    dynamic = locations[2]
    struct.pack_into('<IIQQQQQQ', binary, 64, 2, 4, dynamic, dynamic, dynamic, len(tags), len(tags), 8)
    return binary


class DirectHistoryLinkTest(unittest.TestCase):
    def setUp(self):
        self.plan = read_pinned_json(REPO / PLAN, PLAN_SHA)
        self.link, = self.plan['links']
        self.original = self.link['required_NEEDED']
        exports = frozenset((name, 'FUNC', 'GLOBAL', 'DEFAULT') for name in self.link['required_exports'])
        self.probe = Elf((self.original, 'libc.so.6'), None, True, (), exports, frozenset())
        self.native = Elf(('libc.so.6',), None, True, (),
                          frozenset({('producer', 'FUNC', 'GLOBAL', 'DEFAULT')}), frozenset())

    def verify_expected(self, candidate):
        return expected_probe(self.probe, self.native, candidate, self.link, self.probe.needed, self.native.needed)

    def test_absolute_candidate_path_and_reused_object_order(self):
        argv = link_argv(self.plan, self.link)
        self.assertNotIn(self.original, argv)
        self.assertEqual(argv[argv.index(CANDIDATE_NATIVE) - 1], self.plan['candidate'] + '/probe.o')
        wanted = self.verify_expected(self.native)
        self.assertEqual(wanted, (CANDIDATE_NATIVE, 'libc.so.6'))
        verify_relinked(replace(self.probe, needed=wanted), self.probe, wanted)

    def test_old_soname_or_extra_dependency_cannot_select_the_baseline(self):
        with self.assertRaisesRegex(ValueError, 'SONAME'):
            self.verify_expected(replace(self.native, soname=self.original))
        with self.assertRaisesRegex(ValueError, 'Native dependency'):
            self.verify_expected(replace(self.native, needed=(self.original, 'libc.so.6')))

    def test_relink_keeps_probe_dependency_order_symbols_and_loader_paths(self):
        wanted = self.verify_expected(self.native)
        for bad in (self.probe, replace(self.probe, needed=wanted[::-1]),
                    replace(self.probe, needed=wanted, search_paths=(('RUNPATH', '/extra'),)),
                    replace(self.probe, needed=wanted, exports=frozenset())):
            with self.assertRaisesRegex(ValueError, 'change only the pinned Native dependency'):
                verify_relinked(bad, self.probe, wanted)

    def test_candidate_receipt_requires_exact_scope_and_real_library_sha(self):
        receipt = dict(step='1103606.487', completed=True, passed=True, explicit_GPU_calls=0,
                       new_HOST=1, new_GPU_TU=0, model_weights_read=0, reused_objects=49,
                       linked_objects=50, fresh_M_before_compile=True, library=CANDIDATE_NATIVE,
                       library_SHA256='a' * 64)
        self.assertEqual(parse_candidate_receipt(receipt), 'a' * 64)
        for bad in (dict(receipt, passed=False), dict(receipt, new_GPU_TU=False),
                    dict(receipt, library=self.original), dict(receipt, library_SHA256='unknown')):
            with self.assertRaises(ValueError):
                parse_candidate_receipt(bad)

    @unittest.skipUnless(shutil.which('readelf'), 'GNU readelf required')
    def test_gnu_readelf_passive_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.so'
            path.write_bytes(elf_fixture())
            found = passive_elf(path, shutil.which('readelf'))
        self.assertEqual(found, Elf(('libbefore.so',), 'libfixture.so', True, (),
                                    frozenset({('audited', 'FUNC', 'GLOBAL', 'DEFAULT')}), frozenset()))


if __name__ == '__main__':
    unittest.main()
