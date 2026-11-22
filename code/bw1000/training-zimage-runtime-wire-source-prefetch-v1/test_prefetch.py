from hashlib import sha256
import threading
import time
import unittest

from prefetch import advance, PIECE_BYTES


class PrefetchTest(unittest.TestCase):
    def setUp(self):
        self.length = 2 * PIECE_BYTES + 1
        self.writes = []
        self.operations = []

    def transport(self, argv, data, _):
        op = argv[0]
        self.operations.append(op)
        if op == "inspect":
            return dict(file_id=5, confirmed_bytes=0, segment_bytes=self.length, wire_bytes=0)
        if op == "append":
            offset = int(argv[argv.index("--wire-offset") + 1])
            self.assertEqual(data[0], offset // PIECE_BYTES)
            self.writes.append(offset)
            return dict(file_id=5, confirmed_bytes=0, wire_bytes=offset+len(data),
                        piece_SHA256=sha256(data).hexdigest())
        return dict(file_id=5, confirmed_bytes=self.length)

    def run_window(self, fetch, limit=8):
        return advance(dict(asset_id=5, bytes=self.length), self.transport, None, limit,
                       time.monotonic()+20, lambda *_: None, fetch=fetch)

    def test_overlapping_reads_preserve_order_and_short_tail(self):
        barrier = threading.Barrier(3)

        def read(_, offset, count, remaining, publisher):
            barrier.wait(timeout=3)
            return bytes([offset // PIECE_BYTES])*count

        result = self.run_window(read)
        self.assertEqual(self.writes, [0, PIECE_BYTES, 2*PIECE_BYTES])
        self.assertEqual(result["confirmed_bytes"], self.length)

    def test_unreadable_middle_piece_leaves_preceding_ack_without_commit(self):
        def read(_, offset, count, remaining, publisher):
            if offset == PIECE_BYTES:
                raise OSError("source timeout")
            return bytes([offset // PIECE_BYTES])*count

        with self.assertRaises(OSError):
            self.run_window(read)
        self.assertEqual(self.writes, [0])
        self.assertNotIn("commit", self.operations)

    def test_window_never_reads_an_unplanned_range(self):
        seen = []

        def read(_, offset, count, remaining, publisher):
            seen.append(offset)
            return bytes(count)

        self.run_window(read, limit=1)
        self.assertEqual(seen, [0])
        self.assertNotIn("commit", self.operations)


if __name__ == "__main__":
    unittest.main()
