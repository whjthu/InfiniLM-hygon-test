"""Receive small authenticated wire pieces and commit the original 32 MiB ranges."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import signal
import struct
import sys
from tempfile import NamedTemporaryFile

LEGACY_LABEL = "training-zimage-publisher-runtime-transfer-v1"
PINNED = {
    "receiver.py": "447e3019f69e3b96169062498b1e974c252e33454ea57871d9b3ea809aaa92ef",
    "relay.py": "7cee03646d4a6512f1fdcc205ddc058231b4f430457aa2686421a94e1bb690f6",
    "stage_internvl_weight_segment.py": "dd578f3d92f287225347e0cbcfa181e6d6d3788d48d734a6d9a5be1725700d98",
}
PIECE = 4 << 20
HEADER = struct.Struct("<QI32s")


def load_legacy(root):
    for name, expected in PINNED.items():
        source = root / name
        if source.resolve(strict=True) != source or sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen runtime transfer provider differs")
    sys.path.insert(0, str(root))
    import receiver
    import relay
    import stage_internvl_weight_segment as stage
    for mod in (receiver, relay, stage):
        if Path(mod.__file__).resolve().parent != root:
            raise ValueError("Frozen runtime transfer imported from another location")
    return receiver, relay, stage


class Pieces:
    def __init__(self, root, receiver, stage):
        self.root = root
        self.receiver = receiver
        self.stage = stage

    def segment(self, row, file_offset):
        length = self.stage.segment_length(self.receiver.storage_spec(row), file_offset)
        directory = self.root / "wire-pieces" / str(row["asset_id"]) / str(file_offset)
        return length, directory

    def packed_piece(self, file, offset, count):
        self.stage.need(file.stat().st_size == HEADER.size + count, "Unexpected staged file size")
        raw = file.read_bytes()
        self.stage.need(len(raw) == HEADER.size + count, "Unexpected stored wire piece length")
        start, size, digest = HEADER.unpack_from(raw)
        body = raw[HEADER.size:]
        self.stage.need(start == offset and size == count and sha256(body).digest() == digest,
                        "Stored wire piece identity differs")
        return body

    def collect(self, directory, file_offset, length):
        bodies = []
        for delta in range(0, length, PIECE):
            piece = directory / f"{delta:08d}.piece"
            if not piece.exists() and not piece.is_symlink():
                break
            self.stage.need(piece.is_file() and not piece.is_symlink(), "Non-regular staged wire piece")
            bodies.append(self.packed_piece(piece, file_offset + delta, min(PIECE, length - delta)))
        return bodies

    def state(self, row, original_offset=None):
        original = self.receiver.inspect(self.root, row)
        offset = original["confirmed_bytes"]
        if original_offset is not None:
            self.stage.need(original_offset == offset, "Original durable runtime offset changed")
        status = {"file_id": row["asset_id"], "confirmed_bytes": offset,
                  "whole_file_qualified_at_seal": original["whole_file_qualified_at_seal"], "wire_bytes": 0}
        if offset < row["bytes"]:
            length, directory = self.segment(row, offset)
            if directory.exists() or directory.is_symlink():
                with self.stage.locked(directory):
                    status["wire_bytes"] = sum(map(len, self.collect(directory, offset, length)))
            status["segment_bytes"] = length
        return status

    def append(self, row, file_offset, wire_offset, expected_sha, stream):
        self.state(row, file_offset)
        length, directory = self.segment(row, file_offset)
        self.receiver.provision_directory(directory, parents=True)
        with self.stage.locked(directory):
            self.stage.need(self.receiver.inspect(self.root, row)["confirmed_bytes"] == file_offset,
                            "Original range advanced before append")
            actual_wire_offset = sum(map(len, self.collect(directory, file_offset, length)))
            self.stage.need(wire_offset == actual_wire_offset < length and wire_offset % PIECE == 0,
                            "Wire offset differs or was already committed")
            count = min(PIECE, length - wire_offset)
            body = stream.read(count + 1)
            self.stage.need(len(body) == count and sha256(body).hexdigest() == expected_sha,
                            "Wire piece size or digest differs")
            piece = directory / f"{wire_offset:08d}.piece"
            with NamedTemporaryFile(dir=directory, prefix="wire-", delete=False) as out:
                pending = Path(out.name)
                try:
                    out.write(HEADER.pack(file_offset + wire_offset, count, bytes.fromhex(expected_sha)))
                    out.write(body)
                    out.flush()
                    os.fsync(out.fileno())
                    out.close()
                    os.link(pending, piece)
                    self.stage.sync_dir(directory)
                finally:
                    out.close()
                    pending.unlink(missing_ok=True)
            return {"file_id": row["asset_id"], "confirmed_bytes": file_offset,
                    "wire_bytes": wire_offset + count, "piece_bytes": count, "piece_SHA256": expected_sha}

    def commit(self, row, file_offset):
        self.state(row, file_offset)
        length, directory = self.segment(row, file_offset)
        with self.stage.locked(directory):
            self.stage.need(self.receiver.inspect(self.root, row)["confirmed_bytes"] == file_offset,
                            "Original range advanced before commit")
            bodies = self.collect(directory, file_offset, length)
            self.stage.need(sum(map(len, bodies)) == length, "Original range is incomplete")
            body = b"".join(bodies)
            result = self.receiver.receive(self.root, row, file_offset, sha256(body).hexdigest(), BytesIO(body))
            # These files are temporary duplicates of the fsynced original range.
            for delta in range(0, length, PIECE):
                (directory / f"{delta:08d}.piece").unlink()
            self.stage.sync_dir(directory)
            return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("inspect", "append", "commit"))
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--asset", type=int, required=True)
    parser.add_argument("--file-offset", type=int)
    parser.add_argument("--wire-offset", type=int)
    parser.add_argument("--digest")
    parser.add_argument("--budget", type=int, required=True)
    args = parser.parse_args()
    receiver, _, stage = load_legacy(args.legacy)
    stage.need(1 <= args.budget <= 180, "Wire receiver deadline differs")

    def expired(*_):
        raise TimeoutError("Bounded wire receiver deadline")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, args.budget)
    matches = [row for row in receiver.assets() if row["asset_id"] == args.asset]
    stage.need(len(matches) == 1, "Unknown published asset")
    row = matches[0]
    pieces = Pieces(receiver.TARGET, receiver, stage)
    if args.operation == "inspect":
        result = pieces.state(row)
    elif args.operation == "append":
        result = pieces.append(row, args.file_offset, args.wire_offset, args.digest, sys.stdin.buffer)
    else:
        result = pieces.commit(row, args.file_offset)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
