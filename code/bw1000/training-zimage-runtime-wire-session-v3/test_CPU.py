import io
import json
from pathlib import Path
import subprocess
import sys
import unittest

from wire_session import Client, serve


class Pieces:
    def __init__(self):
        self.received = []

    def state(self, asset):
        return {"file_id": asset["asset_id"], "wire_bytes": len(self.received)}

    def append(self, asset, file_offset, wire_offset, digest, source):
        self.received.append(source.read())
        return {"file_id": asset["asset_id"], "wire_bytes": len(self.received)}


def frame(op, seq, **content):
    return json.dumps(dict(op=op, seq=seq, **content)).encode()+b"\n"


class FramingTest(unittest.TestCase):
    def test_payload_can_contain_newlines_and_second_frame_is_intact(self):
        store, output = Pieces(), io.BytesIO()
        append = frame("append", 1, file_offset=0, wire_offset=0, count=3, digest="a"*64)
        request = io.BytesIO(append+b"a\nb"+frame("inspect", 2))
        self.assertEqual(serve(request, output, store, {"asset_id": 7, "bytes": 100}, "nonce", "source"), 0)
        self.assertEqual(store.received, [b"a\nb"])
        lines = [json.loads(x) for x in output.getvalue().splitlines()]
        self.assertEqual([x["state"] for x in lines], ["SESSION_READY", "ACK", "ACK"])

    def test_truncated_frame_leaves_durable_writer_untouched(self):
        store, output = Pieces(), io.BytesIO()
        request = frame("append", 1, file_offset=0, wire_offset=0, count=7, digest="a"*64)+b"short"
        self.assertEqual(serve(io.BytesIO(request), output, store, {"asset_id": 7, "bytes": 100}, "nonce", "source"), 1)
        self.assertEqual(store.received, [])
        self.assertEqual(json.loads(output.getvalue().splitlines()[-1])["state"], "SESSION_STOPPED")

    def test_live_pipe_can_ack_two_operations_under_one_session(self):
        prog = ("import os,runpy,sys; sys.path.insert(0,os.path.dirname(sys.argv[1])); t=runpy.run_path(sys.argv[1]);"
                "t['serve'](sys.stdin.buffer,sys.stdout.buffer,t['Pieces'](),"
                "{'asset_id':7,'bytes':100},'nonce','source')")
        child = subprocess.Popen([sys.executable, "-B", "-c", prog, str(Path(__file__).resolve())],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        try:
            client = Client(child, 7, "nonce", "source", 5)
            inspect = ("inspect", "--asset", "7", "--budget", "2")
            self.assertEqual(client.transport(inspect, None, 3)["wire_bytes"], 0)
            write = ("append", "--asset", "7", "--budget", "3", "--file-offset", "0", "--wire-offset", "0",
                     "--digest", "a"*64)
            self.assertEqual(client.transport(write, b"\nhello\n", 5)["wire_bytes"], 1)
            self.assertEqual(client.transport(inspect, None, 3)["wire_bytes"], 1)
            child.stdin.close()
            self.assertEqual(child.wait(timeout=5), 0)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)
            child.stdin.close()
            child.stdout.close()


if __name__ == "__main__":
    unittest.main()
