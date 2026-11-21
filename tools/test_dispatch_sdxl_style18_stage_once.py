from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import dispatch_sdxl_style18_stage_once as dispatch


class SDXLStageTransportTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.repo = Path(directory.name).resolve(strict=True)
        receiver = self.repo / dispatch.RECEIVER['path']
        receiver.parent.mkdir()
        receiver.write_bytes((dispatch.REPO / dispatch.RECEIVER['path']).read_bytes())
        self.raw = (bytes(range(256)) * 1136)[:dispatch.PACKET_BYTES]
        self.packet = self.repo / dispatch.PACKET
        self.packet.parent.mkdir(parents=True)
        self.packet.write_bytes(self.raw)
        patches = patch.object(dispatch, 'PACKET_SHA', sha256(self.raw).hexdigest())
        patches.start()
        self.addCleanup(patches.stop)
        self.remote_checkout = str(dispatch.ROOT)

    def response(self, command, state, **kwargs):
        doc = dict(state=state, request=dispatch.REQUEST, training_READY=False, **kwargs)
        return subprocess.CompletedProcess(command, 0, json.dumps(doc).encode(), b'')

    def completed(self, command, state, **kwargs):
        return self.response(command, state, packet_SHA256=dispatch.PACKET_SHA, binding_SHA256=dispatch.BINDING,
                             files=13, train_bytes=289527, **kwargs)

    def receive(self, callback):
        return dispatch.receive_once(self.remote_checkout, callback, repo=self.repo)

    def test_binary_stdio_and_durable_original_intent_before_transfer(self):
        calls = []

        def transport(command, *, input, timeout):
            action = command[-3]
            calls.append(action)
            self.assertEqual(command[-2:], ('--packet-sha256', dispatch.PACKET_SHA))
            self.assertEqual(timeout, dispatch.TIMEOUT)
            if action == 'inspect':
                self.assertEqual(input, b'')
                return self.response(command, 'ORIGINAL_DATA_NAMESPACE_ABSENT')
            self.assertEqual(command[0:2], ('/usr/local/python3.12/bin/python3', '-B'))
            intent = json.loads((self.repo / dispatch.WORK / 'intent.json').read_bytes())
            self.assertEqual((intent['request'], intent['packet_bytes']), (dispatch.REQUEST, 290733))
            self.assertFalse(intent['replay_allowed'])
            local = subprocess.run([sys.executable, '-B', '-c',
                'import sys,hashlib; data=sys.stdin.buffer.read(); print(len(data),hashlib.sha256(data).hexdigest())'],
                input=input, capture_output=True, timeout=3, check=True)
            self.assertEqual(local.stdout.split(), [b'290733', dispatch.PACKET_SHA.encode()])
            return self.completed(command, 'SDXL_TRAIN_DATA_STAGED', wall_seconds=0.01)

        result = self.receive(transport)
        self.assertEqual(calls, ['inspect', 'receive'])
        self.assertEqual(result['state'], 'SDXL_TRAIN_DATA_STAGED')
        self.assertFalse(result['target_installed_verified'])

    def test_unknown_does_not_replay_and_independent_inspect_can_verify(self):
        def lost(command, *, input, timeout):
            if command[-3] == 'inspect':
                return self.response(command, 'ORIGINAL_DATA_NAMESPACE_ABSENT')
            raise subprocess.TimeoutExpired(command, timeout)

        first = self.receive(lost)
        self.assertEqual(first['state'], 'UNKNOWN')
        self.assertEqual(first['transport_or_receipt_error'], 'TimeoutExpired')
        with self.assertRaisesRegex(ValueError, 'use inspect only'):
            self.receive(lambda *args, **kwargs: self.fail('would replay'))

        def observe(command, *, input, timeout):
            self.assertEqual(input, b'')
            return self.completed(command, 'SDXL_TRAIN_DATA_STAGED_AND_RECHECKED')

        checked = dispatch.inspect(self.remote_checkout, observe)
        self.assertTrue(checked['target_installed_verified'])
        self.assertEqual(checked['training_ready_added'], 0)
        self.assertFalse(checked['replay_allowed'])

    def test_incomplete_or_already_installed_never_enters_receive(self):
        for state in ('ORIGINAL_DATA_NAMESPACE_INCOMPLETE', 'SDXL_TRAIN_DATA_STAGED_AND_RECHECKED'):
            def transport(command, *, input, timeout):
                self.assertEqual((command[-3], input), ('inspect', b''))
                if state == 'ORIGINAL_DATA_NAMESPACE_INCOMPLETE':
                    return self.response(command, state)
                return self.completed(command, state)
            result = self.receive(transport)
            self.assertFalse(result['receive_attempted'])
            self.assertEqual(result['state'], state)
        self.assertFalse((self.repo / dispatch.WORK).exists())

    def test_invalid_transport_receipt_or_packet_is_not_staged(self):
        result = self.receive(lambda command, **opts: subprocess.CompletedProcess(command, 0, b'"wrong"', b''))
        self.assertEqual(result['state'], 'UNKNOWN')
        self.assertFalse(result['receive_attempted'])
        self.packet.write_bytes(self.raw + b'\0')
        with self.assertRaisesRegex(ValueError, 'packet bytes differ'):
            self.receive(lambda *args, **kwargs: self.fail('would transfer the wrong bytes'))
        self.assertFalse((self.repo / dispatch.WORK).exists())

    def test_checkout_argument_cannot_escape_the_existing_project(self):
        for path in ('/tmp/other', self.remote_checkout + '/../other', self.remote_checkout + '/a;other'):
            with self.assertRaisesRegex(ValueError, 'canonical checkout'):
                dispatch.argv(path, 'receive')


if __name__ == '__main__':
    unittest.main()
