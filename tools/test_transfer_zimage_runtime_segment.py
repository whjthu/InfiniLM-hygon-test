"""Exercise the Z-Image adapter with the real frozen receiver in a disposable target."""

import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from hashlib import sha256

import transfer_zimage_runtime_segment as transfer


class Response(io.BytesIO):
    def __init__(self, data, status, headers):
        super().__init__(data)
        self.status, self.headers = status, headers

    def geturl(self):
        return 'https://huggingface.co/pinned'


class RelayTests(unittest.TestCase):
    def test_real_published_scope_is_20_unique_assets(self):
        actual = transfer.load_pins()['assets']
        self.assertEqual([row['asset_id'] for row in actual],list(range(1,21)))
        self.assertEqual(len({row['target_relative_path'] for row in actual}),20)
        self.assertEqual(sum(row['bytes'] for row in actual),33018433789)

    def test_complete_http_200_checks_body_sha(self):
        spec = {'bytes':3,'segment_bytes':transfer.SEGMENT_BYTES,'publisher_URL':'https://huggingface.co/pinned',
                'SHA256':sha256(b'abc').hexdigest()}
        self.assertEqual(transfer.source_range(spec,0,time.monotonic()+30,open_url=lambda *_a,**_k:
            Response(b'abc',200,{'Content-Length':'3'})), b'abc')
        with self.assertRaisesRegex(ValueError,'file SHA differs'):
            transfer.source_range(spec,0,time.monotonic()+30,open_url=lambda *_a,**_k:
                Response(b'bad',200,{'Content-Length':'3'}))
        spec['bytes'] = 2*transfer.SEGMENT_BYTES
        with self.assertRaisesRegex(ValueError,'exact immutable range'):
            transfer.source_range(spec,0,time.monotonic()+30,open_url=lambda *_a,**_k:
                Response(b'does not matter',200,{'Content-Length':str(spec['bytes'])}))

    def test_remote_body_resumes_without_replaying_ambiguous_append(self):
        size = transfer.stage.BLOCK_BYTES
        data = b'a'*size+b'last tail'
        row = dict(asset_id=1,filename='config.json',bytes=len(data),SHA256=sha256(data).hexdigest(),
                   kind='base',target_relative_path='base/config.json',publisher_URL='https://huggingface.co/pinned')
        downloaded = []
        dropped = False

        def fetch(spec,offset,deadline):
            downloaded.append(offset)
            return data[offset:offset+size]

        def exchange(argv,body,timeout):
            nonlocal dropped
            self.assertEqual(argv,transfer.REMOTE_COMMAND)
            length, = struct.unpack('<Q',body[:8])
            source = body[8:8+length].decode()
            tail = body[8+length:]
            # The same exact receiver is run in a local subprocess. Provide the
            # monotonic outer timeout on hosts without POSIX SIGALRM.
            if sys.platform == 'win32':
                source = '\n'.join(line for line in source.splitlines()
                                   if not line.startswith(('signal.signal(signal.SIGALRM','signal.setitimer(signal.ITIMER_REAL')))
            harness = ('import sys\n'
                       'n=int.from_bytes(sys.stdin.buffer.read(8),"little")\n'
                       'exec(compile(sys.stdin.buffer.read(n),"test_receiver","exec"))\n')
            source = source.encode()
            result = subprocess.run([sys.executable,'-I','-S','-B','-c',harness],
                input=struct.pack('<Q',len(source))+source+tail,capture_output=True,timeout=min(15,timeout))
            if result.returncode == 0 and not dropped and json.loads(result.stdout)['operation'] == 'append':
                dropped = True
                return subprocess.CompletedProcess([],255,b'',b'connection lost after durable target append')
            return result

        with tempfile.TemporaryDirectory(prefix='zimage-source-relay-test-') as temp:
            sandbox = Path(temp).resolve()
            target = sandbox / 'target'
            logs = sandbox / 'receipts'
            logs.mkdir()
            with (patch.object(transfer,'TARGET',str(target)),patch.object(transfer,'SEGMENT_BYTES',size),
                  patch.object(transfer,'load_pins',return_value={'assets':[row]}),
                  patch.object(transfer,'source_range',side_effect=fetch)):
                def one(offset):
                    return transfer.relay(1,offset,'a'*40,logs,exchange)
                with self.assertRaisesRegex(RuntimeError,'append returned 255'):
                    one(0)
                self.assertEqual(transfer.inspect_target_asset(1,exchange)['confirmed_bytes'],size)
                with self.assertRaisesRegex(ValueError,'target durable prefix differs'):
                    one(0)
                self.assertEqual(one(size)['target']['confirmed_bytes'],len(data))
                self.assertEqual(one(len(data))['action'],'sealed')
                self.assertEqual(one(len(data))['action'],'already_sealed')
            self.assertEqual(downloaded,[0,size])
            self.assertEqual((target/'asset-01/config.json').read_bytes(),data)
            self.assertTrue(all(x.suffix == '.json' and x.stat().st_size < 8192
                                for x in logs.rglob('*') if x.is_file()))


if __name__ == '__main__':
    unittest.main()
