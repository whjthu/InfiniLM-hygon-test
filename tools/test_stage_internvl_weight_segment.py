from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
from random import Random
from subprocess import Popen
import sys
from tempfile import TemporaryDirectory
from time import monotonic
import unittest
import zlib

from relay_internvl_weight_segment import relay
import stage_internvl_weight_segment as stage


class WeightSegmentTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()/'owned'
        self.original = b'abcdefghthe end'
        self.spec = dict(file_id=3,filename='constant.safetensors',bytes=len(self.original),
                         SHA256=sha256(self.original).hexdigest(),segment_bytes=8)
        stage.initialize(self.root,self.spec)

    def put(self, start, raw, expected=None):
        return stage.append(self.root,self.spec,start,expected or sha256(raw).hexdigest(),BytesIO(raw))

    def test_crashed_append_tail_is_discarded_only_beyond_stable_checkpoint(self):
        first, last = self.original[:8], self.original[8:]
        self.put(0,first)
        with (self.root/'payload.partial').open('ab') as stream:
            stream.write(b'uncommitted tail')
        view=stage.inspect(self.root,self.spec)
        self.assertEqual((view['confirmed_bytes'],view['actual_bytes']),(8,24))
        receipt=self.put(8,last)
        self.assertEqual(receipt['recovered_uncommitted_bytes'],16)
        self.assertFalse(receipt['whole_file_qualified'])
        self.assertEqual((self.root/'payload.partial').read_bytes(),self.original)
        self.assertTrue(stage.seal(self.root,self.spec)['whole_file_qualified'])
        self.assertTrue(stage.inspect(self.root,self.spec)['whole_file_qualified_at_seal'])
        self.assertEqual((self.root/self.spec['filename']).read_bytes(),self.original)

    def test_short_overlong_inconsistent_and_duplicate_offsets_never_advance(self):
        for raw,digest in ((b'short',None),(b'much too long',None),(self.original[:8],'0'*64)):
            with self.assertRaisesRegex(ValueError,'different segment'):
                self.put(0,raw,digest)
        self.assertEqual(stage.inspect(self.root,self.spec)['confirmed_bytes'],0)
        self.put(0,self.original[:8])
        with self.assertRaisesRegex(ValueError,'do not replay'):
            self.put(0,self.original[:8])

    def test_valid_per_segment_hashes_cannot_replace_the_original_whole_hash(self):
        self.put(0,b'badstart')
        self.put(8,self.original[8:])
        with self.assertRaisesRegex(ValueError,'whole shard SHA'):
            stage.seal(self.root,self.spec)
        self.assertFalse((self.root/self.spec['filename']).exists())

    def test_seal_recovery_uses_original_link_and_rejects_overwriting_another_file(self):
        self.put(0,self.original[:8]); self.put(8,self.original[8:])
        final=self.root/self.spec['filename']
        final.write_bytes(b'keep')
        with self.assertRaisesRegex(ValueError,'contains another file'):
            stage.seal(self.root,self.spec)
        self.assertEqual(final.read_bytes(),b'keep')
        final.unlink()
        os.link(self.root/'payload.partial',final)  # Link happened before a crash persisted the seal.
        self.assertTrue(stage.seal(self.root,self.spec)['whole_file_qualified'])

    def test_published_identity_and_last_committed_data_cannot_drift(self):
        self.put(0,self.original[:8]); self.put(8,self.original[8:]); stage.seal(self.root,self.spec)
        final=self.root/self.spec['filename']
        with final.open('r+b') as stream:
            stream.write(b'mutate')
        os.utime(final,ns=(1000000000,1000000000))
        with self.assertRaisesRegex(ValueError,'sealed inode'):
            stage.inspect(self.root,self.spec)

    def test_zlib_framing_reuses_the_frozen_wire_contract(self):
        protocol = stage.original_protocol()
        spec=stage.shard_spec(3)
        self.assertEqual(spec['filename'],'model-00003-of-00004.safetensors')
        raw = b'z'*stage.BLOCK_BYTES+b'last'
        framed = BytesIO()
        row=dict(file_id=3,bytes=len(raw),blocks=2)
        for i,part in enumerate((raw[:stage.BLOCK_BYTES],raw[stage.BLOCK_BYTES:])):
            wire=zlib.compress(part,1)
            framed.write(protocol['pack'](row,i,len(wire),sha256(part).digest(),sha256(wire).digest()))
            framed.write(wire)
        blob=framed.getvalue()
        wrapper=dict(row,segment_bytes=2*stage.BLOCK_BYTES)
        self.assertEqual(stage.framed_segment(wrapper,0,BytesIO(blob)).read(),raw)
        with self.assertRaisesRegex(ValueError,'digest differs'):
            stage.framed_segment(wrapper,0,BytesIO(blob[:-1]))
        with self.assertRaisesRegex(ValueError,'exact fixed frame range'):
            stage.framed_segment(wrapper,0,BytesIO(blob+b'x'))


class WeightSegmentRelayTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name).resolve()
        self.source = directory/'source'
        self.root = directory/'target'
        self.original = bytes(range(256))*(stage.BLOCK_BYTES//256)+b'short last segment'
        self.source.write_bytes(self.original)
        self.spec = dict(file_id=3, filename='constant.safetensors', bytes=len(self.original),
                         SHA256=sha256(self.original).hexdigest(), segment_bytes=stage.BLOCK_BYTES)
        self.binding = dict(zip(('device','inode','bytes','mtime_ns'), stage.source_stamp(self.source.stat())),
                            SHA256=self.spec['SHA256'])
        stage.initialize(self.root,self.spec)

    def pipe(self, offset, source_program, target_program, *, seconds=20):
        setup = ('import sys,json\nfrom pathlib import Path\nimport stage_internvl_weight_segment as stage\n'
                 's=json.loads(sys.argv[1]); b=json.loads(sys.argv[2]); off=int(sys.argv[3]); p=Path(sys.argv[4])\n')
        command = [sys.executable, '-B', '-c']
        args = [json.dumps(self.spec), json.dumps(self.binding), str(offset)]
        cwd = Path(__file__).resolve().parent
        return relay(
            self.spec, offset, monotonic()+seconds,
            lambda **stdio: Popen([*command,setup+source_program,*args,str(self.source)], cwd=cwd, **stdio),
            lambda **stdio: Popen([*command,setup+target_program,*args,str(self.root)], cwd=cwd, **stdio),
        )

    sender = 'print(json.dumps(stage.send_segment(p,b,s,off,sys.stdout.buffer)),file=sys.stderr)'
    receiver = 'print(json.dumps(stage.append(p,s,off,None,sys.stdin.buffer,envelope=True)))'

    def test_full_32mib_segment_through_pipe(self):
        self.original = Random(413).randbytes(stage.SEGMENT_BYTES)
        self.source.write_bytes(self.original)
        self.spec.update(bytes=len(self.original), SHA256=sha256(self.original).hexdigest(),
                         segment_bytes=stage.SEGMENT_BYTES)
        self.binding = dict(zip(('device','inode','bytes','mtime_ns'), stage.source_stamp(self.source.stat())),
                            SHA256=self.spec['SHA256'])
        self.root = self.root.with_name('full-segment-target')
        stage.initialize(self.root,self.spec)
        self.assertEqual(self.pipe(0,self.sender,self.receiver)['confirmed_bytes'],stage.SEGMENT_BYTES)
        self.assertTrue(stage.seal(self.root,self.spec)['whole_file_qualified'])

    def test_process_pipe_commits_two_segments_and_verifies_whole_original(self):
        first = self.pipe(0,self.sender,self.receiver)
        self.assertEqual(first['confirmed_bytes'],stage.BLOCK_BYTES)
        last = self.pipe(first['confirmed_bytes'],self.sender,self.receiver)
        self.assertEqual(last['confirmed_bytes'],len(self.original))
        self.assertEqual(last['segment_SHA256'],sha256(b'short last segment').hexdigest())
        self.assertTrue(stage.seal(self.root,self.spec)['whole_file_qualified'])
        self.assertEqual((self.root/self.spec['filename']).read_bytes(),self.original)

    def test_process_pipe_interrupted_segment_leaves_durable_prefix_unchanged(self):
        first = self.pipe(0,self.sender,self.receiver)
        partial = ('from io import BytesIO\nwire=BytesIO();stage.send_segment(p,b,s,off,wire)\n'
                   'sys.stdout.buffer.write(wire.getvalue()[:-1]);sys.exit(2)')
        with self.assertRaisesRegex(RuntimeError,'inspect the original target'):
            self.pipe(first['confirmed_bytes'],partial,self.receiver)
        observed = stage.inspect(self.root,self.spec)
        self.assertEqual(observed['confirmed_bytes'],first['confirmed_bytes'])
        self.assertEqual(observed['actual_bytes'],first['confirmed_bytes'])
        self.assertEqual(self.pipe(observed['confirmed_bytes'],self.sender,self.receiver)['confirmed_bytes'],len(self.original))

    def test_receipt_disagreement_requires_reconciliation_without_replay(self):
        altered = ('receipt=stage.append(p,s,off,None,sys.stdin.buffer,envelope=True)\n'
                   "receipt['segment_SHA256']='0'*64;print(json.dumps(receipt))")
        with self.assertRaisesRegex(RuntimeError,'inspect the original target'):
            self.pipe(0,self.sender,altered)
        observed = stage.inspect(self.root,self.spec)
        self.assertEqual(observed['confirmed_bytes'],stage.BLOCK_BYTES)
        with self.assertRaisesRegex(ValueError,'do not replay'):
            stage.append(self.root,self.spec,0,None,BytesIO(),envelope=True)

    def test_expired_process_pipe_leaves_uncommitted_target_available_for_inspection(self):
        with self.assertRaisesRegex(RuntimeError,'inspect the original target'):
            self.pipe(0,'import time;time.sleep(10)',self.receiver,seconds=1)
        self.assertEqual(stage.inspect(self.root,self.spec)['confirmed_bytes'],0)

    def test_source_replacement_rejected_before_writing_wire_bytes(self):
        replacement = self.source.with_name('replacement')
        replacement.write_bytes(self.original)
        mtime = self.binding['mtime_ns']
        os.utime(replacement,ns=(mtime,mtime))
        replacement.replace(self.source)
        wire = BytesIO()
        with self.assertRaisesRegex(ValueError,'source file stamp changed'):
            stage.send_segment(self.source,self.binding,self.spec,0,wire)
        self.assertEqual(wire.getvalue(),b'')

    def test_envelope_must_match_current_scope_and_whole_segment_hash(self):
        wire = BytesIO()
        stage.send_segment(self.source,self.binding,self.spec,0,wire)
        payload = wire.getvalue()[stage.SEGMENT_HEADER.size:]
        for offset,digest,expected in ((1,bytes(32),'envelope scope'),(0,bytes(32),'different segment')):
            header = stage.SEGMENT_HEADER.pack(stage.SEGMENT_MAGIC,3,offset,stage.BLOCK_BYTES,digest)
            with self.assertRaisesRegex(ValueError,expected):
                stage.append(self.root,self.spec,0,None,BytesIO(header+payload),envelope=True)
        self.assertEqual(stage.inspect(self.root,self.spec)['confirmed_bytes'],0)


if __name__ == '__main__':
    unittest.main()
