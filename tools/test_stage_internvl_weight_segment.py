from hashlib import sha256
from io import BytesIO
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import zlib

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


if __name__ == '__main__':
    unittest.main()
