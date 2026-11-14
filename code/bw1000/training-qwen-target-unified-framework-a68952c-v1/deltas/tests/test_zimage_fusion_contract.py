"""Metadata rejection boundaries; tiny fixtures are not model weights."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'generation/src'))
from gencuda.zimage_fusion_contract import check_metadata, read_header, validate_header
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import hygon_zimage_fusion_check as checker


def tensor(shape, dtype='F32', begin=0):
    import math
    return dict(shape=shape, dtype=dtype,
                data_offsets=[begin, begin+math.prod(shape)*({'F32':4,'BF16':2}[dtype])])


class FusionContract(unittest.TestCase):
    def fixture(self):
        base={'layers.0.attention.to_q.weight':tensor([3,4])}
        adapter={'diffusion_model.layers.0.attention.to_q.lora_A.weight':tensor([2,4],'BF16'),
                 'diffusion_model.layers.0.attention.to_q.lora_B.weight':tensor([3,2],'BF16',16)}
        return base,adapter

    def test_complete_pair_preserves_inputs_and_reports_only_metadata(self):
        base,adapter=self.fixture();before=copy.deepcopy((base,adapter))
        result=check_metadata([base],adapter)
        self.assertEqual((base,adapter),before)
        self.assertEqual((result['preset_pairs'],result['new_rank4_tensors'],result['new_rank4_elements']),(1,2,28))
        self.assertFalse(result['preset_adapter_fused'])
        self.assertEqual(result['tensor_body_bytes_read'],0)

    def test_unmatched_target_cannot_silently_pass_like_native_loader(self):
        base,adapter=self.fixture();base['other.weight']=base.pop(next(iter(base)))
        with self.assertRaisesRegex(ValueError,'absent'):check_metadata([base],adapter)

    def test_orphan_factor_rejected(self):
        base,adapter=self.fixture();del adapter[next(iter(adapter))]
        with self.assertRaisesRegex(ValueError,'orphan'):check_metadata([base],adapter)

    def test_incompatible_rank_and_output_dimension_rejected(self):
        for shape in ([3,3],[4,2]):
            with self.subTest(shape=shape):
                base,adapter=self.fixture();adapter[list(adapter)[1]]['shape']=shape
                with self.assertRaisesRegex(ValueError,'shapes'):check_metadata([base],adapter)

    def test_storage_dtype_cannot_change_original_bf16_fusion_boundary(self):
        base,adapter=self.fixture();base[next(iter(base))]['dtype']='BF16'
        with self.assertRaisesRegex(ValueError,'dtype'):check_metadata([base],adapter)

    def test_extra_adapter_scale_tensor_is_not_ignored(self):
        base,adapter=self.fixture();adapter['diffusion_model.layers.0.alpha']=tensor([1])
        with self.assertRaisesRegex(ValueError,'unsupported'):check_metadata([base],adapter)

    def test_duplicate_shard_tensor_rejected(self):
        base,adapter=self.fixture()
        with self.assertRaisesRegex(ValueError,'duplicate'):check_metadata([base,base],adapter)

    def test_payload_overlap_and_gap_rejected(self):
        for begin in (0,8):
            with self.subTest(begin=begin):
                header={'a':tensor([1]),'b':tensor([1],begin=begin)}
                with self.assertRaisesRegex(ValueError,'range'):
                    validate_header(header,file_bytes=36,header_bytes=20)

    def test_bool_zero_and_wrong_byte_extent_rejected(self):
        for shape in ([True,4],[0,4]):
            with self.subTest(shape=shape):
                header={'a':dict(shape=shape,dtype='F32',data_offsets=[0,16])}
                with self.assertRaisesRegex(ValueError,'shape'):validate_header(header,file_bytes=44,header_bytes=20)
        header={'a':tensor([1])};header['a']['data_offsets']=[0,5]
        with self.assertRaisesRegex(ValueError,'extent'):validate_header(header,file_bytes=33,header_bytes=20)

    def test_file_reader_reads_only_header_and_rejects_changed_header(self):
        raw=json.dumps({'a':tensor([2])}).encode()
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'fixture.safetensors'
            p.write_bytes(struct.pack('<Q',len(raw))+raw+b'\x00'*8)
            pin=dict(bytes=p.stat().st_size,header_bytes=len(raw),header_SHA256=hashlib.sha256(raw).hexdigest())
            self.assertEqual(read_header(p,lambda:None,expected=pin),{'a':tensor([2])})
            pin['header_SHA256']='0'*64
            with self.assertRaisesRegex(ValueError,'differs'):read_header(p,lambda:None,expected=pin)

    def cli_fixture(self, root, *, bad_rank=False):
        model=root/'model';(model/'transformer').mkdir(parents=True)
        source=root/'source';source.mkdir();(source/'loader.py').write_bytes(b'# fixture, never imported\n')
        def write(path, header):
            raw=json.dumps(header).encode();payload=max(v['data_offsets'][1] for v in header.values())
            path.write_bytes(struct.pack('<Q',len(raw))+raw+b'\0'*payload)
            return dict(bytes=path.stat().st_size,header_bytes=len(raw),header_SHA256=hashlib.sha256(raw).hexdigest())
        shards={};bases=[]
        for i in range(3):
            name='shard-'+str(i)+'.safetensors';base={f'layers.{i}.attention.to_q.weight':tensor([3,4])}
            shards[name]=write(model/'transformer'/name,base);bases.append(base)
        _,adapter=self.fixture()
        if bad_rank:adapter[list(adapter)[1]]=tensor([3,3],'BF16',16)
        ap=root/'adapter.safetensors';ap_pin=write(ap,adapter)
        expected=check_metadata(bases,self.fixture()[1])
        contract=dict(schema='zimage-original-fusion-metadata-v1',base_headers=shards,adapter_header=ap_pin,
            source_files={'loader.py':dict(bytes=(source/'loader.py').stat().st_size,SHA256=hashlib.sha256((source/'loader.py').read_bytes()).hexdigest())},
            expected_summary={k:expected[k] for k in ['base_tensors','preset_pairs','new_rank4_elements']})
        cp=root/'contract.json';cp.write_text(json.dumps(contract));now=time.time()
        args=['--contract',str(cp),'--contract-sha256',hashlib.sha256(cp.read_bytes()).hexdigest(),
              '--model-root',str(model),'--preset-adapter',str(ap),'--source-root',str(source),
              '--output',str(root/'result.json'),'--epoch',str(now-1),'--deadline',str(now+30)]
        return args

    def test_cli_normal_metadata_without_importing_loader(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict('os.environ',dict(
                CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='')):
            root=Path(temp);args=self.cli_fixture(root)
            self.assertEqual(checker.main(args),0)
            result=json.loads((root/'result.json').read_bytes())
            self.assertEqual(result['completed_header_files'],4)
            self.assertFalse(result['Torch_imported'])
            self.assertFalse(result['preset_adapter_fused'])
            with self.assertRaisesRegex(ValueError,'already exists'):checker.main(args)

    def test_cli_shape_failure_retains_four_file_denominator_and_progress(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict('os.environ',dict(
                CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='')):
            root=Path(temp);args=self.cli_fixture(root,bad_rank=True)
            self.assertEqual(checker.main(args),1)
            result=json.loads((root/'result.json').read_bytes())
            self.assertEqual((result['planned_header_files'],result['completed_header_files']),(4,4))
            events=[json.loads(line) for line in (root/'result.progress.jsonl').read_text().splitlines()]
            self.assertEqual(events[0]['event'],'plan')
            self.assertFalse(events[-1]['passed'])

    def test_cli_expired_epoch_rejects_before_any_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);args=self.cli_fixture(root)
            args[args.index('--deadline')+1]=str(time.time()-1)
            with self.assertRaisesRegex(ValueError,'expired'):checker.main(args)
            self.assertFalse((root/'result.json').exists())

    def test_cli_visible_gpu_rejects_before_header_reads(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict('os.environ',dict(
                CUDA_VISIBLE_DEVICES='0',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='')):
            root=Path(temp);args=self.cli_fixture(root)
            with self.assertRaisesRegex(ValueError,'visibility'):checker.main(args)
            self.assertFalse((root/'result.json').exists())

    def test_cli_oversized_source_rejected_with_original_plan_preserved(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict('os.environ',dict(
                CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='')):
            root=Path(temp);args=self.cli_fixture(root)
            p=root/'contract.json';contract=json.loads(p.read_bytes())
            contract['source_files']['loader.py']['bytes']=2*1024**2+1
            p.write_text(json.dumps(contract))
            args[args.index('--contract-sha256')+1]=hashlib.sha256(p.read_bytes()).hexdigest()
            self.assertEqual(checker.main(args),1)
            result=json.loads((root/'result.json').read_bytes())
            self.assertEqual((result['planned_header_files'],result['completed_header_files']),(4,0))
            self.assertIn('bounded extent',result['error'])


if __name__=='__main__':unittest.main()
