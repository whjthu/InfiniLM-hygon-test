"""Check exact staged Z-Image source and tensor metadata without loading weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('contract', 'model-root', 'preset-adapter', 'source-root', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--contract-sha256', required=True)
    parser.add_argument('--epoch', type=float, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args(argv)
    start, anchor = time.monotonic(), time.time()
    if not (math.isfinite(args.epoch) and math.isfinite(args.deadline)
            and 0 < args.deadline-anchor <= args.deadline-args.epoch <= 60):
        raise ValueError('invalid or expired original metadata epoch')
    if any(os.environ.get(k) != '' for k in
           ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES')):
        raise ValueError('metadata checker requires explicit empty device visibility')
    remaining = args.deadline-anchor
    def deadline():
        if time.time() >= args.deadline or time.monotonic()-start >= remaining:
            raise TimeoutError('original metadata epoch exhausted')
    if (not args.contract.is_file() or args.contract.is_symlink()
            or not 0 < args.contract.stat().st_size <= 128*1024):
        raise ValueError('original metadata contract exceeds bounded extent')
    raw = args.contract.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.contract_sha256:
        raise ValueError('original metadata contract identity differs')
    contract = json.loads(raw)
    if contract['schema'] != 'zimage-original-fusion-metadata-v1':
        raise ValueError('unsupported original metadata contract')
    progress_path = args.output.with_suffix('.progress.jsonl')
    if args.output.exists() or progress_path.exists():
        raise ValueError('metadata output already exists')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'generation/src'))
    from gencuda.zimage_fusion_contract import check_metadata, read_header
    result = dict(passed=False, planned_header_files=4, completed_header_files=0,
                  planned_source_files=len(contract['source_files']), completed_source_files=0,
                  GPU_cards=0, tensor_body_bytes_read=0, preset_adapter_fused=False)
    with progress_path.open('x', encoding='utf8') as log:
        def progress(event):
            log.write(json.dumps(event, allow_nan=False)+'\n')
            log.flush()
        try:
            progress(dict(event='plan', **result))
            if not 0 < len(contract['source_files']) <= 16:
                raise ValueError('original source file count exceeds bounded extent')
            total_source_bytes=0
            for name, pin in contract['source_files'].items():
                deadline()
                if type(pin['bytes']) is not int or not 0 < pin['bytes'] <= 2*1024**2:
                    raise ValueError('original source file exceeds bounded extent')
                total_source_bytes+=pin['bytes']
                if total_source_bytes > 8*1024**2:
                    raise ValueError('original source bytes exceed bounded extent')
                relative=Path(name)
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError('source path escapes the pinned root')
                p=args.source_root/relative
                if p.is_symlink() or not p.is_file() or p.stat().st_size != pin['bytes']:
                    raise ValueError('original fusion source extent differs')
                source=p.read_bytes()
                if hashlib.sha256(source).hexdigest() != pin['SHA256']:
                    raise ValueError('original fusion source differs')
                result['completed_source_files']+=1
                progress(dict(event='source', name=name, SHA256=pin['SHA256']))
            shards=[]
            if len(contract['base_headers']) != 3:
                raise ValueError('original three-shard transformer required')
            for name,pin in contract['base_headers'].items():
                if Path(name).name != name:
                    raise ValueError('invalid base shard name')
                shards.append(read_header(args.model_root/'transformer'/name, deadline, expected=pin))
                result['completed_header_files']+=1
                progress(dict(event='header', name=name, header_SHA256=pin['header_SHA256']))
            adapter=read_header(args.preset_adapter, deadline, expected=contract['adapter_header'])
            result['completed_header_files']+=1
            progress(dict(event='header', name=args.preset_adapter.name,
                          header_SHA256=contract['adapter_header']['header_SHA256']))
            observed=check_metadata(shards, adapter)
            for k,v in contract['expected_summary'].items():
                if observed[k] != v:
                    raise ValueError('original adapter metadata summary differs: '+k)
            result.update(metadata=observed, contract_SHA256=args.contract_sha256, passed=True)
            deadline()
            if 'torch' in sys.modules:
                raise RuntimeError('metadata check unexpectedly imported Torch')
        except BaseException as error:
            result.update(error_type=type(error).__name__, error=str(error))
        result.update(inside_seconds=time.monotonic()-start, epoch=args.epoch, deadline=args.deadline,
                      Torch_imported='torch' in sys.modules)
        progress(dict(event='complete', passed=result['passed'],
                      completed_header_files=result['completed_header_files']))
    with args.output.open('x', encoding='utf8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({k:result[k] for k in ['passed','planned_header_files','completed_header_files',
        'completed_source_files','GPU_cards','tensor_body_bytes_read','preset_adapter_fused','inside_seconds']}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
