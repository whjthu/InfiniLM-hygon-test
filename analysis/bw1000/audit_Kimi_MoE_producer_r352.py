"""Offline audit of the bounded r352 SDK source/recipe reads; no Slurm/dlopen/GPU."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREFIX = 'Kimi-MoE-producer-r352-'


def load(suffix):
    return json.loads((HERE / (PREFIX + suffix + '.json')).read_bytes())


def main():
    scan, recipe, objects, elf = [load(x) for x in ['scan', 'build-identity', 'object-deps', 'ELF-sections']]
    assert scan['stopped'] is None and scan['seen'] == 4887
    sources = {x['path'].split('/Ops/Kimi/InfiniOps/')[1]: x for x in scan['matches']}
    generic = sources['src/native/cuda/ops/fused_moe_mxfp4/kernel.cuh']
    header = sources['src/native/cuda/hygon/ops/fused_moe_mxfp4/kernel.h']
    assert generic['SHA256'] == 'e33d9e0ab3ead7f6d9a9c928bc99dc92276acf399c682941f256bfc4c84b1897'
    assert header['SHA256'] == 'bd495744e997943a7a82aae391eff7b5ba17bf2b819e2a38187110b59e942fb4'
    assert elf['different_sections'] == ['.dynstr']
    assert elf['installed']['SHA256'] == '6d5a6204f7eef70e84deb9cc737a36c6ae2c1fbc75a0f70645557c8441ed85ca'
    expected = ['src/native/cuda/hygon/ops/fused_moe_mxfp4/kernel.h',
                'src/native/cuda/ops/fused_moe_mxfp4/kernel.cuh',
                'src/native/cuda/ops/fused_moe_mxfp4/mxfp4_math.cuh',
                'src/native/cuda/hygon/ops/fused_moe_mxfp4/kimi_mxfp4_dispatch_r330.cuh']
    dependencies = [x for x in objects['rows'] if x['path'].endswith('.o.d')]
    assert len(dependencies) == 2
    assert all(all(name in x['text'] for name in expected) for x in dependencies)
    receipt = next(x for p, x in recipe['files'].items() if p.endswith('/latest-sdk-r3672-Kimi-build.json'))
    assert receipt['SHA256'] == '4ff09088a653ec288a405a699b02c029c7bd9bfbc6ac22454a06da0b50c186e3'
    assert json.loads(receipt['text'])['library_sha256'] == elf['installed']['SHA256']
    report = {
        'schema': 1, 'card': 'r352', 'state': 'STOP_NO_DISTINCT_BOUNDED_OPERATOR_CANDIDATE',
        'GPU_calls': 0, 'Native_dlopen': False, 'Slurm_jobs': 0, 'resource_claim': None,
        'Native327_SHA256': '4dfdc9a7809b367ef91b476280f12bca6da0c12511eb14dae679147382246505',
        'installed_SDK_SHA256': elf['installed']['SHA256'],
        'closure': 'MoE source/implementation index, retained compiler recipe/dependency objects, and installed SDK code-section identity mapped; not a reproducible rebuild or observed runtime kernel/ISA trace.',
        'dispatch_chain': [
            'Native327 keeps r201/moe.o db8e6363b30be629401ef4b17f8eab0fbdcdb6e7f92f3b983c8968418bca15d3; source r201 lines410-422 selects FusedMoeMxfp4 Hygon implementation_index1 in the fallback.',
            'generated/src/operator_call_instantiations_6.cc lines153-167 explicitly instantiates active indices, DefaultConfig, Call, Make and OperatorFactory for FusedMoeMxfp4; generated/bindings/generated_dispatch_6.cc forwards the factory/call.',
            'Hygon kernel.h implementation1 inherits implementation0 and overrides ParallelRoutes=true; BF16 selects Launch<__nv_bfloat16>, then launchFusedMoeMxfp4R330.',
            'R330 first tries the process-optin T1 tiled branch; its following static path requires I384, so it cannot apply to actual I192. The remaining path is generic launchFusedMoeMxfp4.',
            'Generic launch uses W13 route*I CTAs and W2 token*H CTAs; implementation1/topk16 chooses W2Routes16.'
        ],
        'actual_geometry': {'T_decode': 1, 'T_W8_verify': 8, 'experts': 896, 'topk': 16,
                            'routed_H': 3584, 'local_I': 192, 'activation': 2,
                            'weights': 'packed UInt8 E2M1 two nibbles/byte; UInt8 E8M0 exponent scale per32 weights',
                            'activation_dtype': 'BF16; source arithmetic/reduction float'},
        'already_implemented': [
            'W13 gate/up share each input pair, decode the two packed rows into registers, accumulate FP32, block-reduce both, and fuse SITU plus BF16 activated-store. No full BF16 weight materialization in this generic path.',
            'W2Routes16 assigns warp32 to one route, handles8 routes concurrently per tile over two tiles for topk16, warp-reduces each partial, then sums weighted routes in ascending order before one BF16 store. Proposing parallel routes again is a duplicate.',
            'SDK already contains process-optin INFINILM_MXFP4_TILE_TP_R2559 for BF16/T1/H3584/topk16/activation2 and I64/96/192/384, with I192 Tile4. This is existing code, not a new kernel mechanism.',
            'No native FP4 GEMM/MMAC or packed FP32 pair FMA is demonstrated by the generic source: nibble table plus ldexpf produces float weights, BF16 input converts to float. Actual emitted scalar FMA contraction/ISA was not disassembled.'
        ],
        'packing_and_route_reuse_boundary': [
            'T1 topk16 selects distinct experts within that token; different routes therefore do not automatically share expert weights. T8 cross-token reuse depends on actual expert overlap, which was not captured here.',
            'Generic W13 reuses one input pair for gate and up of a single output channel; it does not group tokens by expert or share decoded weights across tokens. W2Routes16 reuses no output-channel pair; the separate Model W2_PAIR optin is disabled in r346.',
            'Source logical packed-byte loads and useful-FLOP/address-set ratios do not measure HBM traffic, cache hit rate, achieved arithmetic utilization or a roofline.'
        ],
        'retained_build': {
            'directory': recipe['build_dir'],
            'compiler': '/opt/dtk/cuda/cuda/bin/nvcc -x cu -arch=gfx936 -O3 -DNDEBUG, C++17, ABI1',
            'objects_and_dependencies': [{k: x[k] for k in ['path', 'bytes', 'SHA256']} for x in objects['rows'] if '.cc.o' in x['path']],
            'commands': elf['retained_compile_commands'],
            'build_library_SHA256': elf['build']['SHA256'],
            'installed_library_SHA256': elf['installed']['SHA256'],
            'only_different_ELF_section': '.dynstr',
            'matching_code_sections': {k: elf['installed']['sections'][k] for k in ['.text', '.hip_fatbin', '.rodata', '.note.gnu.build-id']},
            'limitation': 'Retained .o.d files bind source paths, not historical source-byte digests. Current source hashes also match earlier archived r8669u/r166 hashes, but no fresh reproducible SDK rebuild was performed.'
        },
        'source_manifest': {k: {'SHA256': v['SHA256'], 'bytes': v['bytes']} for k, v in sources.items()},
        'known_negative_dedup': [
            {'mechanism': 'Direct register decode to BF16 MMAC, coalesced LDS/padding, persistent/hybrid, guarded FDOT2/postscale', 'evidence': 'analysis/bw1000/Kimi-next-expert-placement-card-r8669zi.json', 'decision': 'Previously qualified sparse-route negatives; no equivalent repeat.'},
            {'mechanism': 'W2 group32 A8/exact W8 register scale', 'evidence': 'coordination/tasks/kimi-int8-group32-w2.json', 'decision': '753/757 qualified mixed and PARK; no equivalent repeat.'},
            {'mechanism': 'W2 output-channel reuse2 to4', 'evidence': 'results/bw1000/2026-10-28/kimi1184-W2-quad4-negative.json', 'decision': 'Qualified no single-user gain, PARK; no equivalent repeat.'},
            {'mechanism': 'Expert placement TP8xEP2', 'evidence': 'analysis/bw1000/Kimi-next-expert-placement-card-r8669zi.json', 'decision': 'Different mechanism but already proposed/implemented in the historical family and wider than one minimal kernel edit; not reopened here.'}
        ],
        'remaining_unknown': [
            'Original536 did not preserve the exact process value of INFINILM_MXFP4_TILE_TP_R2559 or a selected-kernel counter; the r346 shell neither sets nor unsets this inherited SDK flag. Therefore generic-versus-existing-T1-tile selection in that original run is not certified by this read-only source card.',
            'An exact current T1/T8 W13-versus-W2 phase split, actual route overlap histogram and emitted GPU ISA were not captured; aggregate MoE critical-path cost cannot rank a fresh low-level change.',
            'Full Model wall impact of any new MoE candidate remains UNKNOWN. No model load, new observer, GPU profiling, grid or resource claim was added.'
        ],
        'actionable_gap': 'Before a new operator card, require a distinct source change absent from the existing tile/parallel-route/MMAC/INT8/quad4 negatives, an explicit eligible branch/path counter, independent FP4+scale+SITU reference, same installed-SDK path baseline, and a fixture covering T1/T8 with actual route reuse. Current evidence does not justify paying GPU time for another equivalent variant.',
        'decision': 'STOP within the CPU/read-only card; preserve original failed536 and parked kernel conclusions. Do not toggle existing tile optin or expand SDK/cache/framework under this task.'
    }
    raw_files = sorted(HERE.glob(PREFIX + '*.json'))
    report['raw_evidence'] = [{'path': str(p.relative_to(HERE.parent.parent)), 'SHA256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size} for p in raw_files]
    out = HERE / 'Kimi-MoE-producer-binding-r352.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps({'completed': True, 'GPU': 0, 'state': report['state'], 'output': str(out), 'SHA256': hashlib.sha256(out.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
