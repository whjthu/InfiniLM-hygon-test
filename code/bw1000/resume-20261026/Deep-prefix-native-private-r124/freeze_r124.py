import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
binding=json.loads((ROOT/'actual_binding_readonly_r124.json').read_text());checks=json.loads((ROOT/'CPU_source_checks_r124.json').read_text());plan=json.loads((ROOT/'compile_plan_r124.json').read_text())
report=dict(schema=1,revision='r124',card='D115A',UTC=datetime.now(timezone.utc).isoformat(),status='CONCRETE_NATIVE_PRIVATE_SOURCE_UNCOMPILED_NOT_QUALIFIED',
 baseline_DSO_SHA256=plan['baseline_DSO_SHA256'],
 implementation=[
 'Private C++ Transaction: begin/stage/forward_done/seal/publish/cancel; owner-thread/model/epoch/cohort checks;87 participants required (43attention+43decoder+model).',
 'Actual bound SWA/HCA/CSA forward commit sites stage accepted-prefix callbacks only under explicit private TLS Scope. Original OFF math/commit branch remains present.',
 'Actual bound NativeDecoder stages lengths and43-layer identity; TextModel stages final length only after head succeeds. Private methods return all-position logits and four-stream residual through existing internal Output.',
 'Seal uses existing score rows to scatter accepted1/2 persistence into fixed-size compressor state. Reuses completed compressed/rotated/key outputs, filtering rejected boundary rows. No projection/kernel/history replay.',
 'Ring: immutable touched physical block256 page COW retains exact raw bytes/padding and per-slot written flags; persistent HOST trie root swap publishes. Gather batches consecutive slots by backing page using existing kernel.',
 'Compressed/index caches: accepted new entries are append-only beyond published length. Prepare their bytes only during seal AFTER whole forward succeeds; written flags remain false until publication. Abort leaves unreachable bytes, never altered old valid entries.',
 'No-throw publication swaps cached page roots/state Tensor handles/length vectors/selection and increments revisions; preparation allocation/GPU status errors occur before publication.',
 'Whole-model reset cancels pending and invalidates epoch before reset; stale/foreign/consumed and ordinary-forward-while-pending reject. Existing failures preserve original poison/fail-stop behavior.',
 'Pure C++ EOS/cap policy helper and actual transaction-header C++ contract test source included. Known terminal anchors skip verify; rejected EOS ignored; accepted EOS/replacement EOS choose correct input prefix.'],
 binding=dict(receipt=str(ROOT/'actual_binding_readonly_r124.json'),SHA256=sha(ROOT/'actual_binding_readonly_r124.json'),
  compressor_source_SHA256='8e3160a8d3defff085c888dea3d78bf1d09f0b8d3c4d6cbf2d8f85735041feee',
  compressor_object='/refactor/candidates/deep-c4-projection-model-r5920/objects/unit-20.o',compressor_object_SHA256=binding['current_link']['objects']['/refactor/candidates/deep-c4-projection-model-r5920/objects/unit-20.o'],
  consumers='9 existing HOST TUs must rebuild with all modified private class headers; plus1 new adapter;39 current objects retained. NativeSparseModel allocator and TemporalCompressor allocator are explicitly replaced. No old8726/header-layout mixing.'),
 validation=dict(source_and_byte_checks=checks,CXX_contract_test_source='prefix_contract_test_r124.cpp',CXX_contract_test_run=False,CXX_compilation=False,native_tensor_or_kernel_execution=False),
 memory=dict(symbolic_r118_667364_bytes='NOT actual r124 prototype allocation bound',fixed_compressor_state_per_configured_physical_row=6103040,
  detail='Existing prepared next_state for ALL43 layers now remains retained until acceptance:6103040 bytes per configured row. Partial acceptance seal may retain another6103040-byte fixed-state clone per configured row. Existing base state remains too. Ring COW149760 bytes per touched physical page per layer;43*149760=6439680 bytes for one page in every layer. Two physical rows share a ring page. Retained pending projection/query/score/topk tensors, driver/allocator memory and metadata are additional; no complete peak/fit claim.'),
 cost='UNKNOWN. verify2 preserves q.tokens==1 guards and pays full21-layer indexer. Seal score copies, fixed-state copies, touched-page COW, gather page groups, HOST trie updates and eight-owner barriers are real costs. Continuing accept1 requires actual MTP catchup later; this component implements no MTP head and claims no acceleration.',
 remaining=[
 'No compilation/link/native forward has been run. Root should review source and the exact HOST plan, then authorize its CPU object build and actual C++ contract test execution.',
 'All8-owner seal acknowledgement/publish dispatch and worker/CAPI entry points remain caller integration work. Existing public CAPI still cannot invoke begin_prefix/forward_prefix; source is not currently enabled by any environment flag.',
 'Per-row reset/generation reuse is not newly implemented; only existing whole-model reset is wired. Terminal serving rows must release through a future defined row-lifecycle API.',
 'A segmented ring cache cannot truthfully expose one contiguous raw Tensor; raw() explicitly throws after first COW publication. Any future harness/cache_tensors consumer must use gather or add a private materialization/enumeration method.',
 'Original module/model poison handling remains fail-stop on runtime/forward errors. Cache staging avoids committed-cache advancement, but arbitrary device failure does not promise resumable rollback.',
 'Appended future raw bytes may remain after cancellation, with validity false. Tests must verify they remain unreadable and are overwritten before later acceptance; do not compare invalid raw bytes as committed state.',
 'Native qualification must exercise real accepted1/2, rejected boundary positions2/3/126/127/128, mixed rows, EOS/cap, stale/foreign/reset/cancel and injected failures across actual43 layers. CPU byte fixtures do not establish this.',
 'Private compile plan preserves original ac937 flags/context and adds known current include directories. Dependency MD must prove all altered class headers resolve privately; missing custom HOST headers must be resolved from the original per-object compile receipt without substituting runtime/SDK/math headers.'],
 next_minimal='Root review -> private CPU C++ contract test +10 HOST object compile with exact source/MD closure -> fix compile/API defects -> only then design a bounded native component qualification. Full MTP head/weights/model load remain outside this stage.',
 execution=dict(local_private_source_writes=True,repo_mutations=0,Git=0,upload=0,GPU=0,weights=0,compile=0,dlopen=0,Slurm=0,SSH_successful_readonly_queries=1,sandbox_failed_attempts=1,automatic_approval_rejections=0))
(ROOT/'implementation_receipt_r124.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
files=[dict(path=str(p.relative_to(ROOT)),SHA256=sha(p),bytes=p.stat().st_size) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name not in ['manifest_r124.json'] and '__pycache__' not in p.parts]
manifest=dict(schema=1,revision='r124',status=report['status'],root=str(ROOT),report='implementation_receipt_r124.json',files=files)
(ROOT/'manifest_r124.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(status=report['status'],files=len(files),report_SHA256=sha(ROOT/'implementation_receipt_r124.json'),manifest_SHA256=sha(ROOT/'manifest_r124.json'))))
