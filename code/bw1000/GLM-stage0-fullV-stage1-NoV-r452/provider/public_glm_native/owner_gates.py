"""Concrete original GLM owner gates around static constructor/load/service.

No gate is an optional callback. Calls operate on the actual passed Native ctx
and original TP/WORLD communicators; a failed vote never becomes READY.
"""
import ctypes
import hashlib
import json
import os
import threading
from pathlib import Path
from .model_affinity_check_r2628 import verify as verify_affinity
from .cache_policy_r452 import for_config,from_environment

BOUNDS = [0, 38, 78]


class NativeGates:
    def __init__(self, *, torch, rank, stage, stream, config, publish_failure):
        self.torch, self.rank, self.stage, self.stream = torch, rank, stage, stream
        self.config, self.publish_failure = config, publish_failure
        self.affinity = None

    def _counts(self, lib, name, count, ctx=None):
        fn = getattr(lib, name)
        if ctx is None:
            fn.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
            values = (ctypes.c_uint64 * count)()
            rc = fn(self.rank, values, count)
        else:
            fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
            values = (ctypes.c_uint64 * count)()
            rc = fn(ctx, values, count)
        if rc: raise RuntimeError(name + " failed")
        return list(values)

    def _vote(self, b, ctx, values):
        x = self.torch.tensor(values, device='cuda', dtype=self.torch.float32)
        b.check(b.lib.modern_glm_model_control(ctx, x.numel(), b.ptr(x), self.stream()))
        result = x.cpu().tolist()
        assert result == [16] * len(values), ('all16 vote', values, result)

    def after_create(self, b, ctx, report):
        lib = b.lib
        lib.modern_glm_model_tp_control.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p]
        lib.modern_glm_cross_control_r5775.argtypes = [ctypes.c_int] * 4
        lib.modern_glm_cross_journal_r5775.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.modern_glm_cross_journal_r5775.restype = ctypes.c_char_p
        error, identity = None, None
        try:
            b.check(lib.modern_glm_cross_control_r5775(self.rank, 0, 0, -1))
            raw = lib.modern_glm_cross_journal_r5775(self.rank, 1)
            assert raw
            identity = json.loads(raw)
            assert sorted(v['layer'] for v in identity['identities']) == list(range(BOUNDS[self.stage], BOUNDS[self.stage + 1]))
            assert all(v['stage'] == self.stage and v['rank'] == self.rank for v in identity['identities'])
        except BaseException as failure:
            error = repr(failure)
        self._vote(b, ctx, [float(error is None)])
        report['registry_owner_before_weightIO'] = dict(identities=identity, all16passed=True,
            synchronous_direct_CAPI_owner_thread=True)

    def before_load(self, b, ctx, report):
        try:
            lib = b.lib
            exact = self._counts(lib, 'modern_glm_exact_banks_stats_r4188', 4, ctx)
            policy=for_config(self.config,self.stage)
            assert from_environment(os.environ,self.stage)==policy
            n = policy.exact_bank_count
            enabled = int(os.environ['GLM_EXACT_BANKS_R4188'])
            assert enabled in (0, 1) and exact[:3] == [n, n * enabled, n * (1 - enabled)]
            report['exact_bank_dispatch_r4188'] = exact
            state = self._counts(lib, 'modern_glm_service_memory_stats_r4279', 9, ctx)
            assert state[:3] == [1, 1, 32768] and state[5] == state[6] and state[5] > 0
            assert state[7:] == [2, 32768 * 256]
            report['service_memory_after_create_r4279'] = state
            assert self._counts(lib, 'modern_glm_projected_host_counts_r4504', 7) == [0] * 7
            warm = []
            for elements in [1, 1024, 65536, 262144]:
                x = self.torch.ones(elements, device='cuda', dtype=self.torch.float32)
                b.check(lib.modern_glm_model_tp_control(ctx, elements, b.ptr(x), self.stream()))
                assert bool((x == 8).all().item())
                y = self.torch.ones(elements, device='cuda', dtype=self.torch.float32)
                b.check(lib.modern_glm_model_control(ctx, elements, b.ptr(y), self.stream()))
                assert bool((y == 16).all().item())
                warm.append(dict(elements=elements, TP8_integer_SUM_exact=True, world16_integer_SUM_exact=True))
                del x, y
            self.torch.cuda.synchronize(self.rank)
            self.torch.cuda.empty_cache()
            report['preload_communicator_warmup'] = warm
            report['memory_after_create'] = self.torch.cuda.mem_get_info(self.rank)
            self._vote(b, ctx, [float(report['memory_after_create'][0] > 8 * 1024 ** 3)])
            report['all16_preweight_8GiB_passed'] = True
        except BaseException as error:
            self.publish_failure(error)
            raise

    def after_load(self, b, ctx, report):
        try:
            lib = b.lib
            invalid = (ctypes.c_uint64 * 6)(*[2 ** 64 - 1] * 6)
            assert lib.modern_glm_unloaded_trim_r4052(ctx, invalid, 6) != 0
            assert list(invalid) == [2 ** 64 - 1] * 6
            report['unloaded_trim_after_loading_rejected'] = True
            self.affinity = verify_affinity(lib, self.rank)
            assert self.affinity['width'] == 4
            report['owner_affinity_after_load'] = self.affinity
            m15 = self._counts(lib, 'modern_glm_m15_oproj_stats_r125', 17)
            batch = self._counts(lib, 'modern_glm_model_batch_cache_stats_r3068', 5, ctx)
            assert m15 == [0] * 17 and batch == [0] * 5
            lib.modern_glm_model_m15_oproj_policy_r125.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.modern_glm_model_batch_cache_policy_r3068.argtypes = [ctypes.c_void_p, ctypes.c_int]
            assert lib.modern_glm_model_m15_oproj_policy_r125(ctx, 2) != 0
            assert self._counts(lib, 'modern_glm_m15_oproj_stats_r125', 17) == m15
            assert lib.modern_glm_model_batch_cache_policy_r3068(ctx, 2) != 0
            assert self._counts(lib, 'modern_glm_model_batch_cache_stats_r3068', 5, ctx) == batch
            # Preserve original default-OFF foreign-owner refusals, without
            # changing policy state or accepting a current-thread fake bind.
            refusal = []
            def foreign():
                refusal.append(lib.modern_glm_model_m15_oproj_policy_r125(ctx, 1))
                refusal.append(lib.modern_glm_model_batch_cache_policy_r3068(ctx, 1))
            thread = threading.Thread(target=foreign)
            thread.start();thread.join(20)
            assert not thread.is_alive() and refusal == [1, 1]
            assert self._counts(lib, 'modern_glm_m15_oproj_stats_r125', 17) == m15
            assert self._counts(lib, 'modern_glm_model_batch_cache_stats_r3068', 5, ctx) == batch
            report.update(M15_default_OFF_verified=True, batch_cache_default_OFF_verified=True)
            paths = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'libinfiniops.so' in line})
            assert len(paths) == 1
            assert hashlib.sha256(Path(paths[0]).read_bytes()).hexdigest() == self.config['SDK_SHA256']
            helpers = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'libglm_fp8_' in line}
            assert len(helpers) == 2
            actual = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in helpers}
            assert actual == self.config['FP8_helpers_SHA256']
            report['actual_FP8_helpers'] = actual
            free, total = self.torch.cuda.mem_get_info(self.rank)
            self._vote(b, ctx, [float(report['loaded_parameters'] > 0), float(free > 4 * 1024 ** 3)])
            report.update(all16_loaded_and_postload_4GiB_passed=True, memory_after_load=[free, total])
        except BaseException as error:
            self.publish_failure(error)
            raise

    def before_destroy(self, b, ctx, report):
        final = verify_affinity(b.lib, self.rank)
        def masks(value):
            return value['width'], sorted((r['device'], r['changes'], r['verified_mask'])
                for r in value['actual_owner_report']['owners'])
        assert masks(final) == masks(self.affinity), 'Actual owner mask changed during service'
        report['owner_affinity_final'] = final
        policy = ctypes.CDLL(None)
        policy.modern_blas_policy_r243_json.restype = ctypes.c_char_p
        state = json.loads(policy.modern_blas_policy_r243_json())
        assert state['failures'] == 0 and all(v['created'] == v['verified'] for v in state['owners'])
        report['BLAS_policy'] = state
        report['host_bank_before_destroy_r4504'] = self._counts(b.lib, 'modern_glm_projected_host_counts_r4504', 7)
        report['cache_before_destroy'] = self._cache(b.lib)
        report['service_memory_before_destroy_r4279'] = self._counts(b.lib, 'modern_glm_service_memory_stats_r4279', 9, ctx)

    def after_destroy(self, b, report):
        raw = self._cache(b.lib)
        assert raw['created'] == raw['destroyed'] and raw['active'] == raw['active_bytes'] == 0
        m15 = self._counts(b.lib, 'modern_glm_m15_oproj_stats_r125', 17)
        assert m15[0] == 0 and m15[7] == m15[8] and m15[12] == m15[13] == 0 and m15[5] == m15[16] == 0
        host = self._counts(b.lib, 'modern_glm_projected_host_counts_r4504', 7)
        assert host[:3] == [0, 0, 0] and all(x >= 0 for x in host[3:])
        registry = b.lib.modern_glm_cross_journal_r5775(self.rank, 1)
        assert registry and not json.loads(registry)['identities']
        report.update(cache_after_destroy=raw, M15_after_destroy=m15,
                      host_bank_after_destroy_r4504=host, registry_destroyed=True)

    def after_shutdown(self, report):
        self.torch.cuda.synchronize(self.rank)
        assert self.torch.isfinite(self.torch.ones(8, device='cuda')).all().item()
        self.torch.cuda.synchronize(self.rank)
        report['post_close_Torch_liveness'] = True

    def _cache(self, lib):
        return dict(zip(['created','destroyed','hits','misses','evictions','active',
            'active_bytes','highwater_bytes','runs','invalid'],
            self._counts(lib, 'modern_glm_fp8_runtime_stats_r809', 10)))
