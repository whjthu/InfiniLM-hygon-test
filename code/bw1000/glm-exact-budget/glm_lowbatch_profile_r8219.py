"""CPU-only snapshot of existing thread-local native counters, no GPU ops."""
import ctypes, math, threading, time
class OwnerProfile:
    def __init__(self, lib, report, stage, rank):
        self.lib, self.report, self.stage, self.rank = lib, report, stage, rank
        self.owner = threading.get_ident()
        lib.modern_glm_pipeline_last_profile.argtypes = []
        lib.modern_glm_pipeline_last_profile.restype = ctypes.POINTER(ctypes.c_double)
    def capture(self, label, step, group, flat_tokens, descriptions):
        if not label.startswith('prof-') or step not in [0, 1, 2, 63, 127]: return
        assert threading.get_ident() == self.owner, 'profile must read same forward owner thread'
        before = time.perf_counter()
        pointer = self.lib.modern_glm_pipeline_last_profile()
        assert pointer
        values = [float(pointer[i]) for i in range(6)]
        assert all(math.isfinite(v) and v >= 0 for v in values), values
        self.report.setdefault('native_pipeline_profile_samples_r8219', []).append({
            'phase':label, 'round':step, 'stage':self.stage, 'rank':self.rank,
            'owner_thread':self.owner, 'rows':list(group), 'flat_tokens':flat_tokens,
            'descriptions':list(descriptions), 'values_seconds':values,
            'getter_wall_seconds':time.perf_counter()-before,
            'fields':['host_incoming','host_layers','host_outgoing_final_sync',
                      'stream_incoming','stream_layers','stream_outgoing'],
            'scope':'Existing native counters. Stream event time includes dependency waits; not GPU busy. Host/stream fields overlap and must not be summed.'})
