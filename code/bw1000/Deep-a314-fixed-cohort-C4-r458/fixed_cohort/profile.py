"""Explicit private profile identity. It does not claim Model readiness."""
import ctypes as C

PROFILE = 'Deep-a314-fixed-cohort-C4-r458'
NATIVE = 'a314ae6891977316189f875b793ed3f2e2eb0898110ad0cc382d2569eaf31dc1'


def config(model_path):
    return dict(profile=PROFILE, rows=4, capacity_per_row=32768, HC_mode=0,
                r366=0, fullpre=0, Native_SHA256=NATIVE, model_path=model_path,
                TP=8, PP=1, online=False, standard_public_C1=False,
                Model_READY=False)


def binding_descriptor(parent):
    assert parent['rows'] == 1 and parent['capacity'] == 32768
    assert parent['library_SHA256'] == NATIVE and parent['Model50objects']
    return dict(parent, rows=4, capacity=32768, capacity_per_row=32768,
                profile=PROFILE, standard_public_C1=False,
                source_only_descriptor=True, Model_READY=False)


def check_modes(lib):
    names = ['deep_hc_canonical_get_r7130', 'deep_hc_model_epilogue_get_r366',
             'deep_hc_fullpre_enabled_r377', 'modern_deep_c4_tile_get_r5920',
             'modern_deep_layout_get_r7052', 'deep_moe_tail_get_r7211']
    for name in names:
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = [], C.c_int
        assert fn() == 0, ('Frozen existing policy must remain OFF', name)
    return dict(HC=0, r366=0, fullpre=0, c4_tile=0, layout=0, tail_SUM=0)
