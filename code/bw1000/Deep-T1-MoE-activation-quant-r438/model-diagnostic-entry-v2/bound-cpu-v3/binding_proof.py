"""Read-only derivation from actual902 Model chain to the new two-HOST DSO."""
import copy
import hashlib
import json
from pathlib import Path
import re

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
OLD_MODEL='e3e39de9524af7c19f6bf757236e3288c4bd7647909b2ad14f3e3ebf6d1426e9'
MODE_NAMES=['deep_model_actquant_'+n+'_r438'for n in ['set','clear','read','metadata']]
CAPTURE_NAMES=['deep_model_capture_'+n+'_r438v2'for n in ['arm','seal','clear','abandon','count','record','tensor','copy']]


def validate_pin(pin, *, require_ready):
    assert pin['parent_library_SHA256']==OLD_MODEL
    assert pin['Native_object_count']==49 and pin['reused_parent_objects']==47 and pin['new_HOST_TUs']==2
    if not pin['CPU_READY']:
        assert pin['CPU_receipt_SHA256']is None and pin['library_SHA256']is None and pin['actual_CPU']is None
        if require_ready:raise RuntimeError('New diagnostic Native CPU artifact is not yet bound; cannot load')
        return False
    assert all(re.fullmatch('[0-9a-f]{64}',pin[k]or'')for k in ['CPU_receipt_SHA256','library_SHA256','source_manifest_SHA256'])
    assert re.fullmatch(r'910407\.\d+',pin['actual_CPU'])
    return True


def derive(pin):
    validate_pin(pin,require_ready=True)
    # The actual public1374 Model binder already validates afc6/latest worker,
    # old902 graph/deps/SDK, and the original full32K guard. No CDLL in binding().
    from public_deep_model.native_binding import binding as original_binding
    _,old_receipt,old_normal=original_binding()
    assert old_receipt['library_SHA256']==OLD_MODEL
    assert sha(pin['CPU_receipt'])==pin['CPU_receipt_SHA256']
    cpu=json.loads(Path(pin['CPU_receipt']).read_bytes())
    assert cpu['completed']and cpu['passed']and cpu['actual_CPU']==pin['actual_CPU']and cpu['CPU_only']
    assert not any(cpu[k]for k in ['Native_dlopen','Runtime_created','weights_payload_opened','GPU_initialized','SDK_initialized','Model_loaded'])
    assert cpu['source_manifest_SHA256']==pin['source_manifest_SHA256']
    assert cpu['library']==pin['library']and cpu['library_SHA256']==pin['library_SHA256']and sha(cpu['library'])==pin['library_SHA256']
    assert cpu['Native_object_count']==49 and cpu['production_HOST_compiles']==2 and cpu['GPU_compiles']==0
    assert cpu['unchanged_actual_Model_objects']==46 and cpu['all_old_strong_exports_retained']and cpu['all_old_C_exports_retained']and cpu['DT_NEEDED_unchanged']
    assert cpu['default_unarmed_OFF']and cpu['capture_default_OFF']and not cpu['old_CPU902_904_suites_replayed']
    check=cpu['actual_changed_save_consumer']
    assert check['completed']and check['passed']and check['GPU']==0 and not check['actual_Native_execution']and not check['actual_Model_activation_captured']
    before=old_normal['link']['full']['objects'];after=cpu['all_objects_SHA256']
    assert len(before)==len(after)==49
    same={p:h for p,h in before.items()if after.get(p)==h};assert len(same)==47
    old_model_cpu=json.loads(Path(pin['parent_CPU_receipt']).read_bytes())
    assert sha(pin['parent_CPU_receipt'])==pin['parent_CPU_receipt_SHA256']
    # Old902 also compiled a standalone pure-CPU scope.o, which is NOT one
    # of the49 linked Model objects. Only its two linked HOST units replace.
    old_host={item['object']for item in old_model_cpu['compiled']if item['object']in before}
    assert len(old_host)==2 and set(before)-set(after)==old_host
    assert len(cpu['compiled'])==2 and set(after)-set(before)=={v['object']for v in cpu['compiled']}
    for path,digest in after.items():assert sha(path)==digest,path
    for unit in cpu['compiled']:
        assert sha(unit['MD'])==unit['MD_SHA256']and unit['fresh_M_equals_MD']
        for path,digest in unit['dependencies_SHA256'].items():assert sha(path)==digest,path
    assert set(MODE_NAMES+CAPTURE_NAMES)<=set(cpu['new_CABI'])
    receipt=copy.deepcopy(old_receipt)
    receipt.update(library=pin['library'],library_SHA256=pin['library_SHA256'],
        diagnostic_derivation_receipt_SHA256=pin['CPU_receipt_SHA256'],actual_Model_unexecuted_CPU_only=True)
    return receipt


def verify_diagnostic_functions(lib,pin):
    """Actual device-stage dladdr, after the original complete declaration."""
    import ctypes as C
    class Info(C.Structure):_fields_=[('fname',C.c_char_p),('base',C.c_void_p),('sname',C.c_char_p),('saddr',C.c_void_p)]
    dl=C.CDLL(None).dladdr;dl.argtypes=[C.c_void_p,C.POINTER(Info)];dl.restype=C.c_int
    expected=Path(pin['library']).resolve();actual={}
    for name in MODE_NAMES+CAPTURE_NAMES:
        info=Info();assert dl(C.cast(getattr(lib,name),C.c_void_p),C.byref(info))
        p=Path(info.fname.decode()).resolve();assert p==expected and sha(p)==pin['library_SHA256']
        actual[name]=str(p)
    metadata=json.loads(lib.deep_model_actquant_metadata_r438())
    assert metadata['owners']==8 and metadata['counter_width']==12 and metadata['default_mode']==0 and not metadata['default_armed']
    assert metadata['HostReturned_is_not_GPU_completed']and metadata['whole_engine_quiescence_required_for_set_clear']
    return dict(providers=actual,mode_metadata=metadata)
