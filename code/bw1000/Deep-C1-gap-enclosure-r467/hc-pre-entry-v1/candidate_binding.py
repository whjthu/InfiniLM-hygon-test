"""Exact actualCPU1033 one-HOST49reuse derivation, no arbitraryprovider fallback."""
from pathlib import Path
import hashlib,json,ctypes
NATIVE='5fd76082aa15f039515728c16bcb320b3b1c245952843e8f93f2038d17c5e0a6'
LIBRARY=Path('/refactor/candidates/Deep-HCpre-phase-r467-CPU-v2/libDeep_HCpre_phase_r467.so')
RECEIPT='97702f1120ebf6fdd7d56c01d58dfc1be4ab03b7b3177ea3c496b33fff3b5a00'
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def receipt_from_worker(worker):
 _,parent,normalized=worker.verify_native_chain();assert parent['library_SHA256']=='a314ae6891977316189f875b793ed3f2e2eb0898110ad0cc382d2569eaf31dc1'
 p=LIBRARY.with_name('CPU-build.json');assert H(p)==RECEIPT;q=json.loads(p.read_bytes());assert q['passed']and q['completed']and q['actual_CPU']=='910407.1033'and q['parent_7f_SHA256']=='7fcea960c5f218234ed8a4733fa7bd94b31055ff7396755c2afd6f5bba502f90'
 assert q['retained_count']==49 and q['new_HOST']==1 and q['new_GPU']==0 and q['old_strong_exports_retained']and q['SDK_unchanged']and q['no_new_sync']and q['strip_equals_original']and q['fresh_M_equals_MD']
 assert not any(q[k]for k in['GPU','Native_dlopen','Runtime','weights','Model']);assert q['library']==str(LIBRARY)and q['library_SHA256']==NATIVE and H(LIBRARY)==NATIVE
 for n,h in q['retained_objects'].items():assert H(n)==h,n
 for n,h in q['dependency_SHA256'].items():assert H(n)==h,n
 assert H(p.with_name('HC.o'))==q['object_SHA256']
 return dict(parent,library=str(LIBRARY),library_SHA256=NATIVE,HCpre_actualCPU='910407.1033',HCpre_receipt_SHA256=RECEIPT,actual_DTK_bindings=parent['actual_DTK_bindings'])
def bind_library(path,expected):
 from public_deep_model.checkpoint_loader import declare
 assert Path(path)==LIBRARY and expected==NATIVE and H(path)==NATIVE;return declare(ctypes.CDLL(str(path)))
