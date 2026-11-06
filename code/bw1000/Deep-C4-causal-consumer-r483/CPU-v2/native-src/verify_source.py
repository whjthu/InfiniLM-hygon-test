"""Exact immutable parent and edit-scope verification, no compiler/Native/device."""
from pathlib import Path
import hashlib
import re
import json
H=lambda b:hashlib.sha256(b).hexdigest()

def verify(root=None):
 d=Path(root or Path(__file__).parent);parent=d.parent/'source-v1'
 old=(parent/'probe.cpp').read_text(encoding='utf-8');new=(d/'probe.cpp').read_text(encoding='utf-8')
 assert H((parent/'probe.cpp').read_bytes())=='9d32d84d99a6b0b005c8a2b7b46bee02d81a032004c957350a1fc5170a48a433'
 assert (d/'binding.py').read_bytes()==(parent/'binding.py').read_bytes()
 apis=lambda s:sorted(re.findall(r'extern "C" [^{;\n]*?(deep_c4_probe_\w+_r483)\(',s))
 assert len(apis(new))==13 and apis(new)==apis(old)
 normalized=new.replace('// Genuine C4 consumer v2: actual guarded input storage, original math and 13 C ABI unchanged.',
                        '// Source-only genuine C4 consumer. No fake RankInfo, cache seeding or bypassed compressor.')
 normalized=normalized.replace('#include "input_guard.hpp"\n','')
 normalized=normalized.replace(' size_t input_audits=0,input_buffers_checked=0,input_bytes_checked=0;double input_audit_seconds=0;\n','')
 start=normalized.index('struct Inputs {');end=normalized.index('\nJSON info',start)
 a=old.index('struct Inputs {');b=old.index('\nJSON info',a)
 normalized=normalized[:start]+old[a:b]+normalized[end:]
 normalized=normalized.replace('work.audit(c);','')
 normalized=normalized.replace('{"input_audits",c.input_audits},{"input_buffers_checked",c.input_buffers_checked},{"input_bytes_checked",c.input_bytes_checked},{"input_audit_seconds",c.input_audit_seconds},{"input_guard_bytes_each_side",256},','')
 assert normalized==old,'change escaped Inputs/metrics/two audit boundaries'
 assert new.count('work.audit(c);')==2
 assert 'stop-start).count();work.audit(c);c.prepared_tokens' in new
 assert 'c.r->syncStream();work.audit(c);c.loaded.insert(name);' in new
 return dict(passed=True,CABI_unchanged=13,binding_bytes_unchanged=True,Native_math_and_transaction_source_unchanged=True,
             new_probe_source_SHA256=H((d/'probe.cpp').read_bytes()),input_guard_SHA256=H((d/'input_guard.hpp').read_bytes()),
             target_Native_compile=False,GPU_GO=False,Model_GO=False)

if __name__=='__main__':print(json.dumps(verify(),indent=2))
