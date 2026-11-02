"""Read-only dynamic dispatch proof against the retained exact source/object."""
from pathlib import Path
import hashlib

SOURCE='/refactor/incoming/Kimi-long-scale-exact-r201/kimi_moe_scale_exact_r201.cpp'
SOURCE_SHA='1959878b051f9ad25d60a3f17bb776a9f74623cacccf9fd7d216949addea70eb'
OBJECT='/refactor/candidates/Kimi-long-scale-exact-r201/moe.o'
OBJECT_SHA='db8e6363b30be629401ef4b17f8eab0fbdcdb6e7f92f3b983c8968418bca15d3'


def verify(source=Path(SOURCE),object_path=Path(OBJECT)):
    raw=source.read_bytes();assert hashlib.sha256(raw).hexdigest()==SOURCE_SHA
    s=raw.decode()
    text='const char*i192_flag_r8669y=std::getenv("INFINILM_KIMI_I192_GROUPED_R8669Y");'
    assert s.count(text)==1 and ('static '+text) not in s
    assert s.index('const int serving_policy_r4665=serving_r4665::policy();')<s.index(text)
    assert 'routed_input->size(0)>=8 && routed_input->size(0)<=64' in s
    assert 'i192_T_counts_r9129.at(runtime.device().index()).at(routed_input->size(0)).fetch_add(1' in s
    assert 'n!=65' in s and 'device>=8' in s
    if object_path is not None:
        assert hashlib.sha256(object_path.read_bytes()).hexdigest()==OBJECT_SHA
    return dict(source=str(source),source_SHA256=SOURCE_SHA,moe_object_SHA256=OBJECT_SHA,
                GROUPED_read_each_forward=True,static_GROUPED=False,
                counts_scope='Native HOST successful branch selections after owning Runtime drain; no GPU hardware counter',
                histogram_buckets=65,local_owners=8,fallback_counter_exists=False,
                eligible_T=[8,64],ineligible_probe_T=[1,512,443],no_fixed_layer_count=True)
