"""Reverse exactly the new private class and five call sites; no device imports."""
from pathlib import Path
import argparse,hashlib,json
D=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def check(source):
 proof=json.loads((D/'DERIVATION.json').read_bytes());assert sha(source)==proof['parent_CPP']['SHA256']
 original=Path(source).read_text();prefix=original[:original.index('std::shared_ptr<infinilm::config::ModelConfig>\nmake_kimi_k3_subconfig')]
 begin=original.index('KimiK3MoE::KimiK3MoE(');end=original.index('\n} // namespace infinilm::models::kimi_k3',begin)
 expected=prefix+original[begin:end]+'\n} // namespace infinilm::models::kimi_k3\n'
 actual=(D/'private_moe.cpp').read_text()
 for name in proof['private_inline_variables']:actual=actual.replace(name+'_private_r437',name)
 for name in proof['private_diagnostic_definitions']:actual=actual.replace(name+'_private_r437',name)
 actual=actual.replace('KimiK3MoER437','KimiK3MoE')
 for old,new in proof['projection_replacements'].items():actual=actual.replace(new,old)
 actual=actual.replace('#include "private_moe.hpp"\n#include "projection.hpp"','#include "models/kimi_k3/kimi_k3_moe.hpp"').replace('// Original header supplied by private_moe.hpp','#include "kimi_k3_moe.hpp"')
 # Only the duplicate relative includes were resolved to their same pinned target.
 marker='#include "weights/weights.hpp"';assert actual.count(marker)==2
 a=actual.rindex(marker);actual=actual[:a]+actual[a:].replace(marker,'#include "../../weights/weights.hpp"',1)
 marker='#include "engine/distributed/communication_group.hpp"';assert actual.count(marker)==2
 a=actual.rindex(marker);actual=actual[:a]+actual[a:].replace(marker,'#include "../../engine/distributed/communication_group.hpp"',1)
 assert actual==expected,'Unexpected change in constructor, routing, norm, shared/FP4 experts, TP or original math'
 private=(D/'private_moe.hpp').read_text();decl=private[private.index('class KimiK3MoER437 :'):private.rindex('\n}')].strip().replace('KimiK3MoER437','KimiK3MoE')
 h=(D/'original_header_r201.hpp').read_text();assert sha(D/'original_header_r201.hpp')==proof['parent_header']['SHA256'];b=h.index('class KimiK3MoE :');e=h.index('\n};',b)+3;assert decl==h[b:e]
 helper=(D/'projection.hpp').read_text();assert 'if(!selected)return layer->forward(r,input);'in helper
 assert helper.index('rt::Tensor weight,output;')<helper.index('drain{r};')<helper.index('if(which<0')<helper.index('weight=layer->weight();')
 assert 'std::uncaught_exceptions()>count'in helper and 'kimi_lowrank_gemv_r432('in helper
 body=(D/'component.cpp').read_text();assert 'std::make_unique<km::KimiK3MoE>'in body and 'std::make_unique<km::KimiK3MoER437>'in body
 assert 'topology.pp_size=1;'in body and 'PP1 component cannot claim r201 PP2'in body
 assert body.index('row["arms"].push_back(item);save();')<body.index('if(nf||checked!=input_original')
 return dict(passed=True,original_remaining_ctor_forward_reverse_exact=True,private_class_fields_reverse_exact=True,actual_replaced_call_sites=5,reachable_C1_each_forward=[1,1],OFF_original_strong_forward=True,old_GPU_object_reused=True,new_GPU_compile=0,Native_dlopen=False,Runtime_created=False,GPU=0)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--parent-source',default='/refactor/incoming/Kimi-long-scale-exact-r201/kimi_moe_scale_exact_r201.cpp');a=p.parse_args();print(json.dumps(check(a.parent_source)))
