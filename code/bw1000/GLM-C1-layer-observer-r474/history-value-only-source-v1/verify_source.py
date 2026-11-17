"""Source structure only; no compile, SDK, device or performance qualification."""
import hashlib,json,sys
from pathlib import Path
D=Path(__file__).resolve().parent
H=lambda b:hashlib.sha256(b).hexdigest()

def check(files):
    old={n:(D/('PARENT_'+n)).read_text() for n in ('glm_moe_dsa_attention_prepare.hpp','prepare_tp16.cpp','NoV_tp16.cpp','selected_values_r400.hpp')}
    declaration=' runtime::Tensor expand_cached_values(runtime::Runtime&,const runtime::Tensor&latent,const runtime::Tensor&rotary_key)const;\n'
    assert files['glm_moe_dsa_attention_prepare.hpp'].count(declaration)==1
    assert files['glm_moe_dsa_attention_prepare.hpp'].replace(declaration,'')==old['glm_moe_dsa_attention_prepare.hpp']
    p=old['prepare_tp16.cpp'];start=p.index('std::pair<rt::Tensor,rt::Tensor> AttentionPrepare::expand_cached(');end=p.rfind('\n}');original=p[start:end]
    new=original.replace('std::pair<rt::Tensor,rt::Tensor> AttentionPrepare::expand_cached(', 'rt::Tensor AttentionPrepare::expand_cached_values(')
    removed='auto key=work.alloc({tokens,heads_,256},rt::DataType::kBFloat16);key->narrow({{2,0,192}})->copy_from(r,expanded->narrow({{2,0,192}}));\n for(size_t head=0;head<heads_;++head)key->narrow({{1,head,1},{2,192,64}})->copy_from(r,rotary);'
    assert removed in new
    new=new.replace(removed,'// Value-only: full448 projection and V256 copy unchanged; unconsumed K materialization omitted.').replace('return {key,value};','return value;')
    assert files['prepare_tp16.cpp']==p[:end]+'\n'+new+p[end:]
    nov=files['NoV_tp16.cpp'];begin=nov.index('// Independent default-OFF candidate;');end=nov.index('return v_on_demand_r400::selected_values',begin);branch=nov[begin:end]
    expected='''// Independent default-OFF candidate; observer state never controls mathematics.
if(v398&&complete_prefix&&tokens==1&&requests.size()==1&&count==1&&request.past>0
   &&logical.size()<=v_on_demand_r400::max_block_rows&&prepare_->local_heads()==4){
 const char* value_only=std::getenv("GLM_HISTORY_VALUE_ONLY_V1");
 if(value_only&&value_only[0]=='1'&&value_only[1]=='\\0')
  return history_value_only_v1::selected_values(*prepare_,r,latent_row,rotary_row,ids,work.held);
}
'''
    assert branch==expected
    nov=nov[:begin]+nov[end:]
    nov=nov.replace('#include "selected_values_r400.hpp"\n#include "history_value_only_v1.hpp"\n#include <cstdlib>','#include "selected_values_r400.hpp"')
    assert nov==old['NoV_tp16.cpp']
    helper=files['history_value_only_v1.hpp'];oldhelper=old['selected_values_r400.hpp']
    begin=oldhelper.index(' size_t U=ids->size(0);');end=oldhelper.index(' if(U<=max_block_rows)')
    assert oldhelper[begin:end] in helper
    assert oldhelper[oldhelper.index(' if(!latent'):begin] in helper
    assert 'auto value=prepare.expand_cached_values(r,l,k->view({U,1,64}));' in helper
    assert '++v_on_demand_r400::completed_expand_calls;held.push_back(value);return value;' in helper
    assert 'if(ids->size(0)>v_on_demand_r400::max_block_rows)throw' in helper
    return ['old_class_fields_methods_unchanged','only_unused_K_removed','full_projection_guard_V_sync_Work_unchanged','NoV_only_defaultOFF_history_branch','two_original_gathers_preserved','counter_value_lifetime_preserved']

def main():
    for n,p in json.loads((D/'PARENT_PINS.json').read_bytes()).items():
        b=(D/('PARENT_'+n)).read_bytes();assert len(b)==p['bytes'] and H(b)==p['SHA256']
    names=('glm_moe_dsa_attention_prepare.hpp','prepare_tp16.cpp','NoV_tp16.cpp','history_value_only_v1.hpp');files={n:(D/n).read_text() for n in names};normal=check(files);rejected=[]
    for name,file,a,b in [('drop_sync','prepare_tp16.cpp','r.syncStream();',';'),('change_full_projection','prepare_tp16.cpp','heads_,448','heads_,256'),('change_class_field','glm_moe_dsa_attention_prepare.hpp','bool ready_=false','bool ready_=true'),('weaken_input_guard','prepare_tp16.cpp','!latent->is_contiguous()','false'),('enable_default','NoV_tp16.cpp',"value_only[0]=='1'","value_only[0]!='0'"),('remove_rotary_gather','history_value_only_v1.hpp','rotary,ids,int64_t{0},k','latent,ids,int64_t{0},k')]:
        altered=dict(files);assert a in altered[file];altered[file]=altered[file].replace(a,b)
        try:check(altered)
        except AssertionError:rejected.append(name)
        else:raise AssertionError('mutation escaped: '+name)
    result=dict(passed=True,normal=normal,rejected=rejected,source_only=True,compile=False,remote=False,GPU=False,performance_evidence=False)
    out=Path(sys.argv[1]);out.parent.mkdir(parents=True,exist_ok=False);out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))
if __name__=='__main__':main()
