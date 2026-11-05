"""One exact HC TU diagnostic derivation; strip instrumentation -> original bytes."""
from pathlib import Path
import argparse,hashlib,json,re
PIN='24348584d9684eee33099ea31f22bb0f0e10b1f31bdc64c00dcba46f2fde2d09'
def derive(b):
 assert hashlib.sha256(b).hexdigest()==PIN;s=b.decode();original=s
 def add(anchor,text,before=False):
  nonlocal s
  assert s.count(anchor)==1,anchor
  tagged='/*R467_BEGIN*/'+text+'/*R467_END*/';s=s.replace(anchor,tagged+anchor if before else anchor+tagged)
 add('/*HC377_END_BRANCH*/','hc_pre_split_r467::Frame hc_profile(r,t);')
 add(' if(device_.index()<0||device_.index()>=8)throw std::invalid_argument("HC canonical owner device outside TP8 lease");','hc_profile.mark(1);',True)
 add('/*HC366_BEGIN_T1*/','hc_profile.mark(2);',True)
 add(' auto normalized_comb=work.alloc({t,4,4});','hc_profile.mark(3);',True)
 add(' comb=normalized_comb;','hc_profile.mark(4);')
 add('residual->dtype());r.syncStream();return {collapsed,post,comb};','') if False else None
 anchor='residual->dtype());r.syncStream();return {collapsed,post,comb};';assert s.count(anchor)==1
 s=s.replace(anchor,'residual->dtype());/*R467_BEGIN*/hc_profile.mark(5);/*R467_END*/r.syncStream();/*R467_BEGIN*/hc_profile.after_original_drain();/*R467_END*/return {collapsed,post,comb};')
 export='\nextern "C" const char* deep_hc_pre_split_read_r467() noexcept {static thread_local std::string out;try{out=hc_pre_split_r467::read();return out.c_str();}catch(...){return nullptr;}}\n'
 s='/*R467_BEGIN*/#include "hc_pre_split.hpp"\n/*R467_END*/'+s+'/*R467_BEGIN*/'+export+'/*R467_END*/'
 assert re.sub(r'/\*R467_BEGIN\*/.*?/\*R467_END\*/','',s,flags=re.S).encode()==b
 assert s.count('r.syncStream()')==original.count('r.syncStream()')
 return s.encode()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('original');p.add_argument('output');a=p.parse_args();b=derive(Path(a.original).read_bytes());Path(a.output).write_bytes(b);print(json.dumps(dict(SHA256=hashlib.sha256(b).hexdigest(),no_new_sync=True,strip_equals_original=True)))
