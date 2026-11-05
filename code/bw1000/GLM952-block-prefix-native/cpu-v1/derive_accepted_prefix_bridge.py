"""Derive the private staged-prefix C ABI from the exact GLM952 bridge."""

import argparse
from hashlib import sha256
from pathlib import Path

PARENT_SHA = "2dd4fe50c3476e6bf0d4dbd78ff9885f1f9862ed5174e64a20580f89c2d19fd2"
BRIDGE = r"""
namespace {
rt::Runtime& prefix_owner(Context*c) {
 if(!c||!c->owner||!c->module||!c->ready||c->pp!=2||c->rows_r5688!=1||c->capacity_r1406!=32768)
  throw std::invalid_argument("GLM staged prefix requires a loaded C1 TP8 PP2 context");
 auto&r=*c->owner;r.assertOwnerThread();int device=-1;
 if(cudaGetDevice(&device)!=cudaSuccess||device!=r.device().index()||r.isGraphRecording())
  throw std::invalid_argument("GLM staged prefix requires original owner/device/eager");
 cudaStreamCaptureStatus capture;
 if(cudaStreamIsCapturing(static_cast<cudaStream_t>(r.stream()),&capture)!=cudaSuccess||capture!=cudaStreamCaptureStatusNone)
  throw std::invalid_argument("GLM staged prefix requires an uncaptured stream");
 return r;
}
}
extern "C" int modern_glm_block_stage_v1(void*raw,uint64_t sequence,size_t past,size_t width,
 const void*input,const void*positions,void*logits,void*stream) {
 auto*c=static_cast<Context*>(raw);
 try{
  prefix_owner(c);
  if(!input||!positions||(c->stage+1==c->pp&&!logits)||!c->prefix_block.can_stage(sequence,past,width))
   throw std::invalid_argument("GLM staged prefix geometry or sequence differs");
  const size_t request[]={0,past,width};
  const int rc=modern_glm_model_forward(raw,width,1,request,input,positions,logits,1,stream);
  c->prefix_block.record_forward(sequence,past,width,rc==0);
  return rc;
 }catch(const std::exception&e){error=e.what();return 1;}
}
extern "C" int modern_glm_block_prepare_v1(void*raw,uint64_t sequence,size_t past,size_t width,size_t accepted) {
 auto*c=static_cast<Context*>(raw);
 try{
  auto&r=prefix_owner(c);const model::block_verify::StagedPrefix p{past,width,accepted};
  if(!c->prefix_block.can_prepare(sequence,p)||!model::block_verify::NativeLengthsAccess::apply(*c->module,r,p,false))
   throw std::invalid_argument("GLM staged prefix preflight differs");
  c->prefix_block.prepared(accepted);
  return 0;
 }catch(const std::exception&e){error=e.what();return 1;}
}
extern "C" int modern_glm_block_publish_v1(void*raw,uint64_t sequence,size_t past,size_t width,size_t accepted) {
 auto*c=static_cast<Context*>(raw);
 try{
  auto&r=prefix_owner(c);const model::block_verify::StagedPrefix p{past,width,accepted};
  if(!c->prefix_block.take_publish(sequence,p)||!model::block_verify::NativeLengthsAccess::apply(*c->module,r,p,true))
   throw std::invalid_argument("GLM staged prefix publish differs");
  c->prefix_block.published();
  return 0;
 }catch(const std::exception&e){error=e.what();return 1;}
}
"""


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("unexpected exact GLM bridge source anchor")
    return source.replace(old, new, 1)


def derive(raw):
    if sha256(raw).hexdigest() != PARENT_SHA:
        raise ValueError("unexpected original GLM bridge bytes")
    newline = "\r\n" if b"\r\n" in raw else "\n"
    source = raw.decode("utf-8")
    for old, new in (
        ('#include "glm_moe_dsa_projected_text_model_r22.hpp"',
         '#include "glm_moe_dsa_projected_text_model_r22.hpp"\n'
         '#include "accepted_prefix_native.hpp"\n#include "accepted_prefix_ticket.hpp"'),
        ("size_t capacity_r1406=0;};",
         "size_t capacity_r1406=0;model::block_verify::PrefixTicket prefix_block;};"),
        (' if(!c->ready||!count||count>128||!requests)throw std::invalid_argument("GLM full-model CAPI metadata differs");producer(stream);auto&r=*c->owner;',
         ' if(!c||!c->owner||!c->ready||!count||count>128||!requests)throw std::invalid_argument("GLM full-model CAPI metadata differs");auto&r=*c->owner;try{r.assertOwnerThread();}catch(const std::exception&e){error=e.what();return 1;}'
         'if(c->prefix_block.unresolved()){error="GLM staged prefix requires publish or full reset";return 1;}producer(stream);'),
        ('extern "C" int modern_glm_model_control(',
         BRIDGE + 'extern "C" int modern_glm_model_control('),
        ("if(c->transient_scratch_r5688)c->transient_scratch_r5688->reset();return 0;",
         "if(c->transient_scratch_r5688)c->transient_scratch_r5688->reset();c->prefix_block.reset_completed();return 0;"),
    ):
        source = replace_once(source, old, new.replace("\n", newline))
    return source.encode("utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = derive(args.parent.read_bytes())
    with args.output.open("xb") as file:
        file.write(output)
