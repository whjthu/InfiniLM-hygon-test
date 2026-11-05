"""Exact two-header/one-caller derivation; no checkpoint or Model execution."""
from hashlib import sha256
CALLER_SHA="d54b80d68497c26c83a0c63fe87fbec376f4b106daa40d0da61238dfb5914be2"
SELECTED_SHA="50ece2ded22ce1e70ea55947972dd3f5f17cdee2dd39f9080f1bb08a3cae0489"
def replace(s,old,new):
    if s.count(old)!=1: raise ValueError("exact caller anchor differs: "+old[:100])
    return s.replace(old,new,1)
def caller(raw):
    if sha256(raw).hexdigest()!=CALLER_SHA: raise ValueError("actual r403 caller SHA differs")
    s=raw.decode();nl="\r\n" if b"\r\n" in raw else "\n"
    edits=[
      ('#include "glm_registry_geometry_r184b.hpp"','#include "glm_registry_geometry_r184b.hpp"\n#include "component_scope.hpp"'),
      ('registry_geometry_r184b::member(rank.tp_size,rank.pp_size,rank.pp_stage,layer);','layer0_component::member(r,rank,layer,rows,capacity,topk);'),
      ('auto features=indexer_->forward(r,hidden,prepared.q_resid,positions);','auto features=indexer_->forward(r,hidden,prepared.q_resid,positions);layer0_component::keep("indexer.query",features.query);layer0_component::keep("indexer.key",features.key);layer0_component::keep("indexer.head_weights",features.head_weights);'),
      ('for(auto&result:results){work.held.insert(work.held.end(),{result.scores,result.indices});selection.push_back(result.indices);}', 'for(auto&result:results){layer0_component::keep("selector.masked_F32_scores",result.scores);work.held.insert(work.held.end(),{result.scores,result.indices});selection.push_back(result.indices);}'),
      ('auto fresh_all=prepare_->expand_cached(r,prepared.kv_latent,prepared.k_rot);','auto fresh_all=prepare_->expand_cached(r,prepared.kv_latent,prepared.k_rot);layer0_component::keep("fresh.K",fresh_all.first);layer0_component::keep("fresh.V",fresh_all.second);'),
    ]
    for a,b in edits:s=replace(s,a,b.replace("\n",nl))
    return s.encode()
def selected(raw):
    if sha256(raw).hexdigest()!=SELECTED_SHA: raise ValueError("actual selected-values header SHA differs")
    s=raw.decode();nl="\r\n"if b"\r\n"in raw else"\n"
    s=replace(s,'#include "chunk_geometry_r400.hpp"','#include "chunk_geometry_r400.hpp"'+nl+'#include "component_scope.hpp"')
    s=replace(s,'if(U<=max_block_rows){auto result=prepare.expand_cached(r,l,k->view({U,1,64}));','if(U<=max_block_rows){auto result=prepare.expand_cached(r,l,k->view({U,1,64}));layer0_component::keep("NoV.union_ids",ids);layer0_component::keep("NoV.union_V",result.second);')
    return s.encode()
