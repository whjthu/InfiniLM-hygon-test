"""Read saved source bytes and passive ELF receipts; no remote/Native/compiler calls."""
import json,hashlib,argparse
from pathlib import Path
PINS={
 'Deep483-original-real-rank-comm-and-C4-contract-private.json':'5bc5f25f0e406f197e79e42b129623d35828b1f1200c88d322e328e49072b4b7',
 'Deep483-original-distconfig-topology-factory-private.json':'63a6fc8d4987ce970b520f48c768db7e244d31d5c6cd8d10611633800e83fc8b',
 'Deep483-original-qualified-DSO-factory-symbols-private.json':'b6c93418a9ce1cd196ae5ad38d2cb24496a4a22b4bc2f51b2570a7c1c1aed26e',
}
def H(b):return hashlib.sha256(b).hexdigest()
def main(L,G):
 data={}
 for name,h in PINS.items():
  b=(L/name).read_bytes();assert H(b)==h;data[name]=json.loads(b)
 sources={}
 for d in data.values():
  for s in d.get('sources',[]):
   b=s['text'].encode();assert H(b)==s['SHA256'] and len(b)==s['bytes'];sources[s['path']]=s
 d=data['Deep483-original-qualified-DSO-factory-symbols-private.json'];s=d['source'];assert H(s['text'].encode())==s['SHA256']=='1056d753a7c5be68ef493b547062eea63a48bb218d710b2616aeac60ca379cca';sources[s['path']]=s
 factory=next(v for p,v in sources.items()if p.endswith('communication_group.cpp'))
 assert factory['SHA256']=='191604f39ba256ace7d2d7156aeaaba4f497cc26d0b05c05993f94b2a5fa4648'
 f=factory['text'];a=f.index('CommunicationGroup::CommunicationGroup');tail=f[a:]
 assert 'if (tp_size > 1)'in tail and 'if (dist_config_.pp_size > 1)'in tail and 'RankInfo CommunicationGroup::get_rank_info(int rank) const'in tail
 project={x['path']:x for x in d['DSOs']};parent=next(x for x in d['DSOs']if x['SHA256']=='6e1a7e7643f84a52ab147cfc3c55eee4816affc6898109dbef31d559feaa86d1');native=next(x for x in d['DSOs']if x['SHA256']=='90ba080a7666c35e4160003412be2f0c127b176eba1fc6c5fa0f1eb21ea244b1')
 defined=lambda lib:[s['name']for s in lib['symbols']if s['section']and s['binding']==1]
 ps=defined(parent);ns=defined(native)
 factory_methods={name:next(s for s in ps if token in s)for name,token in {
 'construct':'CommunicationGroupC1E','get_rank_info':'CommunicationGroup13get_rank_infoEi','destroy':'CommunicationGroupD1E','DistConfig_devices':'DistConfigC1ERKSt6vectorIi'}.items()}
 c4_methods={name:next(s for s in ns if token in s)for name,token in {'construct':'C4IndexerC1E','prepare':'C4Indexer7prepareE','commit':'C4Indexer6commitE','reset':'C4Indexer5resetE','postload':'C4Indexer29process_weights_after_loadingE'}.items()}
 src=G/'code/bw1000/Deep-C4-constant-slot-metadata-r482/Native-CPU-v1/native-src/deep_c4_constant_slot_metadata_r482.cpp';b=src.read_bytes();assert H(b)=='d7d035960c9c05c9b849417fe743f300901503def01426e900d230e7dd19a5da';t=b.decode()
 assert '(tp_!=1&&tp_!=8)'in t and 'heads_=64/size_t(tp_)'in t and 'request.past!=lengths_[request.row]'in t
 assert 'pending.compression=compressor_->prepare_pooled' in t and 'compressor_->commit(r,pending.compression)'in t and 'pending.revision!=revision_'in t
 assert 'std::vector<int64_t>slots;for(size_t i=0;i<before;++i)slots.push_back' in t
 probe=(Path(__file__).parent/'probe.cpp').read_text()
 assert 'c->group->get_rank_info(0)' in probe and 'ds::DistConfig(std::vector<int>{0})'in probe
 assert 'c.index->load_parameter' in probe and 'c.loaded!=c.required'in probe and 'c.index->prepare'in probe and 'c.index->commit'in probe
 assert 'RankInfo rank{'not in probe and 'RankInfo rank;'not in probe
 return {'schema':1,'passed_source_and_passive_ELF_only':True,'private_inputs_SHA256':PINS,'sources':[{k:s[k]for k in ['path','SHA256','bytes']}for s in sources.values()],
 'provider_DSOs':[{k:x[k]for k in ['path','SHA256','bytes']}for x in d['DSOs']],
 'factory_defined_provider':parent['path'],'factory_methods':factory_methods,'actual_Native_C4_methods':c4_methods,
 'factory_source_cpp_historical_build_pin_known':False,
 'factory_source_qualification':'Current read-only source first observed; header matches historical graph and original DSO has exports. Source-to-original factory object provenance and actual runtime topology remain unproved. Do not infer device behavior solely from this source.',
 'source_topology':{'C4_component_accepts_TP':[1,8],'component_TP1_local_heads':64,'Model_requires_TP':8,'Model_TP8_local_heads':8,'TP1_PP1_source_skips_communication_init':True,'device_validation_and_DeviceGuard_remain':True},
 'causal_contract':['Genuine framework group and get_rank_info(0); no manually populated RankInfo.','Required checkpoint tree must be enumerated and each exact key loaded once through ModelWeights.load_parameter before process_weights_after_loading.','Full32768 capacity/8192 compressed slots, actual TemporalCompressor.prepare_pooled and C4Indexer prepare/commit.','Prepare past must equal committed lengths; logical positions are exact causal sequence; commit validates owner/revision and poisons after failure.','Explicit abort releases Pending after drain without wrapper past advance; raw cache/compressor state invariance still requires actual device tests.'],
 'metadata_source_counts':{'C4_count':21,'past':8496,'compressed_history':2124,'INT64_upload_bytes_per_C4':2124*8,'all21_logical_upload_bytes':21*2124*8,'constant_table_payload_per_C4_bytes':8192*8,'all21_constant_table_payload_bytes':21*8192*8,'C4_host_slots_vector_construction_still_present':True},
 'next_required':['Exact new oneHOST fresh-M/source/quote-origin closure against1113 newclass/69libs plus original factory provider ABI and Weight APIs; publish separate bounded CPUclaim before compiler.','Bind real checkpoint/config/layer/prefix and every required weight/input bytesSHA; no synthetic weights described as checkpoint or repeating r390 old score fixture.','One physical TP1 component may establish compressor/causal-state correctness and cost, not TP8 communications or complete causal attention/Model quality. Separate fresh deviceclaim only after CPU qualification.'],
 'new_compile':0,'new_Slurm':0,'Native_dlopen':False,'Runtime_created':False,'GPU_GO':False,'Model_GO':False,'numeric_qualified':False,'performance_qualified':False}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--local-parent',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);a=p.parse_args();print(json.dumps(main(a.local_parent,a.repo),ensure_ascii=False,indent=2))
