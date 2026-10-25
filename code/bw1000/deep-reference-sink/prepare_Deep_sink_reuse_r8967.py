import pathlib,json,hashlib,difflib,struct
from Deep_latent_maps_r8946 import maps,validate
r=pathlib.Path(__file__).resolve().parent;d=r/'Deep-reference-sink-private-r8967';d.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
old=(r/'deep_latent_reference_scratch_r8954.cpp').read_bytes();assert hashlib.sha256(old).hexdigest()=='025729ee3182efe0e169677ad1b7cf44e938349360c297554c3cb81947e2832b'
s=old.decode();s=s.replace('reference_scratch_counts_r8954{};}','reference_scratch_counts_r8954{};std::atomic<int> reference_sink_mode_r8967{0};std::array<std::array<std::atomic<uint64_t>,4>,8> reference_sink_counts_r8967{};}')
needle='  rt::Tensor logits_storage,probability_storage,result_storage;'
assert s.count(needle)==1
s=s.replace(needle,'  const bool reuse_sink= reuse&&reference_sink_mode_r8967.load()==1;\n  size_t last_seeded_n=std::numeric_limits<size_t>::max();uint64_t sink_copies=0,sink_skipped=0;\n'+needle)
needle='logits->narrow({{1,n,1}})->copy_from(r,sink->view({heads,1}));'
assert s.count(needle)==1
s=s.replace(needle,'if(!reuse_sink||n!=last_seeded_n){logits->narrow({{1,n,1}})->copy_from(r,sink->view({heads,1}));last_seeded_n=n;++sink_copies;}else{++sink_skipped;}')
needle='c[reuse?3:2].fetch_add(reuse?3:3*tokens);'
assert s.count(needle)==1
s=s.replace(needle,needle+'auto&sc=reference_sink_counts_r8967.at(r.device().index());sc[0].fetch_add(tokens);sc[1].fetch_add(sink_copies);sc[2].fetch_add(sink_skipped);if(reuse_sink)++sc[3];')
s+='\nextern "C" int deep_reference_sink_mode_r8967(int value){if(value!=0&&value!=1)return 1;infinilm::models::deepseek_v4::reference_sink_mode_r8967.store(value);return 0;}\n\nextern "C" int deep_reference_sink_counts_r8967(int d,uint64_t*out,size_t n){if(d<0||d>=8||!out||n!=4)return 1;for(size_t k=0;k<4;++k)out[k]=infinilm::models::deepseek_v4::reference_sink_counts_r8967[d][k].load();return 0;}\n'
dst=d/'deep_latent_reference_sink_r8967.cpp';dst.write_bytes(s.encode());diff=''.join(difflib.unified_diff(old.decode().splitlines(True),s.splitlines(True),fromfile='qualified923/deep_latent_reference_scratch_r8954.cpp',tofile=dst.name));(d/'proposal.diff').write_bytes(diff.encode())
# All five arithmetic submit statements must remain byte-for-byte unchanged.
ops=lambda t:[line[line.index('ex::submit<'):]for line in t.splitlines()if 'ex::submit<'in line]
assert ops(s)==ops(old.decode())
fixtures=[];validate()
for kind in ['swa','hca128','csa4']:
 si,ci,lens,cr,ratio=maps(kind);ns=[len(set(int(x)for x in si[t,:int(lens[t,0])]))+len(set(int(x)for x in ci[t,:int(lens[t,1])]))for t in range(len(lens))]
 assert ns==[int(x+y)for x,y in lens]
 copies=1+sum(x!=y for x,y in zip(ns,ns[1:]));skipped=len(ns)-copies
 # CPU storage oracle: simulate exact sink writes, Gemm's [0,n) writes and read-only Softmax.
 H=8;maxn=max(ns);a=[0.]*(H*(maxn+1));b=a.copy();sink=[-.5+h/7 for h in range(H)];last=None
 for t,n in enumerate(ns):
  for h in range(H):
   a[h*(n+1)+n]=sink[h]
   if last!=n:b[h*(n+1)+n]=sink[h]
   for k in range(n):a[h*(n+1)+k]=b[h*(n+1)+k]=float((t+3*h+k)%17-8)
  assert a[:H*(n+1)]==b[:H*(n+1)]
  last=n
 fixtures.append(dict(kind=kind,T=len(ns),heads=H,logical_n_SHA256=hashlib.sha256(b''.join(struct.pack('<Q',n)for n in ns)).hexdigest(),n_min=min(ns),n_max=max(ns),original_copies=len(ns),candidate_copies=copies,removed_copies=skipped,removed_percent=100*skipped/len(ns),per_owner_counters_OFF=[len(ns),len(ns),0,0],per_owner_counters_ON=[len(ns),copies,skipped,1],CPU_storage_boundary_oracle=True))
out=dict(hardware='BW1000',level='CPU_SOURCE_READY',status='NATIVE_BUILD_NOT_AUTHORIZED',CPU_only=True,new_native_build=0,new_GPU=0,parent_source_SHA256=hashlib.sha256(old).hexdigest(),parent_DSO_SHA256='0c164dd3c783d6975b361da9d630edb8b242499e206d4bf88c4a5f425f2a5b53',parent_receipt_path='results/deep-reference-scratch-CPU-r8954.json',parent_receipt_SHA256='092e50eafcd36058df8d28961b72c140419e08a8137d7b9ba77148ecf19687fd',source={'path':str(dst),'SHA256':sha(dst)},diff={'path':str(d/'proposal.diff'),'SHA256':sha(d/'proposal.diff')},independent_default_OFF='deep_reference_sink_mode_r8967',counter_CAPI='deep_reference_sink_counts_r8967(owner,uint64_t*,4)',counter_fields=['baseline_sink_copy_calls','actual_sink_copy_calls','skipped_sink_copy_calls','sink_reuse_prefill_calls'],counter_scope='Accumulate only tokens>64 after original successful syncStream; existing scratch counter ABI unchanged.',fixtures=fixtures,math_submit_lines_unchanged=True,native_compile_status='NOT_BUILT: source has no new SDK/API dependencies; needs exactly one replacement host TU plus original link when authorized.',contract=['Candidate only if scratch reuse is ON, tokens>64 and new independent sink mode ON; ordinary OFF and accepted T32 decode execute original copy path.','last_seeded_n is call-local; new n always reseeds, even if that n appeared earlier.','Same n preserves logits view shape [heads,n+1], same storage base and sink addresses h*(n+1)+n.','First GEMM output is logits narrow columns [0,n), physical row stride n+1; sink column n is outside its written matrix.','Softmax input logits is read-only; probabilities and result have separate stores; second GEMM and Cast never target logits.','Sink Tensor is held by caller and is not written by this function; same owner stream and original final/catch drain unchanged. Concurrent external writes are not an authorized input contract.','No cached Tensor view or descriptor was introduced.','Counts describe logical copy_from calls skipped, not measured device memcpy count or performance.'],limits=['CPU oracle checks address/write-set preservation, not GPU arithmetic.','No native compile or GPU/fullModel validation yet; no speedup claim.','Any component must compare accepted scratchON/sinkOFF vs scratchON/sinkON, leave8933 prefill tiles OFF and retain original reference Ops.'])
proof=r/'Deep-sink-copy-contract-r8967b.json';cp=json.loads(proof.read_bytes());copy=next(p for p in cp['files']if p['path'].endswith('/runtime/tensor/copy.cc'));out['source_contract_proof']={'copy_from':{'path':copy['path'],'SHA256':copy['SHA256'],'lines':'223-240 owner/shape/dtype validation;279-284 same-device GPU submits Ops::Copy(src,false,dst) and returns without drain','source_binding_limit':'Read current pinned Runtime candidate source; this card does not rebuild or rederive historical reused-object dependency receipts. Existing qualified923 header/Runtime entry remains unchanged.'},'softmax_readonly':{'source_SHA256':'e8fb72ec1eb228468c494fbb26ae44c445d34f19bd5304f4878abd888106fa12','contract':'CudaSoftmax passes input.data as const T* and out.data as T*; out is separate probability storage.'},'gemm_write_bounds':{'source_SHA256':'9d9e424fc6e890fe6dad857e44c44cc072b0e2f7bc9fa635023c1a5aa20012a4','contract':'BlasGemm uses original narrowed y m/n and ldy. For output narrow [heads,n] stride[n+1,1], last written index per head is h*(n+1)+(n-1); sink is h*(n+1)+n. No change to alpha,beta,transposes,compute type or provider.'},'evidence':{'path':str(proof),'SHA256':sha(proof)}}
p=r/'Deep-reference-sink-CPU-ready-r8967.json';p.write_bytes((json.dumps(out,ensure_ascii=False,indent=2)+'\n').encode());print(json.dumps({'ready':str(p),'bytes':p.stat().st_size,'SHA256':sha(p),'source_SHA256':sha(dst),'fixtures':fixtures}))
