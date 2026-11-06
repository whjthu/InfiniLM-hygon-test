"""Source-only bounded K-outer reuse cost; no Native/SDK/compiler/device calls."""
from pathlib import Path
import argparse,hashlib,json
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
Q=32;K=3072;CONTEXT=32768

def tiles(queries,past):
 assert 1<=queries<=512 and 0<=past and past+queries<=CONTEXT
 first=0;out=[]
 while first<queries:
  rows=min(Q,queries-first,K-(past+first)%K);out.append((first,rows,past+first+rows));first+=rows
 return out

def case(queries,past):
 ts=tiles(queries,past);total=past+queries
 old=[[(start,min(K,end-start),start+min(K,end-start)==end)for start in range(0,end,K)]for first,rows,end in ts]
 new=[[]for _ in ts]
 for start in range(0,total,K):
  physical=min(K,total-start)
  for i,(first,rows,end)in enumerate(ts):
   if start>=end:continue
   logical=min(K,end-start);assert 0<logical<=physical<=K and start+physical<=total<=CONTEXT
   new[i].append((start,logical,start+logical==end))
 assert old==new,'Every Q tile must retain exact ascending K partial order/count/causal flag'
 old_calls=sum(map(len,old));old_keys=sum(count for row in old for _,count,_ in row)
 # Leave the complete original T1 branch byte-for-byte unchanged.
 new_calls=old_calls if queries==1 else(total+K-1)//K
 new_keys=old_keys if queries==1 else total
 old_acc=max(rows for first,rows,end in ts)*3*(128+2+2)*4
 new_acc=old_acc if queries==1 else queries*3*(128+2+2)*4
 length_table=4 if queries==1 else len(ts)*((total+K-1)//K)*4
 return dict(queries=queries,past=past,total=total,Q_tiles=len(ts),FA_calls=old_calls,old_pack_calls=old_calls,new_pack_calls=new_calls,metadata_init_launches=0 if queries==1 else 1,old_pack_key_rows=old_keys,new_pack_key_rows=new_keys,old_KV_logical_input_bytes=old_keys*3*(192+128)*2,new_KV_logical_input_bytes=new_keys*3*(192+128)*2,old_KV_logical_output_bytes=old_keys*3*(192+192)*2,new_KV_logical_output_bytes=new_keys*3*(192+192)*2,old_F32_accumulator_bytes=old_acc,new_F32_accumulator_bytes=new_acc,extra_F32_accumulator_bytes=new_acc-old_acc,new_GPU_length_table_bytes=length_table,extra_GPU_length_table_bytes=length_table-4,partial_order_count_causal_identical=True,T1_branch_unchanged=queries==1)

def analyze(repository,header_receipt):
 b=repository/'code/bw1000/Kimi-vocab-parallel-embedding-r461';backend=b/'split-v128-unpaged-CPU-v6/fa_backend.cpp';kernel=b/'split-v128-source-v1/fa_kernels.cu';contract=b/'split-v128-source-v1/contract.hpp'
 header=json.loads(header_receipt.read_bytes())['out'];header=json.loads(header)if isinstance(header,str)else header;h=header['headers'][0]
 assert h['SHA256']=='7eadbf8fddb83df7a1c8c4f38dcc2ee5f7eda95d6021a5ec99a85f4be2f392c3'and hashlib.sha256(h['text'].encode()).hexdigest()==h['SHA256']
 assert 'Tensor::Shape{q.size(0), q.size(2), q.size(1)}'in h['text']and 'tensor->dtype() == DataType::kInt32 && tensor->IsContiguous()'in h['text']
 rows=[case(512,0),case(512,3071),case(512,32255),case(1,32767),dict(case(512,0),reset_reversed_page_table=True)]
 # Exhaust all legal 512-query start positions for tile/table bounds, using no
 # floating point or GPU. This proves schedule bookkeeping, not FA numerics.
 maximum=max(len(tiles(512,p))for p in range(CONTEXT-512+1));assert maximum<=17
 keys=['FA_calls','old_pack_calls','new_pack_calls','metadata_init_launches','old_KV_logical_input_bytes','new_KV_logical_input_bytes','old_KV_logical_output_bytes','new_KV_logical_output_bytes']
 sums={k:4*sum(r[k]for r in rows)for k in keys}
 return dict(schema=1,state='ONE_SOURCE_COST_CARD_NOT_IMPLEMENTED_OR_READY',CPU_GO=False,GPU_GO=False,Model_GO=False,parent_Native_SHA256='91b761ec61cb840bca4726139bd9257302161abb1b2294186413163949c076a3',source_pins={str(p.relative_to(repository)).replace('\\','/'):H(p)for p in [backend,kernel,contract]},SDK_base_header=dict(path=h['path'],SHA256=h['SHA256']),original_header_receipt_SHA256=H(header_receipt),cases=rows,original_complete_ABBA20=sums,max_full512_Q_tiles_integer_exhaustive=maximum,additional_F32_accumulator_max_bytes=512*3*(128+2+2)*4-32*3*(128+2+2)*4,length_table_upper_bound_bytes=17*11*4,baseline_component_margin_bytes=48*1024*1024,component_margin_is_not_Model_fit=True,hypothesis='K-outer one packed3072 K/V tile reused across independent Q tiles. Whole-query F32 state/factor/sum persists; every Q retains identical ascending K merge order. GPU int32 length-table slices hold exactly original per-Q/K count, with same unpaged cfg and last-chunk causal flag. Original T1 code path remains unchanged.',requirements=['Precompute bounded int32 length metadata on GPU once for queries>1; each FA length view must actually be shape1 contiguous same-device, original values, not fake shape.', 'Only current full batch may be packed; no reads beyond logicaltotal or32768. Extra keys in physical tile must remain invisible through exact original per-call seqlens and bottom-right causal mask; prove this in actual numeric tests.', 'F32 state/merge kernels, per-Q K order, table/slot validation, reserved pages and all old guards remain; preserve complete20ABBA/8196tokens/logical512/24cache/full32768/4GiB.', 'No full KV materialization, V39MB scratch, new K grid, SDK implementation/config change, manualGEMM, fakeRankInfo, class/ABI change, or Model promotion.'],cost_boundary='Logical K/V element load/store counts only; excludes repeated page-table accesses, cache reuse, bandwidth, SDKworkspace, allocator/driver cost. FAcount remains1008; no measured or predicted wall speedup. Additional F32/table storage does not establish live driver peak or Model fit.',exit_criteria='Implement only this one bounded private HOST/GPU candidate after source/SDKview review, exact fresh-M CPU claim then actual full component numerics/lifecycle/capacity and same-scope warmed timing. Reject if causal/LSE checks fail, floor not met, or prefill CABI has no resolved benefit. Do not claim T1 decode acceleration.')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('repository',type=Path);p.add_argument('header_receipt',type=Path);p.add_argument('output',type=Path);a=p.parse_args();q=analyze(a.repository,a.header_receipt);q['reader_SHA256']=H(__file__);a.output.write_text(json.dumps(q,indent=2)+'\n',encoding='utf-8');print(json.dumps({'state':q['state'],'complete20_cost':q['original_complete_ABBA20'],'extra_F32_bytes':q['additional_F32_accumulator_max_bytes'],'GPU_GO':False}))
