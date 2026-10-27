"""Concrete retained state accounting; UNKNOWN allocations are not zero."""
def budget(physical_rows,active_rows=None):
 assert isinstance(physical_rows,int)and 1<=physical_rows<=128
 active_rows=physical_rows if active_rows is None else active_rows
 assert 1<=active_rows<=physical_rows
 fixed_pending=6103040*physical_rows
 seal_copy=fixed_pending # partial acceptance may retain the second full copy
 ring_pages=(physical_rows+1)//2*43
 ring_COW=149760*ring_pages
 return {'physical_rows':physical_rows,'active_rows':active_rows,'pending_fixed_state_bytes':fixed_pending,'partial_seal_extra_fixed_bytes':seal_copy,'max_ring_COW_touched_pages':ring_pages,'max_ring_COW_bytes':ring_COW,'known_incremental_retained_bound_bytes':fixed_pending+seal_copy+ring_COW,'unknown_additional':['retained score/query/topk/indexer projections','allocator fragmentation/pools','native driver/runtime/CCL high water'],'baseline_weights_cache_bytes':'additional, not included','total_device_peak_bound':'UNKNOWN','667364_bytes_per_row_is_actual':False,'admission':'No GPU admission from this partial bound. Require measured actual forward/seal/publish/cancel owner high water and >=4GiB free.'}
if __name__=='__main__':
 import json
 print(json.dumps({'configured_rows':[budget(n)for n in [1,2,4,8,16]],'source_only':True},indent=2))
