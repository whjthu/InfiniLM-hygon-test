"""CPU-only proposed admission/accounting. No Native/provider invocation."""
def eligible(dtype,m,n,k,sr,sc,ro,co):
    return (dtype,m,n,k,sr,sc,co)==(1,2048,3584,512,224,4,0) and ro in range(0,28672,3584)

def key(dtype,m,n,k,sr,sc,ro,co):
    if not eligible(dtype,m,n,k,sr,sc,ro,co):
        raise ValueError('Only the explicit selected-KVB geometry; original path otherwise')
    # Runtime identity, device, stream and thread remain the outer Owner key.
    # W/X pointers and layer identity are deliberately absent: every run
    # still reads/unpacks the caller's current X, W and both scales.
    return dtype,m,n,k,sr,sc,ro,co

def scratch_inventory(k):
    dtype,m,n,inner,sr,sc,ro,co=k
    blocks=(co+inner+127)//128-co//128
    # Existing776 plan uses min(M,256) partial rows, not M for this geometry.
    return dict(x_BF16_bytes=blocks*m*128*2,
                w_BF16_bytes=blocks*n*128*2,
                partial_F32_bytes=blocks*min(m,256)*n*4,
                allocator_peak_or_physical_fit='UNKNOWN')

def check():
    counts=[]
    for rank in range(8):
        k=key(1,2048,3584,512,224,4,3584*rank,0)
        scratch=scratch_inventory(k);n=sum(v for v in scratch.values() if type(v)is int)
        assert n==20447232 and n/(1024**2)==19.5
        # Across layer identities and different W/X objects the same geometry
        # key remains one entry. This is a key test, not a measured GPU hit.
        assert len({key(*k)for layer in range(40)})==1
        assert (192*1024**2)//n==9
        counts.append(dict(rank=rank,key=k,logical_scratch_inventory=scratch,
                           byte_total=n,entries_192MiB_at_this_size=9,
                           actual_allocator_bytes_and_hit_count='UNKNOWN'))
    for shape in [(1,1,3584,512,224,4,0,0),(1,512,3584,512,224,4,0,0),
                  (1,8192,3584,512,224,4,0,0),(1,2048,2048,512,224,4,0,0),
                  (1,2048,3584,512,224,4,1,0),(1,2048,3584,512,224,4,0,1)]:
        assert not eligible(*shape)
    return dict(passed=True,actual_geometry_consumer=True,rank_rows=counts,
                geometry_cases=14,Native=0,GPU=0,Slurm=0,compiled=False,
                proposed_policy_default='OFF',numeric_or_Model_proof=False)

if __name__=='__main__':
    import json
    print(json.dumps(check(),indent=2))
