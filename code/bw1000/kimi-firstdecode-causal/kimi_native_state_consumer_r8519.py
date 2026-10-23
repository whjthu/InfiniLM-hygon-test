"""Decode existing KDA/MLA trace artifacts using authoritative on-disk dtype.
No GPU/native restore. Logical target mapping comes from actual metadata.
"""
import pathlib,json,math
import numpy as np
import hashlib
DT={'bf16':np.dtype('<u2'),'f32':np.dtype('<f4'),'i32':np.dtype('<i4'),'i64':np.dtype('<i8')}
def read_snapshot(directory,record):
 p=pathlib.Path(directory)/record['file'];assert p.name==record['file']
 raw=p.read_bytes();dtype=DT[record['dtype']];assert len(raw)==record['bytes']==math.prod(record['shape'])*dtype.itemsize
 assert record['source_layout']['shape']==record['shape']
 assert len(record['source_layout']['strides'])==len(record['shape'])
 return np.frombuffer(raw,dtype=dtype).reshape(record['shape'])
def active_page_contract(offsets,pasts,totals,tables,slots,*,blocks=132,page_size=256):
 # Native constructor: offsets,pasts,totals,table I32; slots I64.
 assert offsets.dtype==pasts.dtype==totals.dtype==tables.dtype==np.dtype('<i4')
 assert slots.dtype==np.dtype('<i8') and 0<len(pasts)<=4
 assert tables.shape==(len(pasts),4) and len(offsets)==len(pasts)+1
 assert int(offsets[0])==0 and int(offsets[-1])==len(slots)
 users=[];union=set()
 for i,(past,total)in enumerate(zip(pasts,totals)):
  begin,end=map(int,offsets[i:i+2]);past,total=int(past),int(total)
  assert 0<=past<total<=1024 and end-begin==total-past and end>begin
  pages=[int(v)for v in tables[i,:(total+page_size-1)//page_size]]
  assert all(0<=v<blocks for v in pages)
  for t in range(begin,end):
   pos=past+t-begin;assert int(slots[t])==int(tables[i,pos//page_size])*page_size+pos%page_size
  union.update(pages);users.append(dict(packed_request=i,read_prefix_tokens=past,total_tokens=total,actual_active_pages=pages,packed_offsets=[begin,end]))
 assert len(union)<=16
 return dict(users=users,all_companion_page_union=sorted(union),restore_implemented=False,scope='Observation parser, not same-state clone. Compare target validprefix; retain allcompanion union for future faithful component.')


def gather_valid_KV(page_copies,block_table,valid_tokens):
    """Logical valid prefix only. Raw page tail is not state-equality evidence."""
    assert 0<=valid_tokens<=1024 and len(block_table)==4 and page_copies
    sample=next(iter(page_copies.values()));assert sample.ndim==5 and sample.shape[0:3]==(2,1,256)
    assert sample.dtype==np.dtype('<u2')
    if valid_tokens==0:return np.empty((2,0,*sample.shape[3:]),dtype=sample.dtype)
    pieces=[]
    for pos in range(valid_tokens):
        page=page_copies[int(block_table[pos//256])]
        assert page.shape==sample.shape and page.dtype==sample.dtype
        pieces.append(page[:,0,pos%256,:,:])
    return np.stack(pieces,axis=1)

def read_MLA_state_phase(directory,journal_rows,phase):
    assert phase in ('before','after')
    bypoint={row['point']:row for row in journal_rows};prefix='kv_'+phase+'_'
    def get(name):return read_snapshot(directory,bypoint[prefix+name])
    offsets,pasts,totals,tables,slots=[get(k)for k in ('offsets','pasts','totals','block_tables','slot_mapping')]
    mapping=active_page_contract(offsets,pasts,totals,tables,slots)
    pages={int(k.removeprefix(prefix+'block_')):read_snapshot(directory,v)for k,v in bypoint.items()if k.startswith(prefix+'block_')}
    assert set(mapping['all_companion_page_union'])==set(pages)
    logical=[gather_valid_KV(pages,tables[i],int(pasts[i]if phase=='before'else totals[i]))for i in range(len(pasts))]
    return dict(metadata=mapping,logical_valid_KV=logical,raw_pages=pages,restore_implemented=False)

def state_row_indices(journal_rows,directory):
    bypoint={v['point']:v for v in journal_rows}
    values=[read_snapshot(directory,bypoint[k]).reshape(-1)for k in ('read_rows','write_rows')]
    assert all(v.dtype in (np.dtype('<i4'),np.dtype('<i8'))for v in values)
    reads,writes=values
    assert 0<len(reads)==len(writes)<=4
    return reads,writes

def read_KDA_state_phase(directory,journal_rows,phase,packed_target):
    assert phase in ('before','after')
    reads,writes=state_row_indices(journal_rows,directory)
    assert 0<=packed_target<len(reads)
    row=int((reads if phase=='before' else writes)[packed_target])
    bypoint={v['point']:v for v in journal_rows}
    result={}
    for kind in ('conv','ssm'):
        rec=bypoint[kind+'_'+phase+'_row_'+str(row)]
        layout=rec['source_layout']
        assert layout['pool_row']==row
        assert layout['pool_row_offset_elements']==row*layout['pool_strides'][0]
        assert layout['pool_shape'][0]>row>=0
        assert rec['shape'][0]==1 and rec['shape'][1:]==layout['pool_shape'][1:]
        result[kind]=read_snapshot(directory,rec)
    return dict(actual_state_row=row,tensors=result,restore_implemented=False)

def array_summary(value):
    return dict(shape=list(value.shape),storage_dtype=value.dtype.str,
                bytes=value.nbytes,SHA256=hashlib.sha256(value.tobytes()).hexdigest())

def target_intermediate(value,point,begin,end,total):
    if point=='normalized_f32':
        assert value.ndim==2 and value.shape[0]==total*12
        return value.reshape(total,12,value.shape[1])[begin:end]
    if value.ndim>=3 and value.shape[0]==1 and value.shape[1]==total:
        return value[:,begin:end]
    if value.ndim>=2 and value.shape[0]==total:
        return value[begin:end]
    raise AssertionError(('unknown_target_token_axis',point,value.shape,total))

def summarize_native_target(directory,job,packed_target):
    """Read raw arrays on the server and export only target hashes/layouts.

    Uses actual read/write indices, table slots and packed offsets. Does not
    restore native state or compare unused page tails as valid KV values.
    """
    p=pathlib.Path(directory);out=[]
    widths=list(map(len,job['inputs']));begin=sum(widths[:packed_target]);end=begin+widths[packed_target];total=sum(widths)
    for journal in sorted(p.glob('*-kda-call*.jsonl')):
        rows=[json.loads(line)for line in journal.read_text().splitlines()]
        assert rows and len(rows)<=64
        points={v['point']for v in rows}
        base=dict(stage=rows[0]['stage'],rank=rows[0]['rank'],layer=rows[0]['layer'],
                  raw_bytes=sum(v['bytes']for v in rows),raw_files=len(rows),
                  journal=str(journal),states={},intermediates={})
        for phase in ('before','after'):
            if 'kv_'+phase+'_offsets' in points:
                state=read_MLA_state_phase(p,rows,phase)
                base['states']['KV_'+phase]=array_summary(state['logical_valid_KV'][packed_target])
                base['states']['KV_'+phase]['actual_mapping']=state['metadata']['users'][packed_target]
            elif 'read_rows' in points:
                state=read_KDA_state_phase(p,rows,phase,packed_target)
                for k,v in state['tensors'].items():
                    base['states'][k+'_'+phase]=dict(**array_summary(v),actual_state_row=state['actual_state_row'])
        for row in rows:
            point=row['point']
            if row['dtype']not in ('bf16','f32')or point.startswith(('kv_','conv_before','conv_after','ssm_before','ssm_after'))or point in ('a_log','dt_bias'):continue
            value=read_snapshot(p,row)
            selected=target_intermediate(value,point,begin,end,total)
            base['intermediates'][point]=dict(**array_summary(selected),source_layout=row['source_layout'],packed_offsets=[begin,end])
        out.append(base)
    return out
