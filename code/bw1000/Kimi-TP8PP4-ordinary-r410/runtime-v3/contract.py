"""Production ordinary topology contract; no mocked preengine admission."""
import hashlib,json
from pathlib import Path
CUTS=[[0,24],[24,47],[47,71],[71,93]]
GEOMETRY=dict(max_rows=2,pages=260,page_size=256,KDA_pool_rows=3,pages_per_request=128,max_total_tokens=32768)
CAP=dict(version=1,TP=8,PP=4,C=1,physical_rows=2,state_rows=3,pages=260,page_size=256,context=32768,prompt_chunk=512,decode_width=1,MLA_shape=[2,260,256,12,192],KDA_heads=12,local_MoE_I=384,speculation=False,capture=False,GPU_qualified=False)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def layout(node):
 assert node in range(4)
 return dict(node=node,local_workers=8,TP=8,TP_node_rank=0,PP=4,PP_stage=node,sampling_node=3,public_graph_enabled=False,owned_metadata_enabled=False)
def memory_gate(packets):
 assert len(packets)==4 and {p['node']for p in packets}==set(range(4))
 values=[v[0]for p in packets for v in p['snapshot']['memory']]
 assert len(values)==32 and min(values)>4<<30,'Original all32owner 4GiB floor unchanged'
 return min(values)
def validate_config(cfg):
 t=cfg['text_config'];assert cfg['model_type']=='kimi_k3'and t['num_hidden_layers']==93 and t['hidden_size']==7168 and t['vocab_size']==163840
 assert t['num_attention_heads']==96 and t['qk_nope_head_dim']==128 and t['kv_lora_rank']==512 and t['qk_rope_head_dim']==64
 assert t['linear_attn_config']['num_heads']==96 and t['linear_attn_config']['head_dim']==128 and t['max_position_embeddings']>=32768
 return t
def frozen_fixtures(path,pins):
 assert sha(path)==pins['frozen_inputs_SHA256'];f=json.loads(Path(path).read_bytes());rows=f['fixtures'];assert len(rows)==2
 from ordinary import ids_sha
 for i,(row,meta)in enumerate(zip(rows,pins['fixtures'])):
  assert all(row[k]==v for k,v in meta.items()),'frozen fixture metadata changed'
  assert row['input_tokens']==[8635,32638][i]and len(row['prompt_tokens'])==row['input_tokens']and ids_sha(row['prompt_tokens'])==row['prompt_ID_SHA256']
  assert row['output_budget']==128 and row['eos']==[163585,163586]and row['input_tokens']+128<=32768
 return rows
