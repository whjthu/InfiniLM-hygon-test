"""Bounded CPU file-seek extraction; all weight bytes stay remote/in RAM."""
import pathlib,json,struct,hashlib
def extract():
 r=pathlib.Path('/refactor');m=pathlib.Path('/old/models/ZhipuAI-GLM-5.3')
 fixture=r/'results/glm-I512-real-inputs-r5713.json';assert hashlib.sha256(fixture.read_bytes()).hexdigest()=='2f1076c4933394579097fc58fb399b53667f11cbce81a06d263aa5b4e2d83bbc'
 prompts=[v['input_ids']for v in json.loads(fixture.read_text())['cases']];assert len(prompts)==16 and all(len(p)==512 for p in prompts)
 tokens=sorted(set(x for row in prompts for x in row));assert len(tokens)==341
 index=json.loads((m/'model.safetensors.index.json').read_text())['weight_map'];headers={};offsets={}
 def metadata(k):
  p=m/index[k]
  if p not in headers:
   with p.open('rb')as f:n=struct.unpack('<Q',f.read(8))[0];assert n<16*1024**2;headers[p]=json.loads(f.read(n));offsets[p]=8+n
  return p,headers[p][k]
 def read(k):
  p,h=metadata(k);a,b=h['data_offsets']
  with p.open('rb')as f:f.seek(offsets[p]+a);data=f.read(b-a)
  assert len(data)==b-a;return {'shape':h['shape'],'dtype':h['dtype'],'bytes':data}
 keys=['input_layernorm.weight']+['self_attn.'+n for n in ['q_a_proj.weight','q_a_proj.weight_scale_inv','q_b_proj.weight','q_b_proj.weight_scale_inv','kv_a_proj_with_mqa.weight','kv_a_proj_with_mqa.weight_scale_inv','kv_b_proj.weight','kv_b_proj.weight_scale_inv','q_a_layernorm.weight','kv_a_layernorm.weight']]
 weights={k:read('model.layers.0.'+k)for k in keys};assert sum(len(v['bytes'])for v in weights.values())==64389568
 p,h=metadata('model.embed_tokens.weight');assert h['shape']==[154880,6144]and h['dtype']=='BF16'
 rows=[]
 with p.open('rb')as f:
  for token in tokens:f.seek(offsets[p]+h['data_offsets'][0]+token*12288);data=f.read(12288);assert len(data)==12288;rows.append(data)
 remap={t:i for i,t in enumerate(tokens)}
 return weights,b''.join(rows),[[remap[t]for t in row]for row in prompts]
if __name__=='__main__':
 w,e,p=extract();print(json.dumps({'completed':True,'weights':{k:{'shape':v['shape'],'dtype':v['dtype'],'bytes':len(v['bytes']),'SHA256':hashlib.sha256(v['bytes']).hexdigest()}for k,v in w.items()},'embedding_rows':341,'embedding_bytes':len(e),'embedding_SHA256':hashlib.sha256(e).hexdigest(),'remapped_ID_range':[min(map(min,p)),max(map(max,p))],'GPU_initialized':False,'raw_kept_remote':True}))
