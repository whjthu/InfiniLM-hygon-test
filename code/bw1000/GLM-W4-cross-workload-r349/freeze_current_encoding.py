import pathlib,json,hashlib
D=pathlib.Path(__file__).parent;results=D.parents[2]/'results/bw1000/2026-11-02'
observed=json.loads((results/'GLM349-official-encoder-v3-observed.json').read_bytes())
entry=next(x for x in observed['metadata'] if x['data']['actual_CPU']=='910407.673');report=entry['data'];assert report['passed'] and report['completed'] and report['GPU']==report['Model']==0
card=json.loads((D/'fixtures/FIXTURE_CARD.json').read_bytes())
for f,current in zip(card['fixtures'],report['fixtures']):
 assert f['id']==current['id'] and f['request_SHA256']==current['request_SHA256'] and current['input_unchanged']
 f.update(current_prepared_prompt_IDs=current['private_IDs_path'],current_prepared_prompt_IDs_SHA256=current['prompt_IDs_SHA256'],current_prepared_prompt_tokens=current['prompt_tokens'],current_prepared_prompt_render_SHA256=current['prompt_render_SHA256'],current_prepared_private_IDs_file_SHA256=current['private_IDs_file_SHA256'])
card['official_encoding_CPU']={'actual_step':'910407.673','state':'COMPLETED','exit_code':'0:0','passed':True,'root_core_llm_guarded':True,'actual_root_or_ABI_qualified':False,'full_history_unchanged':True,'private_Native_dlopen':False,'weights_loaded':False,'metadata_external_path':entry['path'],'prepared_entry_identity':report['prepared_entry_identity']}
raw=(json.dumps(card,ensure_ascii=False,indent=2)+'\n').encode();(D/'fixtures/FIXTURE_CARD.json').write_bytes(raw);(D/'owner/FIXTURE_CARD.json').write_bytes(raw)
curated={'completed':True,'passed':True,'actual_CPU':'910407.673','GPU':0,'Model':0,'scope':'Actual prepared processor/server/engine CPU encoding with guarded root/core/llm; not ABI/GPU qualification','preserved_failures':['910407.670','910407.671'],'metadata_external_path':entry['path'],'fixtures':[{k:v for k,v in f.items() if k!='history_render_audit'} for f in report['fixtures']],'prepared_entry_identity':report['prepared_entry_identity'],'full_IDs_sequences_stay_private':True,'fixture_card_SHA256':hashlib.sha256(raw).hexdigest()}
(results/'GLM349-official-encoding-CPU673-PASS-metadata.json').write_bytes((json.dumps(curated,ensure_ascii=False,indent=2)+'\n').encode())
print(json.dumps({'fixture_card_SHA256':hashlib.sha256(raw).hexdigest(),'passed':True}))
