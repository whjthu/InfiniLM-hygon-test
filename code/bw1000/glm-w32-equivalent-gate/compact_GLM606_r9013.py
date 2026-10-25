import pathlib,json,hashlib,statistics,sys
p=pathlib.Path(sys.argv[1]);d=json.loads(p.read_text());q=d['GLM-W32-layer-r9013-terminal.json']
assert q['state']=='terminal' and q['actual_terminal']
out={'hardware':'BW1000','level':'CURRENT_LAYER_DECODE_COMPONENT','status':'TERMINAL_606_NO_MODEL_ADOPTION','source_commit':'eac9591e5fda06a921206c6099414e2429d8eeed','actual_handle':q['bound_handle'],'numeric_returncode':q['numeric_returncode'],'Slurm_rows':q['Slurm_rows'],'numeric_attempts':q['numeric_attempts'],'post':q.get('post'),'post_pass':q.get('post_pass'),'exceptions':q['exceptions'],'qualified_component':q.get('qualified_component',False),'native_SHA256':'fd655b0825b94b7b64943ee0e0b240deae2d591d3808eb40e6aa2b8ddedef9f3','scope':'layer38/M8/past512 TP8 synthetic seed8703 and realweights; originalW64V2 vs equivalentW32V4 gate only; PMC/observerOFF; onceweights; no Model TPS or adoption','source_observation_SHA256':hashlib.sha256(p.read_bytes()).hexdigest()}
if q.get('qualified_component'):
 rows=q['TPmax_trials'];vals={k:[x for x in rows[2:10] if x['mode']==k]for k in [0,1]}
 out['TPmax_medians_ms']={str(k):{m:statistics.median(x[m]for x in vals[k])for m in ['wholelayer_native_wall_ms','wholelayer_driver_CAPI_ms']}for k in vals}
 a,b=[out['TPmax_medians_ms'][str(k)]['wholelayer_native_wall_ms']for k in [0,1]]
 out['native_median_reduction_pct']=(a-b)/a*100
 out['all8_normal']=q['all8_normal'];out['audit_errors']=q['audit_errors'];out['cases']=sum(len(x['data']['trials'])for x in q['receipts'])
 out['all_crossarm_bits_exact']=all(x['data']['all_outputs_fullbits']for x in q['receipts'])
 out['owner_reduction_pct']={str(x['rank']):(x['data']['ordinary_summary']['0']['wholelayer_native_wall_ms']-x['data']['ordinary_summary']['1']['wholelayer_native_wall_ms'])/x['data']['ordinary_summary']['0']['wholelayer_native_wall_ms']*100 for x in q['receipts']}
 out['paired_TPmax_native_OFF_minus_ON_ms']=[rows[i]['wholelayer_native_wall_ms']-rows[j]['wholelayer_native_wall_ms']for i,j in [(2,3),(5,4),(7,6),(8,9)]]
 out['TPmax_trials']=rows
out['Model_adoption']=False
target=pathlib.Path('GLM606-W32-current-layer-compact-r9013.json');assert not target.exists();target.write_text(json.dumps(out,indent=2)+'\n',encoding='utf8',newline='\n');print(json.dumps(out,indent=2))
