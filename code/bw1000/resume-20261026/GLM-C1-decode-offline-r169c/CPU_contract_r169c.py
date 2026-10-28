import pathlib,sys,json,copy,collections
D=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(D));from profile_consumer_r169c import consume,ordered_tape_r169c
def fixture():
 names=['CAPI.owner.begin','PP.incoming.begin','PP.incoming.end']
 for layer,N,kind in [(0,1536,'full'),(3,256,'shared')]:
  key='detail.layer.'+str(layer)+'.prepare.projection.N'+str(N)+'.K6144'
  names+=['layer.'+str(layer)+'.'+kind+'.begin','detail.layer.'+str(layer)+'.attention.begin']
  for occurrence in range(2):names +=[key+'.quant.begin',key+'.quant.end',key+'.projection.end.before_original_fence']
  names+=['detail.layer.'+str(layer)+'.attention.end','layer.'+str(layer)+'.end']
 names+=['PP.outgoing.begin','PP.outgoing.end','CAPI.owner.end.before_original_drain']
 return [{'name':n,'owner_host_s':100+i*.001,'gpu_from_first_s':i*.001}for i,n in enumerate(names)]
tape=fixture();p,l,d,projections,c=ordered_tape_r169c(tape);assert len(projections)==4 and len(d)==6 and [p['occurrence']for p in projections]==[1,2,1,2]
for prefix in ['detail.layer.0.prepare.projection.N1536.K6144','detail.layer.3.prepare.projection.N256.K6144']:
 for suffix in ['.quant.begin','.quant.end','.projection.end.before_original_fence']:assert c[prefix+suffix]==2
checks=[]
def rejected(name,names):
 try:ordered_tape_r169c([dict(m,name=n)for m,n in zip(fixture()+fixture(),names)])
 except (ValueError,IndexError):checks.append(name);return
 raise AssertionError(name+' accepted')
names=[m['name']for m in tape]
bad=names.copy();bad.remove('detail.layer.0.prepare.projection.N1536.K6144.quant.end');rejected('missing_quant_end',bad)
bad=names.copy();bad.insert(2,'PP.incoming.end');rejected('duplicate_structural_endpoint',bad)
bad=names.copy();i=bad.index('detail.layer.0.prepare.projection.N1536.K6144.quant.begin');bad[i],bad[i+1]=bad[i+1],bad[i];rejected('end_before_begin',bad)
bad=names.copy();i=bad.index('detail.layer.0.attention.end');bad.insert(i,'detail.layer.0.prepare.projection.N1536.K6144.quant.begin');rejected('crossed_scopes',bad)
bad=names.copy();i=bad.index('detail.layer.0.prepare.projection.N1536.K6144.projection.end.before_original_fence');bad.insert(i,bad[i]);rejected('duplicate_orphan_projection_fence',bad)
bad=names.copy();i=bad.index('layer.0.end');bad[i]='layer.3.end';rejected('mismatched_layer_end',bad)
calls=[{'event_index':i,'op':'DECODE','rows':[0],'pasts':[100+i],'counts':[1]}for i in range(20)];events=[{'prefill':False,'native_seconds':.03}for _ in calls]
obs={'selector_revision':'r169','required':[0,0,0,0,3],'coverage':[0,0,0,0,3],'samples':[{'selector':'decode_T1','tokens':1,'sequence':j,'eligible_decode1_index':j,'row_past_count':[[0,99+j,1]],'marks':fixture()}for j in [17,18,19]]}
assert consume(obs,calls,events)['coverage']=='COMPLETE'
bad=copy.deepcopy(calls);bad[16]['op']='ANCHOR'
try:consume(obs,bad,events)
except ValueError:checks.append('ANCHOR_is_not_decode')
else:raise AssertionError('ANCHOR accepted')
bad=copy.deepcopy(obs);bad['samples'][0]['marks'][5]['owner_host_s']=0
try:consume(bad,calls,events)
except ValueError:checks.append('nonmonotonic_clock')
else:raise AssertionError('bad clock accepted')
print(json.dumps({'passed':True,'CPU_only':True,'GPU':False,'model':False,'synthetic_clocks_only':True,'actual_metadata_repeat_names_used':True,'all_4_projection_occurrences_retained':True,'legal_repeat_name_multiplicity':2,'negative_contracts':checks}))
