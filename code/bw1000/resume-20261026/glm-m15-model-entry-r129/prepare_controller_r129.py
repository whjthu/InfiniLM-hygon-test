import pathlib,json,hashlib,ast
o=pathlib.Path(__file__).resolve().parent;r=pathlib.Path(r'C:\Users\WangHaojie\Documents\ChatGPT\海光集群');base=r/'GLM-down-row2-Model-private-r9284b';template=(base/'controller.py').read_text();s=template.replace('GLM-down-row2-Model-r9284b','GLM-M15-Model-r129').replace('r9284b','r129').replace('GLM_ARRIVAL_NODE_R9284B_NORMAL','GLM_ARRIVAL_NODE_R129_NORMAL')
s=s.replace("r=pathlib.Path.home()/'hygon-inference-refactor-20261014'", "r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')")
s=s.replace("assert keeper.startswith(parent+'.')and keeper.split('.')[-1].isdigit()", "assert keeper.startswith('910407.')and keeper.split('.')[-1].isdigit()and socket.gethostname()=='a14r2n20'")
s=s.replace("import pathlib,json,hashlib,subprocess,os,sys,time,uuid,datetime,socket,traceback,re", "import pathlib,json,hashlib,subprocess,os,sys,time,uuid,datetime,socket,traceback,re,argparse")
needle="record=r/(label+'-launch.json');"
prefix="""ap=argparse.ArgumentParser();ap.add_argument('--go-file',required=True);ap.add_argument('--manifest-sha',required=True);opt=ap.parse_args()
assert hashlib.sha256((inc/'GPU-manifest.json').read_bytes()).hexdigest()==opt.manifest_sha
go_path=pathlib.Path(opt.go_file).resolve();assert go_path.is_relative_to(r.resolve());go=json.loads(go_path.read_bytes())
assert go['GPU_authorized']is True and go['source_published']is True and go['controller_manifest_SHA256']==opt.manifest_sha and go['CPU_receipt_SHA256']=='83bd99026decb9f7b38acd4301506854288bfe2887c3265e32b3051f3813c009'and re.fullmatch('[0-9a-f]{40}',go['publication_commit'])and go['publication_branch']=='bw1000-experiments'
"""
assert s.count(needle)==1;s=s.replace(needle,prefix+needle)
s=s.replace("'restarts':0};write(record,d)","'restarts':0,'root_GO_SHA256':sha(go_path),'publication_commit':go['publication_commit']};write(record,d)")
needle=" cp=json.loads((inc/'CPU-result.json').read_bytes());assert cp['completed']and cp['actual_nodes']==nodes"
gate=""" cp_path=r/'candidates/GLM-M15-oproj-Model-r125/CPU-build.json';assert sha(cp_path)==go['CPU_receipt_SHA256']==m['CPU_receipt_SHA256'];cp=json.loads(cp_path.read_bytes());assert cp['completed']and cp['CPU_only']and cp['HOST_TUs_compiled']==3 and cp['GPU_TUs_compiled']==0 and not cp['Context_layout_changed'];assert sha(str(cp['library']).replace('/refactor',str(r),1))==cp['library_SHA256']==m['Model_SHA256']
 # No launch while r127 is pending: exact result and terminal/post watch pins are mandatory.
 offset_result=r/'results/GLM-M15-offset8-r127-910376.761.json';assert go['offset127_result_SHA256']==m['offset127_result_SHA256']==sha(offset_result)
 offset_watch=r/'GLM-M15-offset8-r127-watch.json';assert re.fullmatch('[0-9a-f]{64}',go['offset127_watch_SHA256'])and sha(offset_watch)==go['offset127_watch_SHA256']==m['offset127_watch_SHA256'];ow=json.loads(offset_watch.read_bytes());oq=json.loads(offset_result.read_bytes())
 assert ow['actual_handle']=='910376.761'and ow['qualified']and ow['numeric_positive']and ow['post_passed']and ow['terminal'][2:4]==['COMPLETED','0:0']and ow['result_SHA256']==sha(offset_result)
 assert oq['completed']and oq['normal_close']and oq['all_offsets_validated']and oq['all_offsets_positive']and oq['all_offsets_qualified']and oq['library_SHA256']=='20b240a4403c0b21b6814f4448634dd6dfca8df2511af5d146ddd00512b36cda'
 assert [a['logical_rank']for a in oq['offsets']]==list(range(8))and all(a['positive_pairs']==4 and a['fresh_weight_audit']and a['fresh_scale_audit']and a['clear_reuse_audit']and a['normal_close']for a in oq['offsets'])
 d.update(offset127_result_SHA256=sha(offset_result),offset127_watch_SHA256=sha(offset_watch),CPU_receipt_SHA256=sha(cp_path));write(record,d)"""
assert s.count(needle)==1;s=s.replace(needle,gate)
s=s.replace("'--cpu-bind=none','-N2'", "'--immediate=30','--cpu-bind=none','-N2'")
s=s.replace("'--cpu-bind=none','--kill-on-bad-exit=1'", "'--immediate=30','--cpu-bind=none','--kill-on-bad-exit=1'")
s=s.replace('GLM-M15-Model-r129/GPU_inside.sh','GLM-M15-Model-r129/GPU_inside_r129.sh').replace("str(inc/'audit.py')", "str(inc/'audit_r129.py')")
ast.parse(s);(o/'controller_r129.py').write_text(s,encoding='utf8')
launch=(o.parent/'glm-m15-controller-r123/launch_CPU_keeper_r123.py').read_text().replace('GLM-M15-controller-r123','GLM-M15-Model-r129').replace('GLM-M15-once-r123','GLM-M15-Model-r129').replace('GLM-M15-once-keeper-r123','GLM-M15-Model-keeper-r129').replace('manifest-r123.json','GPU-manifest.json').replace('controller_r123.py','controller_r129.py').replace("'00:30:00'","'01:45:00'").replace('candidates/GLM-M15-component-r121b/CPU-build.json','candidates/GLM-M15-oproj-Model-r125/CPU-build.json')
ast.parse(launch);(o/'launch_CPU_keeper_r129.py').write_text(launch,encoding='utf8');print('realModel16card controller generated; r127 proof and rootGO required')
