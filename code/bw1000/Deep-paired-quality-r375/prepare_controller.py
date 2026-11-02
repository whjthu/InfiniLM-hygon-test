"""Derive one replica controller from the successful366 controller; no launch."""
import ast
import hashlib
import json
import pathlib
import shutil

D = pathlib.Path(__file__).resolve().parent
BASE = D.parent / 'Deep-HC-T1-Model-r366/controller-v2'
OUT = D / 'controller'
OUT.mkdir(exist_ok=True)
for name in ['dso_binding.py', 'namespace_binding.py', 'resource_wave_host_alias_outer.sh']:
    shutil.copyfile(BASE / name, OUT / name)
s = (BASE / 'controller_366.py').read_text()


def one(old, new, count=1):
    global s
    assert s.count(old) == count, (old, s.count(old), count)
    s = s.replace(old, new)


one("S=R/'incoming/Deep-HC-T1-Model-r366/owner-v2';LABEL='Deep-HC-T1-Model-r366';NODE='a02r4n01';",
    "S=R/'incoming/Deep-paired-quality-r375/owner';REPLICA=int(os.environ['DEEP_R375_REPLICA']);assert REPLICA in range(4);PARENT,NODE=[(910374,'a02r4n01'),(910374,'a02r4n05'),(910374,'a02r4n07'),(910371,'a02r3n16')][REPLICA];LABEL='Deep-paired-quality-r375-r'+str(REPLICA);")
a = s.index("OWNER_SHA="); b = s.index('\n\n', a)
s = s[:a] + "OWNER_SHA='1b93c3db22ab59c6768e401dc46f71b630a768934fe316135b53d780d1b3fafb'\nCPU_SOURCE_SHA='7e7cfdf907a80fc2fbf01791cd6c70fbbda1640347c054241cd7c3c1fb9a128e'\n" + s[b:]
a = s.index('def validate_scope(g):'); b = s.index('def clean_env()', a)
s = s[:a] + '''def validate_scope(g):
 if g.get('GPU_authorized') is not True:raise PermissionError('Published exact four-replica delegated quality claim required')
 assert g['publication_branch']=='bw1000-experiments' and re.fullmatch('[0-9a-f]{40}',g['publication_commit'])
 assert g['replica']==REPLICA and g['GPU_parent']==PARENT and g['GPU_node']==NODE
 assert g['owner_manifest_SHA256']==OWNER_SHA and re.fullmatch('[0-9a-f]{64}',g['controller_manifest_SHA256'])
 assert g['CPU_keeper_parent']==910407 and g['CPU_keeper_node']=='a14r2n20'
 for k in ['replica','GPU_parent','cards','TP','PP','context','ordinary_budget','thinking_budget','max_HTTP','max_Native','GPU_max_s','Native_max_s','CPU_keeper_max_s']:assert type(g[k])is int
 assert (g['cards'],g['TP'],g['PP'],g['context'],g['ordinary_budget'],g['thinking_budget'],g['max_HTTP'],g['max_Native'])==(8,8,1,32768,4096,8192,66,66)
 assert (g['GPU_max_s'],g['Native_max_s'],g['CPU_keeper_max_s'])==(7200,7140,7800)
 assert g['one_attempt'] is True and g['no_retry_reload'] is True and g['prefill_chunk']==1024
 assert g['CPU_source_manifest_SHA256']==CPU_SOURCE_SHA
 assert re.fullmatch('[0-9a-f]{64}',g['CPU_entry_receipt_SHA256']) and re.fullmatch(r'910407[.]\\d+',g['CPU_actual_step'])
 assert g['wrapper_SHA256']==NAMESPACE_WRAPPER_SHA and g['DSO_CPU621_receipt_SHA256']==CPU621_SHA
 return g

def host_path(value):
 p=str(value);return R/p[len('/refactor/'):] if p.startswith('/refactor/') else pathlib.Path(p)

def verify_sources(g):
 validate_scope(g);assert sha(D/'SOURCE_MANIFEST.json')==g['controller_manifest_SHA256']
 m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(D/n)==h for n,h in m['files_SHA256'].items())
 assert sha(S/'SOURCE_MANIFEST.json')==OWNER_SHA;sm=json.loads((S/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(S/n)==h for n,h in sm['files_SHA256'].items())
 cp=R/'candidates/Deep-paired-quality-r375'/('CPU-'+g['CPU_actual_step'])/'CPU-result.json'
 assert sha(cp)==g['CPU_entry_receipt_SHA256'];c=json.loads(cp.read_bytes())
 assert c['completed'] and c['passed'] and c['CPU_only'] and c['GPU']==c['Model']==0 and not c['Native_dlopen'] and not c['weights_opened']
 assert c['actual_CPU']==g['CPU_actual_step'] and c['owner_manifest_SHA256']==OWNER_SHA and c['CPU_source_manifest_SHA256']==CPU_SOURCE_SHA
 assert c['inherited_encoding_749'] and c['no_reencoding_of_129']
 source=R/'incoming/Deep-paired-quality-r375/repair-v4/CPU_SOURCE_MANIFEST.json';assert sha(source)==CPU_SOURCE_SHA
 cm=json.loads(source.read_bytes());assert all(sha(source.parent/n)==h for n,h in cm['files_SHA256'].items())
 cases=c['whole_entry_cases'];seams=[x for x in cases if x['kind']=='actual_worker_report_to_actual_client_main66']
 assert {x['replica'] for x in seams}==set(range(4)) and all(x['completed_HTTP']==66 and x['passed'] for x in seams)
 assert any(x['kind']=='actual_client_main_recorded4096cap_continues66' and x['completed_HTTP']==66 for x in cases)
 assert {x['scenario'] for x in cases if x['kind']=='actual_worker_serve_cleanup' and x['passed']} >= {'success66','coldload','Native-failure','HC-ready-failure','HC-hit-failure','client-failure'}
 ep=host_path(c['encoding_receipt_path']);assert sha(ep)==c['encoding_receipt_SHA256']==g['encoding_receipt_SHA256']=='f17342d2ce78e73b9f8853720c6b477c947ccf8d71a7503dcd7e04e391e16447'
 assert g['encoding_receipt_container_path']==c['encoding_receipt_path']
 e=json.loads(ep.read_bytes());assert e['completed'] and e['passed'] and e['actual_prepared_ingress'] and e['retired_actual_requests']==129
 assert e['GPU']==e['Model']==0 and not e['weights_loaded'] and e['first_actual_encode_finish_real_stream_finally_retired']
 pin=json.loads((S/'NATIVE_BUILD_PIN.json').read_bytes());native=R/'candidates/Deep-HC-T1-Model-r366-cpu-v1/CPU-build.json';assert sha(native)==pin['CPU_receipt_SHA256'];build=json.loads(native.read_bytes())
 assert build['passed'] and build['library_SHA256']==g['Native_SHA256']==pin['library_SHA256']
 for n,h in build['all_objects_SHA256'].items():assert sha(host_path(n))==h
 assert sha(R/'candidates/Deep-HC-T1-Model-r366-cpu-v1/libDeep_HC_T1_Model_r366.so')==g['Native_SHA256']
 verify_namespace(R,D);verify_dso(R,R/'incoming/Deep-latest-full-owner-r300','Deep')
 return c

def terminal_report(path,visibility_s=90):
 end=time.monotonic()+visibility_s
 while True:
  try:
   value=json.loads(path.read_bytes())
   if value.get('completed')is True and (value.get('passed')is False or value.get('normal_Native_shutdown')is True):return value
  except (FileNotFoundError,json.JSONDecodeError):pass
  if time.monotonic()>=end:raise TimeoutError('Terminal report not complete within bounded90s visibility; no GPU replay')
  time.sleep(1)

''' + s[b:]
one("re.fullmatch(r'910374[.]\\d+',row[0])", "re.fullmatch(rf'{PARENT}[.]\\d+',row[0])", 2)
one("parent(910374,NODE,env)", "parent(PARENT,NODE,env)")
one("'-j','910374'", "'-j',str(PARENT)", 2)
one("'--jobid=910374'", "'--jobid='+str(PARENT)", 2)
one("'GPU_max_s':2700,'CPU_keeper_max_s':3000", "'GPU_max_s':7200,'CPU_keeper_max_s':7800")
one("'scope':'Derived afc6 singleHC-TU Native976f TP8PP1; frozen8489prompt fixed1024/output128; warmOFFON thenOFFONONOFF; no inheritedcomponent Modelgain; bufferedHTTP not clientTTFT/SLO'", "'scope':'One of four independent8card Native976f HC OFFON/ONOFF quality replicas; custom0shot128 cases globally; full4096/32768/1024; scores/caps do not gate completion'")
one("str(D/'idle_r283.py')", "str(D/'idle_r375.py'),'--node',NODE")
one('def idle(suffix):', 'def idle(suffix,owner_tids=()):')
one("str(D/'idle_r375.py'),'--node',NODE", "str(D/'idle_r375.py'),'--node',NODE,'--owner-tids',json.dumps(list(owner_tids))")
one("and rows[0]['HTTP43882_bindable'];return rows[0]", "and rows[0]['HTTP43882_bindable'] and rows[0]['owner_tids_all_gone'];return rows[0]")
one("WAVE_INSIDE_R2323=Deep-HC-T1-Model-r366/owner-v2/GPU_inside_r277.sh", "WAVE_INSIDE_R2323=Deep-paired-quality-r375/owner/GPU_inside_r277.sh")
one("+OWNER_SHA\n  args=", "+OWNER_SHA+',DEEP_R375_REPLICA='+str(REPLICA)+',DEEP_R375_ENCODING_PATH='+g['encoding_receipt_container_path']+',DEEP_R375_ENCODING_SHA='+g['encoding_receipt_SHA256']\n  args=")
gpu_prefix="  args=['srun','--jobid='+str(PARENT),'--overlap','--exact','--immediate=30','--cpu-bind=none','--kill-on-bad-exit=1'"
one(gpu_prefix, "  gpu_env=dict(env)\n  for item in export.split(',')[1:]:\n   key,value=item.split('=',1);gpu_env[key]=value\n"+gpu_prefix)
one('proc=subprocess.Popen(args,env=env,', 'proc=subprocess.Popen(args,env=gpu_env,')
one("'00:45:00'", "'02:00:00'")
one('end=start+2760', 'end=start+7260')
one("Deep-HC-T1-Model-r366-'+actual", "Deep-paired-quality-r375-'+actual")
one("q.get('actual_HTTP')==6 and q.get('actual_Native')==6", "q.get('actual_HTTP')==66 and q.get('actual_Native')==66 and q.get('attempted_Native')==66 and q.get('completed_Native')==66")
one("  w['post']=idle('post');w['post_empty']=True", "  exit_end=time.monotonic()+90\n  while proc.poll() is None and time.monotonic()<exit_end:time.sleep(1)\n  w['srun_exit_code']=proc.poll();assert w['srun_exit_code'] is not None,'srun exit unresolved after terminal; no replay'\n  final_report=None;owner_tids=[];report_error=None\n  if not terminal_failure(w['terminal']) and w['srun_exit_code']==0:\n   try:\n    final_report=terminal_report(R/'results'/('Deep-paired-quality-r375-'+actual+'.json'))\n    owner_tids=[x['tid'] for x in final_report.get('owner_exit_observation',{}).get('owners',[])]\n   except BaseException as exc:report_error=exc;w['terminal_report_error']=repr(exc)\n  w['post']=idle('post',owner_tids);w['post_empty']=True;w['post_owner_TID_count']=len(owner_tids);w['post_owner_TID_evidence']='KNOWN' if owner_tids else 'UNKNOWN_NO_THREAD_PROOF';save()\n  if report_error is not None:raise report_error")
one("if terminal_failure(w['terminal']):", "if terminal_failure(w['terminal']) or w['srun_exit_code']!=0:")
one("  if result.exists():\n   q=json.loads(result.read_bytes());", "  if True:\n   q=final_report if final_report is not None else terminal_report(result);")
one(" save()\n try:\n", " with W.open('x') as first:json.dump(w,first);first.flush();os.fsync(first.fileno())\n save()\n try:\n")
ast.parse(s)
(OUT / 'controller_375.py').write_text(s, encoding='utf8', newline='\n')
idle = (BASE / 'idle_r283.py').read_text().replace('import pathlib,hashlib,runpy,io,contextlib,json,socket,os', 'import pathlib,hashlib,runpy,io,contextlib,json,socket,os,argparse')
idle = idle.replace("q['node']=='a02r4n01'", "q['node']==args.node ")
idle = idle.replace("p=pathlib.Path(", "ap=argparse.ArgumentParser();ap.add_argument('--node',required=True);ap.add_argument('--owner-tids',required=True);args=ap.parse_args();tids=json.loads(args.owner_tids);assert isinstance(tids,list)and all(type(x)is int and x>0 for x in tids)and len(tids)==len(set(tids))\np=pathlib.Path(", 1)
idle = idle.replace("HTTP43882_bindable=True,GPU_imported=False", "HTTP43882_bindable=True,GPU_imported=False,owner_tids=tids,owner_tids_all_gone=not any(pathlib.Path('/proc/'+str(t)).exists() for t in tids)")
ast.parse(idle)
(OUT / 'idle_r375.py').write_text(idle, encoding='utf8', newline='\n')
print(json.dumps({'completed': True, 'controller_SHA256': hashlib.sha256((OUT / 'controller_375.py').read_bytes()).hexdigest()}))
