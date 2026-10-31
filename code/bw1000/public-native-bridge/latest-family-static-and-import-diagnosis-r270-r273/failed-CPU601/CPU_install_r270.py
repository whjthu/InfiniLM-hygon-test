"""Install complete public Python and the passed exact family extensions."""
import pathlib,hashlib,json,os,sys,subprocess,signal,time,traceback,argparse
D=pathlib.Path('/refactor/incoming/latest-family-service-r270');R=pathlib.Path('/refactor');B=R/'candidates/latest-family-service-r270';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest-sha',required=True);a=ap.parse_args();mf=D/'SOURCE_MANIFEST.json';assert sha(mf)==a.manifest_sha;m=json.loads(mf.read_bytes())
 for n,h in m['files_SHA256'].items():assert sha(D/n)==h,n
 assert all(os.environ.get(v)==''for v in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']);B.mkdir(exist_ok=False);start=time.monotonic();deadline=start+205
 q={'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'completed':False,'passed':False,'CPU_only':True,'GPU':False,'Native_Model_loaded':False,'weights_opened':False,'new_compile_or_link':False,'actual_extension_import_planned':True,'families':{}}
 try:
  p=R/'candidates/latest-family-native-r269/CPU-build.json';assert sha(p)=='3399988b130529836f85826e912673173831e4dd06429073578949e3c6430fc5';build=json.loads(p.read_bytes());assert build['passed']and build['actual_step']=='910407.600'
  S=R/'incoming/Kimi-latest-service-r267';sm=S/'service-source-manifest-r267.json';assert sha(sm)=='9118cf2606f141719a55ef7f96c718f6c3cd8bc9b2fdce44f7994408af025dc5';source=json.loads(sm.read_bytes());prefix='public/python/infinilm/';public={n[len(prefix):]:h for n,h in source['files_SHA256'].items()if n.startswith(prefix)};assert sum(n.endswith('.py')for n in public)==72
  A=R/'incoming/latest-family-entry-r268';am=A/'SOURCE_MANIFEST.json';assert sha(am)=='213fa034864796990cf9a858aee2d75039eaccf36420298a2d3d5007c633bdb7';entry=json.loads(am.read_bytes());helpers=['native_engine_bridge_r268.py','prepared_chat.py','prepared_history.py','prepared_family_entry_r268.py','service_queue_bridge_r268.py']
  for fam in ['GLM','Deep']:
   q['stage']=fam+'-install';where=B/fam;(where/'python/infinilm').mkdir(parents=True);(where/'runtime').mkdir()
   for n,h in public.items():
    p=S/prefix/n;assert sha(p)==h;dest=where/'python/infinilm'/n;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes());assert sha(dest)==h
   for n in helpers:
    p=A/n;assert sha(p)==entry['files_SHA256'][n];(where/'runtime'/n).write_bytes(p.read_bytes())
   extension=build['families'][fam]['Common_extension'];p=pathlib.Path(extension['new']);assert sha(p)==extension['new_SHA256'];(where/'python'/p.name).symlink_to(p)
   common=R/('candidates/rebased-5f91ad7-r3668/Common/'+fam+'/InfiniLM');retained=common/'python/infinilm/_lib';links={}
   for p in retained.iterdir():
    target=where/'python/infinilm/_lib'/p.name
    if target.exists():assert p.suffix=='.py'and target.is_file(),'Do not replace existing latest public source'
    else:target.symlink_to(p,target_is_directory=p.is_dir());links[str(target)]=str(p.resolve())
   install={'upstream_commit':build['upstream_commit'],'public_source_manifest_SHA256':sha(sm),'source_files_SHA256':public,'public_Python_files':72,'entry_source_manifest_SHA256':sha(am),'helper_SHA256':{n:sha(where/'runtime'/n)for n in helpers},'CPU600_receipt_SHA256':sha(R/'candidates/latest-family-native-r269/CPU-build.json'),'extension_real_path':str(pathlib.Path(extension['new']).resolve()),'extension_SHA256':extension['new_SHA256'],'retained_binary_links':links,'Native_weight_directory_unchanged':True};(where/'installation.json').write_text(json.dumps(install,indent=2)+chr(10));q['families'][fam]={'installation_SHA256':sha(where/'installation.json'),'extension_SHA256':extension['new_SHA256'],'public_Python_files':72}
   q['stage']=fam+'-actual-import';env=os.environ.copy();env['PYTHONPATH']=':'.join([str(where/'runtime'),str(where/'python'),str(common/'python'),'/old/offline-deps']);env['LD_LIBRARY_PATH']=':'.join([str(where/'python'),str(where/'python/infinilm/_lib'),'/refactor/install/hygon-'+fam.lower()+'-5f91ad7-r3672/lib','/refactor/install/hygon-'+fam.lower()+'-5f91ad7-r3672/lib64',env.get('LD_LIBRARY_PATH','')]);child=subprocess.Popen([sys.executable,'-B',str(D/'CPU_actual_entry_r270.py'),fam],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
   try:out=child.communicate(timeout=min(70,deadline-time.monotonic()))[0]
   except BaseException:
    try:os.killpg(child.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:out=child.communicate(timeout=3)[0]
    except subprocess.TimeoutExpired:
     try:os.killpg(child.pid,signal.SIGKILL)
     except ProcessLookupError:pass
     out=child.communicate(timeout=3)[0]
    (where/'actual-import.log').write_text(out);raise RuntimeError('CPU_ACTUAL_IMPORT_BOUNDED_TIMEOUT:'+fam)
   (where/'actual-import.log').write_text(out);rp=where/'CPU-actual-entry.json';q['families'][fam]['actual_import_returncode']=child.returncode
   if rp.is_file():
    q['families'][fam]['actual_import_receipt_SHA256']=sha(rp);result=json.loads(rp.read_bytes());q['families'][fam]['actual_import']=result
   assert child.returncode==0 and rp.is_file()and result['passed'],'CPU_ACTUAL_IMPORT_FAILED:'+fam
  q.update(passed=True,stage='complete',source_manifest_SHA256=a.manifest_sha,CPU600_receipt_SHA256=sha(R/'candidates/latest-family-native-r269/CPU-build.json'),GPU_GO=False)
 except BaseException as exc:q.update(error_type=type(exc).__name__,error=str(exc)[:1600],failure_locations=[{'file':v.filename,'line':v.lineno,'function':v.name}for v in traceback.extract_tb(exc.__traceback__)])
 finally:q['completed']=True;q['elapsed_s']=time.monotonic()-start;(B/'CPU-prepare.json').write_text(json.dumps(q,indent=2)+chr(10));print(json.dumps({k:q.get(k)for k in ['actual_step','completed','passed','stage','elapsed_s','error_type','error']}))
 assert q['passed']
if __name__=='__main__':main()
