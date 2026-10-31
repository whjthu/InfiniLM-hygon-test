"""Install complete public Python and the passed exact family extensions."""
import pathlib,hashlib,json,os,sys,subprocess,signal,time,traceback,argparse
D=pathlib.Path('/refactor/incoming/latest-family-static-r273');R=pathlib.Path('/refactor');B=R/'candidates/latest-family-service-r270';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest-sha',required=True);a=ap.parse_args();mf=D/'SOURCE_MANIFEST.json';assert sha(mf)==a.manifest_sha;m=json.loads(mf.read_bytes())
 for n,h in m['files_SHA256'].items():assert sha(D/n)==h,n
 assert all(os.environ.get(v)==''for v in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']);assert B.is_dir();assert sha(B/'CPU-prepare.json')=='623114e064c05a85820306c145c098484d8f52e899e496cd1a8390f5b0a69579';assert sha(B/'CPU-prepare-r271.json')=='8321b84b34580cc92ee3bc8ac7f60299f87a57182c810a6d085bf113a0d71586';start=time.monotonic();deadline=start+205
 q={'preserved_failed_CPU601_receipt_SHA256':'623114e064c05a85820306c145c098484d8f52e899e496cd1a8390f5b0a69579','preserved_failed_CPU602_receipt_SHA256':'8321b84b34580cc92ee3bc8ac7f60299f87a57182c810a6d085bf113a0d71586','no_passed_compile_or_frontend_matrix_repeated':True,'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'completed':False,'passed':False,'CPU_only':True,'GPU':False,'Native_Model_loaded':False,'weights_opened':False,'new_compile_or_link':False,'actual_extension_import':False,'static_only':True,'Torch_imported':False,'Native_dlopen':False,'actual_GPU_import_PENDING':True,'families':{}}
 try:
  p=R/'candidates/latest-family-native-r269/CPU-build.json';assert sha(p)=='3399988b130529836f85826e912673173831e4dd06429073578949e3c6430fc5';build=json.loads(p.read_bytes());assert build['passed']and build['actual_step']=='910407.600'
  S=R/'incoming/Kimi-latest-service-r267';sm=S/'service-source-manifest-r267.json';assert sha(sm)=='9118cf2606f141719a55ef7f96c718f6c3cd8bc9b2fdce44f7994408af025dc5';source=json.loads(sm.read_bytes());prefix='public/python/infinilm/';public={n[len(prefix):]:h for n,h in source['files_SHA256'].items()if n.startswith(prefix)};assert sum(n.endswith('.py')for n in public)==72
  A=R/'incoming/latest-family-entry-r268';am=A/'SOURCE_MANIFEST.json';assert sha(am)=='213fa034864796990cf9a858aee2d75039eaccf36420298a2d3d5007c633bdb7';entry=json.loads(am.read_bytes());helpers=['native_engine_bridge_r268.py','prepared_chat.py','prepared_history.py','prepared_family_entry_r268.py','service_queue_bridge_r268.py']
  for fam in ['GLM','Deep']:
   q['stage']=fam+'-install';where=B/fam;(where/'python/infinilm').mkdir(parents=True,exist_ok=True);(where/'runtime').mkdir(exist_ok=True)
   for n,h in public.items():
    p=S/prefix/n;assert sha(p)==h;dest=where/'python/infinilm'/n;dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists() or sha(dest)==h;
    if not dest.exists():dest.write_bytes(p.read_bytes())
    assert sha(dest)==h
   for n in helpers:
    p=A/n;assert sha(p)==entry['files_SHA256'][n];dest=where/'runtime'/n;assert not dest.exists()or sha(dest)==sha(p);
    if not dest.exists():dest.write_bytes(p.read_bytes())
   extension=build['families'][fam]['Common_extension'];p=pathlib.Path(extension['new']);assert sha(p)==extension['new_SHA256'];target=where/'python'/p.name;assert not target.exists()or target.resolve()==p.resolve();
   if not target.exists():target.symlink_to(p)
   common=R/('candidates/rebased-5f91ad7-r3668/Common/'+fam+'/InfiniLM');retained=common/'python/infinilm/_lib';links={}
   for p in retained.iterdir():
    target=where/'python/infinilm/_lib'/p.name
    if target.is_symlink():
     assert target.resolve()==p.resolve();links[str(target)]=str(p.resolve())
    elif target.exists():assert p.suffix=='.py'and target.is_file(),'Do not replace existing latest public source'
    else:target.symlink_to(p,target_is_directory=p.is_dir());links[str(target)]=str(p.resolve())
   install={'upstream_commit':build['upstream_commit'],'public_source_manifest_SHA256':sha(sm),'source_files_SHA256':public,'public_Python_files':72,'entry_source_manifest_SHA256':sha(am),'helper_SHA256':{n:sha(where/'runtime'/n)for n in helpers},'CPU600_receipt_SHA256':sha(R/'candidates/latest-family-native-r269/CPU-build.json'),'extension_real_path':str(pathlib.Path(extension['new']).resolve()),'extension_SHA256':extension['new_SHA256'],'retained_binary_links':links,'Native_weight_directory_unchanged':True};ip=where/'installation.json';data=json.dumps(install,indent=2)+chr(10);assert not ip.exists()or ip.read_text()==data;
   if not ip.exists():ip.write_text(data)
   q['families'][fam]={'installation_SHA256':sha(where/'installation.json'),'extension_SHA256':extension['new_SHA256'],'public_Python_files':72}
  q.update(passed=True,stage='complete',source_manifest_SHA256=a.manifest_sha,CPU600_receipt_SHA256=sha(R/'candidates/latest-family-native-r269/CPU-build.json'),GPU_GO=False)
 except BaseException as exc:q.update(error_type=type(exc).__name__,error=str(exc)[:1600],failure_locations=[{'file':v.filename,'line':v.lineno,'function':v.name}for v in traceback.extract_tb(exc.__traceback__)])
 finally:q['completed']=True;q['elapsed_s']=time.monotonic()-start;(B/'CPU-static-r273.json').write_text(json.dumps(q,indent=2)+chr(10));print(json.dumps({k:q.get(k)for k in ['actual_step','completed','passed','stage','elapsed_s','error_type','error']}))
 assert q['passed']
if __name__=='__main__':main()
