import pathlib,json,hashlib,os,sys,time,subprocess,socket
r=pathlib.Path('/refactor');tag=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];source=pathlib.Path('/public/home/aclsylqidf/shared/Kimi-K3');name='model-00011-of-000096.safetensors';f=source/name;target=pathlib.Path('/work2/aclsylqidf-hygon-inference-r7302/raw-shards');out={'step':tag,'node':socket.gethostname(),'completed':False,'CPU_only':True,'GPU_or_Model_load':False,'single_shard_bytes':16990916912,'source_cache':'Uncontrolled; no physical cold claim','not_a_complete_model_directory':True}
stat=lambda p:tuple(getattr(p.stat(),k)for k in ['st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns'])
path=r/'results'/('r7306-raw-cache-'+tag+'.json');assert not path.exists()
try:
 before=stat(f);assert before[2]==16990916912;out['source_stat_before']=before
 p=r/'incoming/r7302_raw_shard_cache.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='d6d38d9aa39392c4bb6d617dd3701c6d3570f490f496bebc1824b8a1c11d1d0a'
 loader=r/'candidates/InfiniLoader-88d6bac-r3599';assert subprocess.check_output(['git','-C',str(loader),'rev-parse','HEAD'],text=True).strip()=='88d6bac6a80a424201da1688989d1d87e3217a93'
 env=dict(os.environ,PYTHONPATH=str(loader/'src'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',HIP_VISIBLE_DEVICES='',CUDA_VISIBLE_DEVICES='')
 result=subprocess.run([sys.executable,'-B',str(p),'--source',str(source),'--cache-root',str(target),'--shard',name,'--apply'],env=env,capture_output=True,text=True,timeout=600);out['API_exit']=result.returncode;out['API_stderr']=result.stderr[-2000:];assert result.returncode==0,result.stderr
 q=json.loads(result.stdout);out['API']=q;assert q['verify']['valid']and q['verify']['checked_files']==1
 manifest=json.loads((pathlib.Path(q['result']['cache_path'])/'.iloader-cache/manifest.json').read_text());assert manifest['file_count']==1 and manifest['cached_files'][0]['relative_path']==name
 out['source_streaming_SHA256']=manifest['cached_files'][0]['sha256'];out['cache_manifest_SHA256']=manifest['cached_files'][0]['sha256'];out['source_hash_scope']='Computed during original-source copy by InfiniLoader; full destination verify checks this digest. No additional full source reread.'
 after=stat(f);out['source_stat_after']=after;assert after==before
 out.update(completed=True,source_unchanged=True,raw_bytes_identical=True,first_copy_measured=not q['result']['reused'],scope='One typical16.991GB originalHFshard rawcachecopy/byteintegrity; copy includes source streamingSHA, verify destination reread separately charged; no extra sourcehash reread. No physicalcold/Modelgain/fullmodelview/CPUresident/memfd claim.')
except BaseException as e:out['error']=repr(e);raise
finally:
 t=path.with_suffix('.tmp');t.write_text(json.dumps(out,indent=2));t.replace(path);print(json.dumps({'completed':out['completed'],'step':tag,'error':out.get('error')}))
