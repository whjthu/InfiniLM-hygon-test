import pathlib,json,hashlib,shutil,subprocess,sys,datetime
r=pathlib.Path.cwd();repo=r/'InfiniLM-hygon-test';g=['git','-c','safe.directory='+repo.as_posix(),'-C',str(repo)];sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert not subprocess.check_output(g+['diff','--cached','--name-only'],text=True).strip();dst=repo/'code/bw1000/kimi-w13-a8-abi';changed=[]
names=['finalize_Kimi957_r8994.py','read_Kimi_A8_ABI_terminal_hashes_r8994.py','Kimi957-terminal-hashes-r8994.json','Kimi-A8-ABI-real-watch-a-r8994.json','Kimi957-producers-r8994.json','inspect_Kimi957_ISA_r8995.py','Kimi957-ISA-r8995.json','read_Kimi957_ISA_lookup_r8995.py','Kimi957-ISA-lookup-r8995.json','publish_Kimi957_result_r8995.py']
for n in names:shutil.copyfile(r/n,dst/n);changed.append((dst/n).relative_to(repo).as_posix())
result=repo/'results/bw1000/2026-10-25/kimi957-w13-a8-real-negative.json';shutil.copyfile(r/'Kimi957-W13-A8-real-component-compact-r8994.json',result);changed.append(result.relative_to(repo).as_posix())
task=repo/'coordination/tasks/kimi-w13-a8-abi.json';q=json.loads(task.read_bytes());q.update(state='NEGATIVE_STOP_PARK_DEFAULT_OFF',updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_handle='910374.957',terminal='COMPLETED0',result=result.relative_to(repo).as_posix(),result_SHA256=sha(result),normal_owner_close=True,post_KFD='910374.958 n01 empty',next_action='STOP; no Model, rerun, grid, or new nativebuild');task.write_bytes((json.dumps(q,ensure_ascii=False,indent=2)+'\n').encode());changed.append(task.relative_to(repo).as_posix())
mf=repo/'manifests/bw1000/recent-source-files.json';q=json.loads(mf.read_bytes());known={x['path']for x in q['files']}
for n in changed:
 if n.startswith('code/')and n not in known:q['files'].append({'path':n,'source_name':pathlib.Path(n).name,'SHA256':sha(repo/n),'bytes':(repo/n).stat().st_size})
mf.write_bytes((json.dumps(q,ensure_ascii=False,indent=2)+'\n').encode());changed.append(mf.relative_to(repo).as_posix())
pf=repo/'manifests/bw1000/profile-evidence.json';q=json.loads(pf.read_bytes());n=result.relative_to(repo).as_posix();assert not any(x['path']==n for x in q['files']);q['files'].append({'path':n,'SHA256':sha(result),'scope':'957 complete real routed W13A8 negative; numericCPU raw exact, slower fullprovider, notModelgain'})
for x in q['files']:
 if x['path']==mf.relative_to(repo).as_posix():x['SHA256']=sha(mf)
pf.write_bytes((json.dumps(q,ensure_ascii=False,indent=2)+'\n').encode());changed.append(pf.relative_to(repo).as_posix())
subprocess.run([sys.executable,str(repo/'tools/verify_repository.py')],check=True);subprocess.run(g+['add','--',*changed],check=True);assert sorted(subprocess.check_output(g+['diff','--cached','--name-only'],text=True).splitlines())==sorted(changed)
subprocess.run(g+['-c','user.name=whjthu','-c','user.email=haojie0429@gmail.com','commit','-m','Record negative real-routed INT8 W13 result'],check=True);print(subprocess.check_output(g+['log','-1','--format=%H%nAuthor:%an <%ae>%nCommitter:%cn <%ce>'],text=True))
