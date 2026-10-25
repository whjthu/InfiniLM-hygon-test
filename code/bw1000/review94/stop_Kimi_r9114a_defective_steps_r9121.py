import subprocess,json,datetime
ids=['910375.711','910374.1045'];q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Only own two defective r9114a execution steps; never parent allocations','reason':'Independent source verification: state-only replay does not crop InferEngine kimi_native_lengths_; next partial path rejects history. Both linked infer_engine.o dependency and actual freeze/commit source checked.'}
raw=subprocess.check_output(['sacct','-S','2026-10-25','-j',','.join(ids),'-n','-P','-o','JobIDRaw,JobName%120,User,State,ExitCode'],text=True,timeout=30);q['before']=raw.splitlines();rows=[v.split('|')for v in raw.splitlines()if v.split('|')[0]in ids];assert {v[0]for v in rows}==set(ids),('missingexactstep',rows)
q['parent_ownership']={}
for parent in ['910375','910374']:
 info=subprocess.check_output(['scontrol','show','job','-o',parent],text=True,timeout=30);assert 'UserId=aclsylqidf(' in info,('parentowner',parent);q['parent_ownership'][parent]='aclsylqidf'
for row in rows:assert row[2] in ('','aclsylqidf') and ('state' in row[1].lower() or 'r9114' in row[1].lower()),('unexpectedowner/jobname',row)
active=[v[0]for v in rows if v[3]in ('RUNNING','PENDING','COMPLETING','CONFIGURING')]
if active:
 p=subprocess.run(['scancel',*active],capture_output=True,text=True,timeout=30);q['cancel']={'ids':active,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr};assert p.returncode==0
else:q['cancel']={'ids':[],'reason':'alreadyterminal,noaction'}
print(json.dumps(q))
