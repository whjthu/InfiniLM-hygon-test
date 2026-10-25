import subprocess,json,datetime
parents=['910371','910373','910374','910375','910376','910377','910379','910380','910381','910406','910407']
p=subprocess.run(['sacct','-S','2026-10-25','-j',','.join(parents),'-n','-P','-o','JobID,JobName%100,State%30,ExitCode,Elapsed,NodeList%100'],capture_output=True,text=True,timeout=40);assert p.returncode==0,p.stderr
rows=[x.split('|')for x in p.stdout.splitlines()if x];active=[x for x in rows if x[2]in ['RUNNING','PENDING','COMPLETING','CONFIGURING']];print(json.dumps({'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protected_parent_910384_not_queried':True,'active':active,'recent_step_terminal':rows[-20:]}))
