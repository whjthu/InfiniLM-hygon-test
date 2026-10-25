import pathlib,json,hashlib,runpy,subprocess
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';m=r/'incoming/Deep-Cast-Model-manifest-r9060.json';assert hashlib.sha256(m.read_bytes()).hexdigest()=='4063c5f4276a797a9268bffa1a374717e1f275a5b58fd7c4aea43ed06ed5cf1a'
rows=[s.split('|')for s in subprocess.check_output(['sacct','-S','2026-10-25','-j','910407.269','-n','-P','-o','JobID,State%30,ExitCode'],text=True,timeout=30).splitlines()];assert any(s[:3]==['910407.269','COMPLETED','0:0']for s in rows)
p=r/'logs/Deep-Cast-frontend-CPU-r9060-910407.269.log';q=[json.loads(s)for s in p.read_text().splitlines()if s.startswith('{')];assert len(q)==1 and q[0]['passed']and q[0]['no_GPU']and q[0]['manifest_SHA256']==hashlib.sha256(m.read_bytes()).hexdigest()
subprocess.run(['bash','-n',str(r/'incoming/Deep_Cast_Model_r9060_inside.sh')],check=True)
runpy.run_path(str(r/'incoming/launch_Deep_Cast_Model_r9060.py'),run_name='__main__')
