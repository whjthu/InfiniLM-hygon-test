import pathlib,hashlib,subprocess,json
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';p=r/'incoming/launch_Kimi_A8_ABC_real_r8996.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='eb038eb03791d6b4ec3cea8b0cdb039460740d7902dd34bef389e42996d345c8';z=subprocess.run(['python3',str(p)],capture_output=True,text=True,timeout=60);assert z.returncode==0,z.stderr;print(z.stdout)
