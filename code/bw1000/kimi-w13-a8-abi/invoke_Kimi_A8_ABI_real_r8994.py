import pathlib,hashlib,subprocess,json
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';p=r/'incoming/launch_Kimi_A8_ABI_real_r8994.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='180e7c10db9508975e7c1efb5abb462967f46ee246fab11c5d3fc99ddbfae7a4';z=subprocess.run(['python3',str(p)],capture_output=True,text=True,timeout=60);assert z.returncode==0,z.stderr;print(z.stdout)
