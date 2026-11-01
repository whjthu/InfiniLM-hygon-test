"""Retain677 PASS; include all nested original source inventory in new pin."""
import pathlib
D=pathlib.Path(__file__).parent
for n in ['prepare_owner.py','prepare_controller.py','check_controller_CPU.py']:
 p=D/n;s=p.read_text();p.write_bytes(s.replace('runtime_v2/owner','runtime_v4/owner').encode())
s=(D/'freeze_CPU_package_v3.py').read_text()
s=s.replace('runtime_v3','runtime_v4').replace('whole-glue-v3','whole-glue-v4').replace('remote_upload_whole_v3.py','remote_upload_whole_v4.py')
s=s.replace("p.name:sha(p) for p in sorted(O.iterdir())", "p.relative_to(O).as_posix():sha(p) for p in sorted(O.rglob('*'))")
s=s.replace("'owner/'+p.name:base64.b64encode(p.read_bytes()).decode() for p in O.iterdir()", "'owner/'+p.relative_to(O).as_posix():base64.b64encode(p.read_bytes()).decode() for p in O.rglob('*')")
(D/'freeze_CPU_package_v4.py').write_bytes(s.encode())
