"""Load frozen transport only after the independently pinned service manifest."""
import hashlib,importlib.util,json,pathlib
SERVICE_MANIFEST_SHA = '2e48f0fdd66f018908d0fe76d08dab7d2ae7202ae4a82b63eced7a3f26497678'
SERVICE_MANIFEST_NAME = 'service-source-manifest-r275.json'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def load_transport(service, name):
 service=pathlib.Path(service);mp=service/SERVICE_MANIFEST_NAME
 if sha(mp)!=SERVICE_MANIFEST_SHA:raise ValueError('Frozen service manifest mismatch')
 m=json.loads(mp.read_bytes());members=m['files_SHA256'];source=service/'client/client_p0_r143.py';launch=service/'client/client-launch-r223.json'
 if sha(launch)!=members['client/client-launch-r223.json']:raise ValueError('Frozen launchspec mismatch')
 expected=members['client/client_p0_r143.py']
 if json.loads(launch.read_bytes())['p0_SHA256']!=expected:raise ValueError('Transport pin disagrees with pinned launchspec')
 if sha(source)!=expected:raise ValueError('Frozen HTTP transport source mismatch')
 spec=importlib.util.spec_from_file_location(name,source);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
