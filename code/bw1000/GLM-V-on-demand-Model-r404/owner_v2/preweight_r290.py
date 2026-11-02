"""Latest actual Python/CPP identity in the necessary assigned-GPU model process."""
import pathlib,json,hashlib,os
from source_pin_r290 import verify,profiles,sha,D
from deadline_r290 import check
R=pathlib.Path('/refactor')

def validate():
 verify();profiles(os.environ['GLM_SOURCE_MANIFEST_SHA']);check('before GPU initialization',120)
 install=R/'candidates/latest-family-service-r270/GLM/installation.json';assert sha(install)=='bf9bdef51cfcab59cdc72c59b7f371fbfb822153dde42eee1f5ce5e1942e2b15';ins=json.loads(install.read_bytes());public=R/'candidates/latest-family-service-r270/GLM/python'
 for name,h in ins['source_files_SHA256'].items():assert sha(public/'infinilm'/name)==h
 assert sum(n.endswith('.py')for n in ins['source_files_SHA256'])==72
 import torch
 assert torch.cuda.device_count()==8;torch.cuda.set_device(0)
 import infinilm,_infinilm
 assert pathlib.Path(infinilm.__file__).resolve()==(public/'infinilm/__init__.py').resolve()
 assert pathlib.Path(_infinilm.__file__).resolve()==pathlib.Path(ins['extension_real_path']).resolve()and sha(_infinilm.__file__)==ins['extension_SHA256']=='16910b4d649f81ba3b63eea9e5bc7c71be5e698c06e00cc669b4ad4d0ec88b1a'
 from prepared_family_entry_r268 import prepare_app
 prepared=prepare_app('GLM','/old/models/ZhipuAI-GLM-5.3')
 assert prepared['identity']['upstream_commit']=='99f5c90a40f9b2546bd0202cba3397ed1d55c5a0'and prepared['identity']['eos_token_ids']==[154820,154827,154829]
 proof={'actual_latest_root_imported':True,'source_manifest_SHA256':sha(D/'SOURCE_MANIFEST.json'),'Common_extension_SHA256':ins['extension_SHA256'],'installation_SHA256':sha(install),'weights_opened':False,'Native_constructed':False,'actual_Model_or_ABI_qualified':False,'prepared_entry_identity':prepared['identity']}
 return dict(proof=proof,prepared=prepared)
