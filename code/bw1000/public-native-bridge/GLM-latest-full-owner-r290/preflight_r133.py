import pathlib,json,hashlib,os,sys
sys.path.insert(1,'/refactor/incoming')
from native_chain_r186 import verify
def read_private_metadata():return verify()
def validate():
 from GLM_transient_preflight_r9059 import validate as parentvalidate
 q=parentvalidate();q['parent648_model_metadata']=q['model_library_metadata'];q['model_library_metadata']=read_private_metadata();return q
if __name__=='__main__':print(json.dumps(validate()))
