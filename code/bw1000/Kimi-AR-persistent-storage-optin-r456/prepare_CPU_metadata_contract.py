"""Extract the actual metadata acceptance/copy implementation, not a rewrite."""
from pathlib import Path
import hashlib,json
def prepare(source,output):
 s=Path(source).read_text();i=s.index('inline rt::Tensor freeze_cpu(');j=s.rindex('\n}')
 body=s[i:j];assert 'freeze_input('in body and 'ar_storage_r456' in body
 out='#pragma once\n#include "recording_cpu_tensor_leaf.hpp"\n#include "ar_storage_r456.hpp"\n#include "long_agent_geometry_r179.hpp"\nnamespace infinilm::models::kimi_native {\n'+body+'\n}\n'
 Path(output).write_bytes(out.encode());return dict(production_header_SHA256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),actual_template_body_SHA256=hashlib.sha256(body.encode()).hexdigest(),generated_header_SHA256=hashlib.sha256(out.encode()).hexdigest(),Tensor_leaf='recording CPU metadata storage, explicitly not actual SDK',Native_dlopen=False,Runtime_created=False,GPU=0)
if __name__=='__main__':
 import sys
 print(json.dumps(prepare(sys.argv[1],sys.argv[2])))
