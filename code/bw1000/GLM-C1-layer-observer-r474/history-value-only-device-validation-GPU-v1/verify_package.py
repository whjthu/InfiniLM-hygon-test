"""Local source bytes/schema only; no Native import or target execution."""
import ast,base64,hashlib,json,re
from pathlib import Path
H=lambda b:hashlib.sha256(b).hexdigest()
def verify(directory):
    d=Path(directory);mf=json.loads((d/'SOURCE_FILES.json').read_bytes())
    assert {p.name for p in d.iterdir()}==set(mf)|{'SOURCE_FILES.json'}
    for name,pin in mf.items():
        p=d/name;assert Path(name).name==name and not p.is_symlink() and p.is_file()
        raw=p.read_bytes();assert len(raw)==pin['bytes'] and H(raw)==pin['SHA256']
        if name.endswith('.py'):ast.parse(raw,filename=name)
    c=json.loads((d/'CLAIM.json').read_bytes());e=json.loads((d/'CPU_EVIDENCE.json').read_bytes());proof=json.loads((d/'SOURCE_PROOF.json').read_bytes())
    assert c['max_invocations']==1 and c['parent']=='1103606' and not c['Model_GO'] and not c['Native_compile_GO'] and not c['CPU_compile_GO']
    assert (c['whole_epoch_seconds'],c['worker_from_origin_seconds'],c['GPU_from_origin_seconds'],c['keeper_from_origin_seconds'],c['post_from_origin_seconds'])==(660,420,480,570,650)
    assert c['ROOT_GO']==c['GPU_GO']==c['device_validation_GO']
    assert c['resources']['GPU']==dict(node='f13r1n05',CPU=4,memory_MiB=8192,physical_GPU_cards=1)
    assert c['resources']['keeper']==dict(node='f13r1n07',CPU=2,memory_MiB=1024,GRES=0)
    assert c['planned']==dict(warm_pairs=2,declared_pairs=3,orders=['AB','BA','AB'],handle_rejections=24,input_guards=10,redzone_detector_checks=6,normal_close=1,weight_parameters=10,weight_range_bytes=64377280)
    assert e['qualified'] and e['CPU_step']==c['required_CPU_step']=='1103606.182'
    assert e['DSO']['SHA256']==c['current_DSO_SHA256']=='f8b74a42b5aa3baa531df0734a34daf6785344882f794e1cc1fb67211e0556bc'
    assert e['corrected_CPP_SHA256']==proof['CPU_v3_corrected_CPP']['SHA256']=='2e62e6afbe17256dddb13d0df92ef47cc4ae420860a35fd02a0f59ca6991071c'
    pin=e['independent_static_evidence'];raw=base64.b64decode(pin['raw_b64'],validate=True);assert len(raw)==pin['bytes'] and H(raw)==pin['SHA256'];s=json.loads(raw)
    assert s['passed'] and s['CPU_static_qualified'] and s['original_collector_passed'] is False and s['original_collector_build'] is None
    assert s['files']['build-result.json']['SHA256']==e['build']['SHA256'] and s['library']['SHA256']==e['DSO']['SHA256']
    for name,pin in proof['byte_equal_copies'].items():assert H((d/name).read_bytes())==pin['SHA256'] and (d/name).stat().st_size==pin['bytes']
    for name,key in [('CPU_EVIDENCE.json','CPU_evidence'),('RUNTIME_PINS.json','runtime_pins')]:assert H((d/name).read_bytes())==c[key]['SHA256']
    assert H((d/'QUALIFIED_PARAMETER_PINS.json').read_bytes())=='6ac2d255d0e0044ec360f063414080588392f4a2a557d8361d4eda9aeba874c6'
    for name in ['ROOT_GO','GPU_GO','device_validation_GO']:assert type(c[name]) is bool
    return dict(source_schema_passed=True,source_files=len(mf)+1,ROOT_GO=c['ROOT_GO'],actual_device_calls=0)
