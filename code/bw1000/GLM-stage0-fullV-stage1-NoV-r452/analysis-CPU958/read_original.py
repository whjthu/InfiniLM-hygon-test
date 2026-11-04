"""Stdlib audit of the unique HOST CPU receipt; no execution of producer code."""
from pathlib import Path
import argparse,base64,hashlib,json
SOURCE='2334660a916b3dac92775a5c6a79ec800b74d597266698a2041106a927e7c448'
PROVIDER='04773158b3044b3a4a44b4acfb5c39db97ebccfafc0bdfe9a49c11c1c64d3e11'
def analyze(path):
    path=Path(path);raw=path.read_bytes();observation=json.loads(raw);files={}
    for name,pin in observation['files'].items():
        data=base64.b64decode(pin['base64'],validate=True)
        assert len(data)==pin['bytes']and hashlib.sha256(data).hexdigest()==pin['SHA256'],name
        files[name]=(data,pin)
    receipts=[(name,data,pin)for name,(data,pin)in files.items()if name.endswith('/candidates/GLM-stage0-fullV-stage1-NoV-r452-CPU-v1/result.json')]
    assert len(receipts)==1;name,data,pin=receipts[0];q=json.loads(data)
    assert q['actual_CPU']=='910407.958'and q['completed']is True and q['passed']is True
    assert q['source_manifest_SHA256']==SOURCE and q['provider_manifest_SHA256']==PROVIDER
    assert q['GPU']==0 and q['Native_dlopen']is False and q['weights']is False and q['compile']is False
    assert not q['project_maps']and q['hidden']==dict(CUDA_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1')
    rows=[r.split('|')for r in observation['accounting'].splitlines()if r.startswith('910407.958|')]
    assert len(rows)==1 and rows[0][1:4]==['GLM452-mixed-cache-HOST-CPU-v1','COMPLETED','0:0']and rows[0][-1]=='a14r2n20'
    c=q['new_HOST_consumer'];assert c['passed']and c['source_only']and c['recording_only']and c['actual_worker_CLI_layout_census_ready_collector_new_boundaries']
    assert c['GPU']==0 and c['Native_dlopen']is False and c['weights']is False and c['compile']is False
    assert c['unchanged_CCL_and_memory_admission_not_executed']is True and c['physical_fit_peak_Model_quality_or_performance_proven']is False
    stages=c['stages'];assert len(stages)==2
    for row,stage,bank,vbytes in zip(stages,[0,1],[152,120],[5100273664,0]):
        assert row==dict(stage=stage,all8_actual_configure_layout_census=True,projected_value_bytes=vbytes,exact_bank_count=bank,Native_executed=False)
    expected=['conflicting_constructor_flag_rejected','wrong_stage_V_census_rejected','wrong_V_bytecount_rejected','wrong_TP_ready_policy_rejected','wrong_config_stage_flags_rejected','request_cache_family_switch_rejected']
    assert set(c['negative_checks'])==set(expected)and all(c['negative_checks'].values())
    return dict(passed=True,scope='Actual HOST CPU receipt/accounting and explicit recording new boundaries only',actual_CPU=q['actual_CPU'],accounting=rows[0],source_manifest_SHA256=SOURCE,provider_manifest_SHA256=PROVIDER,raw_receipt=dict(path=name,bytes=len(data),SHA256=pin['SHA256']),observation_SHA256=hashlib.sha256(raw).hexdigest(),verified_raw_files=len(files),stages=stages,negative_checks=c['negative_checks'],Native_or_SDK_executed=False,weights=False,compile=False,GPU=0,CCL_or_real_memory_admission_executed=False,actual_V_bank_allocation_or_peak_or_Model_fit='UNKNOWN',actual_Model_quality_or_performance='UNKNOWN',GPU_GO=False,Model_GO=False)
def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--observation',required=True);p.add_argument('--output');a=p.parse_args(argv);q=analyze(a.observation);raw=(json.dumps(q,ensure_ascii=False,indent=2)+'\n').encode()
    if a.output:
        output=Path(a.output);assert not output.exists();output.write_bytes(raw)
    else:print(raw.decode(),end='')
    return 0
if __name__=='__main__':raise SystemExit(main())
