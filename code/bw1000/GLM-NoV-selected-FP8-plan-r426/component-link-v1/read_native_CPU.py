"""Independent stdlib-only read of852 originalbytes; no runtime imports."""
from pathlib import Path
import argparse,hashlib,json
def read(observation,source):
 o=json.loads(Path(observation).read_bytes());r=o['receipt'];raw=r['rawtext'].encode();sha=lambda b:hashlib.sha256(b).hexdigest()
 assert sha(raw)==r['SHA256']=='06f607a40a75543ef225b76464212b89307f510ec98a41cf7cb50cc9a9545b62'and len(raw)==r['bytes'];q=json.loads(raw);assert q==r['data']
 assert q['completed']and q['passed']and q['producer']=='910407.852'and q['HOST_TUs_compiled']==2 and q['contract_TUs_compiled']==1 and q['GPU_TUs_compiled']==0 and q['GPU']==0 and not q['Native_dlopen']and not q['SDK_executed']and not q['checkpoint_IO']
 D=Path(source);mf=D/'CPU_SOURCE_MANIFEST.json';assert sha(mf.read_bytes())==q['source_manifest_SHA256']=='398493ad7cc0e6d3cd4bff135934cb25e5b567e1b23768943a2c931dbef78f06'
 for n,h in json.loads(mf.read_bytes())['files_SHA256'].items():assert sha((D/n).read_bytes())==h,n
 assert len(q['reused_objects_SHA256'])==28 and len(q['final_objects_SHA256'])==q['final_objects']==30 and len(q['compiled'])==2
 assert q['original_headers_dependencies_equivalent']and q['all_parent_strong_CABI_retained']and q['ldd_r_parent_equal']and q['class_header_layout_unchanged']and q['helper776_unchanged']and q['default_OFF']and not q['actual_device_or_numeric_qualified']
 c=q['actual_cache_consumer'];assert c['passed']and c['actual_private_cache_logic']and c['recording_776_leaf']and c['cross_layer_invocations']==40 and c['first_sequence_created']==1 and c['first_sequence_hits']==39 and c['logical_plan_scratch_bytes']==20447232 and c['normal_clear_and_error_drain']and c['sticky_failstop']and c['GPU']==c['Native']==0 and not c['SDK_executed']
 assert q['destroy_failure_failstop_RC']in(-6,134)and q['library_SHA256']=='c29e6e5212b51145a7c206df5b2b947b4159771983bb84b5720d4f5419d148a2'
 a=[v.split('|')for v in o['accounting']if v.startswith('910407.852|')];assert len(a)==1 and a[0][2:4]==['COMPLETED','0:0']
 return dict(passed=True,raw_receipt_SHA256=r['SHA256'],source_manifest_SHA256=q['source_manifest_SHA256'],actual_CPU=q['producer'],Slurm=a[0],HOST_TUs=2,contract_TUs=1,actual_shared_cache_logic_consumer=True,external776_recorded_only=True,Native_device_or_numeric_qualified=False,GPU=0,library=q['library'],library_SHA256=q['library_SHA256'],all30objects_and904_current_consumer_headers_pending_link_only_validation=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--observation',required=True);p.add_argument('--source',required=True);a=p.parse_args();print(json.dumps(read(a.observation,a.source),indent=2))
