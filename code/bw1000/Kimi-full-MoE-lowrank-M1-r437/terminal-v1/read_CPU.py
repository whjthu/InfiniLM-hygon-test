"""stdlib audit of original883/885 failures and immutable read-only qualification."""
from pathlib import Path
import argparse,hashlib,json
D=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def receipt(path):
 observation=json.loads(Path(path).read_bytes());item=observation['receipt'];raw=item['rawtext'].encode()
 assert len(raw)==item['bytes']and hashlib.sha256(raw).hexdigest()==item['SHA256']
 return observation,json.loads(raw),item

def main(a):
 obs,q,raw=receipt(a.observed);old_obs,old,old_raw=receipt(a.failed883);link_obs,link,link_raw=receipt(a.failed885)
 assert old_raw['SHA256']=='dfa142029c9fc168cddc3a7cffa133bbbad68a9ebdd8149f433402968680cb65'
 assert link_raw['SHA256']=='8d664b80d9f4d069cfc19200af3da4a8d81205c03dfeaa4ce084c6bef663b84b'
 assert old['actual_CPU']=='910407.883'and not old['passed']and len(old['compiled'])==2
 assert link['actual_CPU']=='910407.885'and not link['passed']and all(v['rc']==0 for v in link['commands'])and len(link['commands'])==12
 assert old['source_reverse_contract']['passed']and old['actual_payload_FFI_and_retained_input']['passed']
 assert old['actual_payload_FFI_and_retained_input']['real_retained_C1_input']['SHA256']=='3f84e8d80178c05212edcc5b703f907290df20925684fb610838c6d1f259ecca'
 assert q['completed']and q['passed']and q['CPU_only']and q['GPU']==0 and not q['Native_dlopen']and not q['Runtime_created']
 assert q['compile']==q['link']==q['objcopy']==q['weights']==0 and q['CPU885_original_FAIL_preserved']
 assert q['source_manifest_SHA256']=='a65c3c989babd94b228bc78a5f2d537d6e343f1352914176988f97fcc3788d5f'
 assert q['library']=='/refactor/candidates/Kimi-full-MoE-lowrank-M1-r437-CPU-link-v2/libKimi_full_MoE_lowrank_M1_r437.so'
 assert q['library_SHA256']=='f5b99922693fe4463a00066379afdb16d259a552c13700163be2975064d6882f'
 assert q['helper_SHA256']=='a171060a1b568b302812f3952bdda144c320b4f28a33fc78db585b7135030b1c'
 assert q['helper_binding']['kind']=='exact_absolute_DT_NEEDED'and q['helper_binding']['path']=='/refactor/candidates/models-5f91ad7-r3675/Kimi/helpers/libkimi_grouped_prefill.so'
 localized=q['localized_private_object'];assert len(localized['only_localized'])==7 and len(localized['all_text_sections_SHA256'])==564
 assert localized['text_bytes_unchanged']and localized['all_other_global_symbols_unchanged']and q['old_strong_CABI_preserved']>0
 for sub,expected in [('', '2f526581acd9ba12815387e2bb476ff4f15265c017344a4210c9f2c14d4ede3a'),('cpu-link-v2','a327e706f473a59f7d6e10566d53e9e85a5470ada8d94cfd2aa8a28205b77ed9'),('cpu-qualify-v3','a65c3c989babd94b228bc78a5f2d537d6e343f1352914176988f97fcc3788d5f')]:
  directory=D/sub;assert sha(directory/'SOURCE_MANIFEST.json')==expected
  for p,h in json.loads((directory/'SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():assert sha(directory/p)==h,p
 actual=q['actual_CPU'];lines=[v.split('|')for v in obs['accounting']['stdout'].splitlines()if v.split('|')[0]==actual];assert len(lines)==1 and lines[0][2:4]==['COMPLETED','0:0']
 result=dict(passed=True,actual_CPU=actual,Slurm='COMPLETED0',CPU_only=True,GPU=0,Native_dlopen=False,Runtime_created=False,weights=0,source_manifest_SHA256=q['source_manifest_SHA256'],receipt_SHA256=raw['SHA256'],receipt_bytes=raw['bytes'],library=q['library'],library_SHA256=q['library_SHA256'],original_883_failed_preserved=True,original_885_failed_preserved=True,actual_new_HOST_compile_total=2,actual_new_GPU_compile_total=0,actual_successful_link_total=1,actual_link_attempts=2,qualifier_compile_link_objcopy=[0,0,0],text_sections_preserved=564,only_clone_C_wrappers_localized=7,old_strong_CABI_preserved=q['old_strong_CABI_preserved'],PP1_padded_scale_boundary=True,actual_Model_numeric_or_TP16_owner_qualification=False,GPU_GO=False,Model_GO=False)
 Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('observed');p.add_argument('--failed883',required=True);p.add_argument('--failed885',required=True);p.add_argument('--output',required=True);main(p.parse_args())
