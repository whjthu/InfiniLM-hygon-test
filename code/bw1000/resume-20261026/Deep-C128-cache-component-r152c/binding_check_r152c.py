"""Required project DSOs derive from actual wrapper ELF + CPU492 ldd closure."""
import pathlib,re,posixpath
def derive_required(expected,ldd_r,dt_needed):
 assert ldd_r['returncode']==0 and 'not found'not in ldd_r['stdout']+ldd_r['stderr']and 'undefined symbol:'not in ldd_r['stdout']+ldd_r['stderr'],'CPU dynamic closure failed'
 resolved={}
 for line in ldd_r['stdout'].splitlines():
  match=re.match(r'^\s*(\S+)\s+=>\s+(/\S+)\s+\(',line)
  if match:resolved[match[1]]=match[2]
 pins={pathlib.PurePosixPath(p).name:p for p in expected};required=set(pins)&set(resolved);direct=set(pins)&set(dt_needed)
 assert required and direct<=required,'Actual direct project DT_NEEDED must resolve in CPU closure'
 for name in required:assert posixpath.normpath(pins[name])==posixpath.normpath(resolved[name]),('CPU_pinned_origin_differs_from_ldd',name,pins[name],resolved[name])
 return frozenset(required),{'derivation':'Pinned project DSOs intersect actual CPU492 transitive ldd closure; actual wrapper project DT_NEEDED must be a subset. No model-dependency assumption.','wrapper_DT_NEEDED':dt_needed,'required_project_names':sorted(required),'direct_project_names':sorted(direct),'not_required_project_names':sorted(set(pins)-required),'actual_CPU_resolutions':{n:resolved[n]for n in sorted(required)}}
def verify_bindings(expected,loaded,hasher,required_names):
 loaded={pathlib.Path(x).resolve()for x in loaded};rows=[]
 assert set(required_names)<={pathlib.Path(p).name for p in expected},'Missing required project pins'
 for p,pin in expected.items():
  want=pathlib.Path(p).resolve();required=want.name in required_names;stem=want.name.split('.so')[0];actual={x for x in loaded if stem in x.name};row={'expected_path':str(want),'required':required,'expected_SHA256':pin['SHA256'],'actual_paths':sorted(str(x)for x in actual)}
  if not actual and not required:row['status']='not_required_absent';rows.append(row);continue
  assert actual=={want},('required_missing_or_wrong_loaded_origin',row)
  got=hasher(want);row['actual_SHA256']=got;assert got==pin['SHA256'],('loaded_DSO_SHA256_mismatch',row);row['status']='required_loaded_pinned'if required else'optional_loaded_pinned';rows.append(row)
 return rows
