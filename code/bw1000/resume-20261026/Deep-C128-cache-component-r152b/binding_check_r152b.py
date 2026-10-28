"""Pure stdlib load-origin check; an unused CCL DSO need not be loaded."""
import pathlib
REQUIRED_NAMES=frozenset(['libinfiniops.so','libinfinirt.so','libinfinilm_runtime.so','_infinilm.cpython-310-x86_64-linux-gnu.so'])
def verify_bindings(expected,loaded,hasher):
 loaded={pathlib.Path(x).resolve()for x in loaded};rows=[]
 expected_names={pathlib.Path(p).name for p in expected};assert REQUIRED_NAMES<=expected_names,'CPU receipt lacks actual required operator/runtime pins'
 for p,pin in expected.items():
  want=pathlib.Path(p).resolve();required=want.name in REQUIRED_NAMES;stem=want.name.split('.so')[0];actual={x for x in loaded if stem in x.name};row={'expected_path':str(want),'required':required,'expected_SHA256':pin['SHA256'],'actual_paths':sorted(str(x)for x in actual)}
  if not actual and not required:row['status']='not_required_absent';rows.append(row);continue
  assert actual=={want},('required_missing_or_wrong_loaded_origin',row)
  got=hasher(want);row['actual_SHA256']=got;assert got==pin['SHA256'],('loaded_DSO_SHA256_mismatch',row);row['status']='required_loaded_pinned'if required else'optional_loaded_pinned';rows.append(row)
 return rows
