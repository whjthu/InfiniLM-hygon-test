import pathlib,json,hashlib,subprocess
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');f=r/'candidates/Deep-prefix-native-r128/CPU-build.json';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert sha(f)=='76ddaf145324582ef46e04310f8ce92e949dd222838d00ec5c3145856e6a73f7';q=json.loads(f.read_bytes());assert not q['completed']and len(q['commands'])==29
parent=r/'candidates/deep-c4-smallt-model-r9254/CPU-build.json';p=json.loads(parent.read_bytes());assert sha(parent)=='43d4478102462e26bbfb0784942ec2f8f05228f53fb42eae0b23c051d1ed6d4c'
old=pathlib.Path(p['libraries']['full']['path'].replace('/refactor',str(r),1));new=r/'candidates/Deep-prefix-native-r128/libDeep_prefix_native_r128.so'
assert sha(old)=='ac9370c8f188bfda8555718c52690b914883fc6b02b9267cbd279be3ecdd3835'
def symbols(path):
 s=subprocess.check_output(['nm','-D','--defined-only',str(path)],text=True,timeout=30);return {a[-1]:a[-2]for line in s.splitlines()if len(a:=line.split())>=3}
a,b=symbols(old),symbols(new);missing=sorted(set(a)-set(b));added=sorted(set(b)-set(a));assert len(missing)<500
demangled=subprocess.check_output(['c++filt'],input='\n'.join(missing),text=True,timeout=30).splitlines()
print(json.dumps({'read_only':True,'scope':'nm/c++filt metadata only, no dlopen/compile/relink','failed_receipt_SHA256':sha(f),'new_DSO_SHA256':sha(new),'parent_DSO_SHA256':sha(old),'missing':[{'symbol':n,'type':a[n],'demangled':d}for n,d in zip(missing,demangled)],'added_count':len(added),'missing_C_API':[n for n in missing if not n.startswith('_Z')],'new_required_C_API':{n:n in b for n in ['modern_deep_prefix_begin_r128','modern_deep_prefix_verify_r128','modern_deep_prefix_commit_r128','modern_deep_prefix_cancel_r128','modern_deep_prefix_status_r128']}}))
