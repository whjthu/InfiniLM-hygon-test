import pathlib,json,hashlib,ast,subprocess,sys
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'incoming/Deep-ordinary-shape-r148';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();m=json.loads((D/'manifest-r148.json').read_bytes());q={'CPU_only':True,'GPU':False,'Torch_imported':False,'Native_dlopen':False,'source_manifest_SHA256':sha(D/'manifest-r148.json')}
for n,h in m['source_SHA256'].items():
 assert sha(D/n)==h,n
 if n.endswith('.py'):ast.parse((D/n).read_bytes())
cp=R/'candidates/Deep-prefix-device-scope-r135/CPU-build.json';assert sha(cp)=='4a151a66a5149ad9b423b87c2a74e03e40bcaa3a1094bdb6eddad117b4fe9bf0';c=json.loads(cp.read_bytes());assert c['completed']and c['CPU_only']and not c['GPU_initialized']and not c['dlopen'];map_path=lambda p:pathlib.Path(str(p).replace('/refactor',str(R),1));assert sha(map_path(c['library']))=='f100e038c6e883e0210380724d7492810cbad1c20384ea4cd162dca4ca2326b2'
for n,pin in c['actual_DTK_bindings'].items():assert sha(map_path(n))==pin['SHA256'],n
assert 'torch'not in sys.modules;q.update(completed=True,reused_actual_CPU_receipt_SHA256=sha(cp),DSO_SHA256=c['library_SHA256'],source_AST_passed=True,dependency_pin_count=len(c['actual_DTK_bindings']),new_Native_compile=False);print(json.dumps(q))
