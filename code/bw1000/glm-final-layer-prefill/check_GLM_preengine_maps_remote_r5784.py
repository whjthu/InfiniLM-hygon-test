import pathlib,json,hashlib
real=pathlib.Path;private=real('/public/home/aclsylqidf/hygon-inference-refactor-20261014/candidates/glm-layer77-model-r5775/libglm_layer77_model_r5775.so');expected='fb059e2a772afa32fa9617b252fa4d67ee749404a6dd80e3e1ddc03ccf919cf5';assert hashlib.sha256(private.read_bytes()).hexdigest()==expected
lines=real('/proc/self/maps').read_text().splitlines();sample=next(v for v in lines if v.split()and v.split()[-1].startswith('/')and'.so'in real(v.split()[-1]).name);prefix=sample.rsplit(None,1)[0];text=''
class Proxy:
 @staticmethod
 def Path(v):
  if str(v)=='/proc/self/maps':
   class Fixture:
    def read_text(self):return text
   return Fixture()
  return real(v)
pathlib=Proxy
expression="sorted({str(pathlib.Path(v.split()[-1]).resolve()) for v in pathlib.Path('/proc/self/maps').read_text().splitlines() if v.split() and v.split()[-1].startswith('/') and ('.so' in pathlib.Path(v.split()[-1]).name)})"
cases={}
for name,paths,accept in [('actualprivate',[str(private)],True),('privateplusSDK',[str(private),'/opt/dtk/lib/libgalaxyhip.so'],True),('unrelatedonly',['/opt/dtk/lib/libgalaxyhip.so'],False),('samebasenamewrongpath',['/tmp/'+private.name],False),('resolvedalias',[str(private.parent/'../glm-layer77-model-r5775'/private.name)],True),('oldhybridpresent',[str(private),'/refactor/candidates/glm-hybrid-decode-model-r5390/libmodern_glm_full_model_probe.so'],False)]:
 text='\n'.join([prefix+' '+p for p in paths]);mapped=eval(expression)
 ok=str(private.resolve())in mapped and not any('glm-hybrid-decode-model-r5390'in p for p in mapped)
 assert ok==accept,(name,mapped,ok,accept);cases[name]=ok
print(json.dumps({'passed':True,'CPU_only':True,'GPU_dlopen':False,'actualLinuxmaps_format_sample':sample,'actual_private_SHA256':expected,'cases':cases,'source_scope':'EvaluatedactualpreengineASTmappedexpressiononrealLinuxmapsformat, injectedpathnamecases; doesnotclaimtargetDSOloadedinCPUprocess.'}))
