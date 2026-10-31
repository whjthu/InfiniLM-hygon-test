"""Compile actual production geometry and arming functions; no fake GPU runtime."""
from pathlib import Path
import hashlib,json,subprocess,sys,shutil,tempfile,re,datetime
D=Path(__file__).resolve().parent;G=D.parents[2];N=G/'local-only/Kimi-long-feature-capture-r305/native';O=G/'local-only/Kimi-append512-host-candidate-r247/native'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def function(text,marker):
 b=text.index(marker);i=text.index('{',b);depth=1;j=i+1
 while depth:depth+=(text[j]=='{')-(text[j]=='}');j+=1
 return text[b:j]
def run():
 old=(O/'batch_live_taps_r9122.hpp').read_bytes();new=(N/'batch_live_taps_r9122.hpp').read_bytes();marker=b'extern "C" int modern_kimi_batch_taps_begin_r9122';assert new.endswith(old[old.index(marker):])
 assert (N/'pair_probe.cpp').read_bytes()==(O/'pair_probe.cpp').read_bytes()+b'\n'+(D/'long_capture_wrapper_r305.inc').read_bytes()
 text=new.decode();decl=text[text.index('namespace kimi_batch_live_r9122 {'):text.index('bool representative(')]+'}\n';arm=function(text,'int begin_long_validated_r305(');legacy=function(text,marker.decode())
 source=(D/'CPU_geometry.cpp.in').read_text();source=source.replace('/*ACTUAL_METADATA_DECLARATIONS*/',decl).replace('/*ACTUAL_LONG_ARM_FUNCTION*/',arm).replace('/*ACTUAL_LEGACY_BEGIN_FUNCTION*/',legacy)
 original=(O/'pair_probe.cpp').read_text();formula='int32_t page=1+(rows[i]-1)*int32_t(c->pages_per_row_r4974);for(size_t j=0;j<c->pages_per_row_r4974;++j)table[i*c->pages_per_row_r4974+j]=page+int32_t(j);';assert formula in original
 actualformula=formula.replace('rows[i]','row').replace('c->pages_per_row_r4974','count').replace('i*count+j','j');source=source.replace('/*ACTUAL_PAGE_TABLE_FORMULA*/',actualformula)
 assert 'Shape{C,4}'not in text and 'bb(C*4)'not in text and 'blocks->data(),C*16'not in text
 compiler=shutil.which('g++');assert compiler
 with tempfile.TemporaryDirectory(prefix='kimi-long-capture-')as td:
  p=Path(td);(p/'long_capture_geometry_r305.hpp').write_bytes((D/'long_capture_geometry_r305.hpp').read_bytes());(p/'test.cpp').write_bytes(source.encode());exe=p/('test.exe'if sys.platform=='win32'else'test')
  z=subprocess.run([compiler,'-std=c++17','-O2','-pthread',str(p/'test.cpp'),'-o',str(exe)],capture_output=True,text=True,timeout=45,cwd=p);assert z.returncode==0,z.stderr[-2500:]
  result=subprocess.run([str(exe)],capture_output=True,text=True,timeout=15,cwd=p);assert result.returncode==0,(result.stdout,result.stderr);tests=json.loads(result.stdout)
 return {'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_only':True,'tests':tests,'actual_production_geometry_header_SHA256':sha(D/'long_capture_geometry_r305.hpp'),'actual_legacy_begin_fragment_SHA256':hashlib.sha256(legacy.encode()).hexdigest(),'actual_long_arm_fragment_SHA256':hashlib.sha256(arm.encode()).hexdigest(),'actual_original_page_formula_SHA256':hashlib.sha256(formula.encode()).hexdigest(),'old_CAPI_suffix_byte_identical':True,'pair_probe_old_body_byte_identical':True,'whole_trace_HOST_compiled':False,'whole_pair_HOST_compiled':False,'scope':'Production pure geometry validator and actual arm functions compiled/executed; real Context wrapper only source checked here, full HOST build pending','GPU':0,'Native_dlopen':0,'Model':0,'weights':0,'HTTP':0,'GPU_GO':False}
if __name__=='__main__':
 q=run();p=Path(sys.argv[1]);assert not p.exists();p.write_bytes((json.dumps(q,indent=2)+'\n').encode());print(json.dumps(q,indent=2))
