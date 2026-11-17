"""Build recipe data only. This module never starts preprocessing or compilation."""
import hashlib,json,posixpath
from pathlib import Path
PLAN_SHA='d5356d5b2bc60825537d544ee9f1d100e659f0569eb2521177a466acc7a82fea'
PARENT='/refactor/results/GLM474-history-value-only-common-AB-CPU-v1'
def recipe(plan_bytes,source_path,source_sha,output_dir):
    if hashlib.sha256(plan_bytes).hexdigest()!=PLAN_SHA:raise ValueError('original179 PLAN differs')
    p=json.loads(plan_bytes);mirror=PARENT+'/B/project-mirror';bso=PARENT+'/B/libGLM_history_value_B.so'
    for s in (source_path,output_dir):
        if not s.startswith('/refactor/results/') or '..' in Path(s).parts:raise ValueError('owned new probe path')
    helper=mirror+p['helper_path'][len('/refactor'):]
    inc=[posixpath.dirname(helper)]+[(mirror+x[len('/refactor'):] if x.startswith('/refactor/') else x) for x in p['include_roots']]
    prefix=['/usr/bin/g++']+p['new_common_flags']['HOST']+['-I'+x for x in inc]
    obj=output_dir+'/component.o';before=output_dir+'/component.before.d';actual=output_dir+'/component.actual.d';dso=output_dir+'/libhistory_value_component.so'
    link=[x for x in p['original169']['link_argv'] if x not in p['original169']['object_order']]
    link[1:1]=[obj,bso,'/lib/x86_64-linux-gnu/libpthread.so.0'];link[link.index('-o')+1]=dso;link+=['-Wl,-rpath,'+posixpath.dirname(bso)]
    allowed=dict(p['system_pins'])
    for path,sha in p['project_pins'].items():
        sha=p['B_overrides'].get(path,p['common_override'].get(path,sha))
        allowed[mirror+path[len('/refactor'):]]=sha
        allowed[mirror+posixpath.normpath(path)[len('/refactor'):]]=sha
    allowed[source_path]=source_sha
    return dict(preprocess=prefix+['-M','-MF',before,'-MT',obj,source_path],compile=prefix+['-MD','-MF',actual,'-MT',obj,'-c',source_path,'-o',obj],link=link,allowed_dependency_pins=allowed,order='freshM+allactualbytes beforecompile; actualMD==freshM; oneobject+oneDSO; passiveELF only; noNative entry',new_HOST=1,new_GPU=0,ROOT_GO=False)
