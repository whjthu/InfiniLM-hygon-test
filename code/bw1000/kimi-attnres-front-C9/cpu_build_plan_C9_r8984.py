"""PRINT ONLY: two-TU build plan for review. Does not create jobs or invoke compilers."""
import json,pathlib
r=pathlib.Path('/refactor')
b=r/'candidates/Kimi-attnres-front-C9-r8984'
ops=r/'candidates/rebased-5f91ad7-r3668/Ops/Kimi/InfiniOps'
sdk=r/'install/hygon-kimi-5f91ad7-r3672'
common=r/'candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM'
model=r/'candidates/kimi-serving-MoE-5f91ad7-r4665/InfiniLM'
inc=['-I/opt/dtk/cuda/cuda/include','-I'+str(model/'csrc'),'-I'+str(common/'csrc'),'-I'+str(sdk/'include'),'-I'+str(common/'third_party/json/include'),'-I'+str(common/'third_party/spdlog/include')]
commands=[
 ['/opt/dtk/cuda/cuda/bin/nvcc','-std=c++17','-O3','-arch=gfx936','-Xcompiler=-fPIC','-D_GLIBCXX_USE_CXX11_ABI=1','-I'+str(ops/'src'),'-I'+str(ops/'generated'),'-I'+str(sdk/'include'),'-MMD','-MF',str(b/'front.d'),'-c',str(b/'src/attnres_front_C9_r8984.cu'),'-o',str(b/'front.o')],
 ['g++','-std=c++17','-O2','-fno-omit-frame-pointer','-fPIC','-shared','-D_GLIBCXX_USE_CXX11_ABI=1',*inc,'-MMD','-MF',str(b/'adapter.d'),str(b/'src/attnres_explicit_C9_r8984.cpp'),str(b/'front.o'),'-Wl,--no-as-needed',str(common/'python/_infinilm.cpython-310-x86_64-linux-gnu.so'),'-Wl,--as-needed','-L'+str(common/'python/infinilm/_lib'),'-linfinilm_runtime','-L'+str(sdk/'lib'),'-linfiniops','-linfinirt','-linfiniccl','-ldl','-Wl,-rpath,'+str(common/'python/infinilm/_lib')+':'+str(sdk/'lib'),'-o',str(b/'libattnres_front_C9_r8984.so')]
]
print(json.dumps({'status':'PLAN_NOT_EXECUTED','CPU_only':True,'GPU':False,'dlopen':False,
 'scope':'one GPU TU, one host TU + link; reuse Runtime/current SDK; no Model/native/SDK rebuild',
 'commands':commands,'required_after_build':['fresh .d dependency SHA/readelf needed','nm actual five CAPI and launch definition','native SHA and actual loaded SDK pin before GPU GO'],
 'resource_plan':'CPU-only node c4/m8192, hard5min, total<=10min; fail STOP; no dlopen'},indent=2))
