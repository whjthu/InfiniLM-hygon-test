# GLM history value-only source candidate

SOURCE ONLY. ROOT/CPU/Native/device/Model/GPU GO are all false. Parent169/177 and all reports remain unchanged. No compile, remote, model or device invocation occurred.

`expand_cached_values` is one added nonvirtual method; no fields or existing signatures change. It derives from the pinned original expansion and keeps the full448/head (TP16 N1792) FP8 projection, original input guards, V256 contiguous copy, syncStream and Work exception/retire path. Only unconsumed K allocation, NoPE copy and perhead rotary copies disappear. It does not reduce GEMM work.

NoV explicitly calls the new history helper only after the packed branch is excluded, with original complete-prefix proof, v398, trueT1 single request, positivepast, U<=8192 andlocalheads4. Independent `GLM_HISTORY_VALUE_ONLY_V1=1` opt-in is exact; missing/0/other values retain original behavior. Observer enable does not control mathematics. FreshK+V remains original. The new helper retains both original gathers, rotor validation, successful expand counter and value lifetime holds.

`PARENT_*` files are exact immutable review baselines, not additional compile inputs. Only unprefixed candidate CPP/header files are candidateoverlay inputs; original r403selected_values and other includes remain pinnedexternal dependencies. The patch shows3existingfile deltas; new history helper is separate. Class-header change requires freshM overall32 recipes before compile, actualMD equality, allaffectedinlineconsumer rebuild and newELFsymbol/provider enumeration. Do not claim31reuse/old280exports from source alone.

`verify_source.py` performed one new localstdlib structural pass with6normal invariants and6rejected mutations. This is not a compile/numerical/performance result. `COMPONENT_PREREQUISITES.json` records missing actualcheckpoint binding and3controlledsynthetic BF16 inputpairs, numerical/guard/lifetime/post/livebuffer/floor evidence. No newModel activation capture is required forfirstcomponent. LaterModel still requires realfixedinputs,widequality/service andfullcontext evidence. TP1Factory route has64heads andcannotqualify intendedTP16heads4cost/NoVdispatch. No accepted input digests or actualweights are fabricated.

Measuredhistorywindow1.38/1.18ms motivates the candidate; isolatedremovedKcost is UNKNOWN. No Model multiplier, independent48sample claim or pairedwin. Original177 staysFAILED.
