import ast,pathlib,json
r=pathlib.Path('.');cpp=(r/'kimi_k3_moe_I192_caller_smoke_r8669zf.cpp').read_text();assert cpp.count('i192_actual_caller_r8669zf(')==3;assert '__attribute__((noinline)) void i192_actual_caller_r8669zf'in cpp
model=cpp[cpp.index('infinilm::runtime::Tensor KimiK3MoE::forward'):cpp.index('// Bounded synthetic caller-contract')];assert model.count('i192_actual_caller_r8669zf(')==1 and 'modern_kimi_grouped_I192_r8669y('not in model
helper=cpp[cpp.index('__attribute__((noinline)) void i192_actual_caller_r8669zf'):cpp.index('namespace infinilm::models::kimi_k3 {')];assert helper.count('modern_kimi_grouped_I192_r8669y(')==1 and 'DeviceGuard i192_device_guard_r8669zd'in helper
smoke=cpp[cpp.index('// Bounded synthetic caller-contract'):];assert 'cudaSetDevice(0)'in smoke and 'current!=0'in smoke and 'bytes=compact_workspace_bytes_r8669zc,canary=256'in smoke
for n in ['Kimi_I192_caller_smoke_r8669zf.py','launch_Kimi_I192_caller_smoke_r8669zf.py']:ast.parse((r/n).read_text())
entry=(r/'launch_Kimi_I192_caller_smoke_r8669zf.py').read_text();assert '--gres=dcu:2'in entry and '00:05:00'in entry and "'--mem=4096M'"in entry and 'same ready watcher died before GPU Popen'in entry
inside=(r/'Kimi_I192_caller_smoke_r8669zf_inside.sh').read_text();assert '240 python'in inside and 'GLOBAL_MODEL_MODE_R2354'not in inside
out={'Model_and_tiny_entry_same_noinline_helper':True,'Model_no_direct_raw_provider_bypass':True,'synthetic_label_and_canary_restore_contract':True,'finite2DCU_5min_4GiB_packet_AST':True,'GPU_executed':False};(r/'Kimi-I192-smoke-CPU-contract-r8669zf.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8',newline='\n');print(out)
