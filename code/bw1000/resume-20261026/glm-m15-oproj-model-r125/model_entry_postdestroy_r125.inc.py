# Called on each of16owner processes after originalmodern_glm_model_destroy(ctx).
released=(ctypes.c_uint64*17)();check(lib.modern_glm_m15_oproj_stats_r125(rank,released,17));report['M15_after_destroy']=list(released)
assert released[0]==0 and released[7]==released[8] and released[12]==released[13]==0 and released[5]==released[16]==0
