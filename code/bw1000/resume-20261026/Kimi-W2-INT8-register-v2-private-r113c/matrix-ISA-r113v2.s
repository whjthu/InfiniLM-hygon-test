	.protected	_ZN6i8r1136matrixEPfPKaPKfPKiS6_S6_S6_S6_PKhS8_m ; -- Begin function _ZN6i8r1136matrixEPfPKaPKfPKiS6_S6_S6_S6_PKhS8_m
	.globl	_ZN6i8r1136matrixEPfPKaPKfPKiS6_S6_S6_S6_PKhS8_m
	.p2align	8
	.type	_ZN6i8r1136matrixEPfPKaPKfPKiS6_S6_S6_S6_PKhS8_m,@function
_ZN6i8r1136matrixEPfPKaPKfPKiS6_S6_S6_S6_PKhS8_m: ; @_ZN6i8r1136matrixEPfPKaPKfPKiS6_S6_S6_S6_PKhS8_m
; %bb.0:
	v_lshrrev_b32_e32 v30, 4, v0
	v_and_b32_e32 v29, 15, v0
	v_lshlrev_b32_e32 v4, 2, v30
	v_lshl_or_b32 v32, v29, 6, v4
	v_or_b32_e32 v33, 0x800, v4
	v_add_u32_e32 v4, 64, v0
	v_lshrrev_b32_e32 v4, 2, v4
	v_and_b32_e32 v4, 0x7c, v4
	v_or_b32_e32 v34, 0x800, v4
	v_add_u32_e32 v4, 0x80, v0
	v_lshrrev_b32_e32 v4, 2, v4
	v_and_b32_e32 v4, 0x7c, v4
	s_load_dwordx16 s[12:27], s[4:5], 0x0
	s_load_dwordx2 s[0:1], s[4:5], 0x50
	s_load_dwordx4 s[8:11], s[4:5], 0x40
	v_or_b32_e32 v35, 0x800, v4
	v_add_u32_e32 v4, 0xc0, v0
	v_lshrrev_b32_e32 v4, 2, v4
	v_and_b32_e32 v4, 0x7c, v4
	v_lshlrev_b32_e32 v3, 3, v30
	v_or_b32_e32 v36, 0x800, v4
	v_lshrrev_b32_e32 v4, 3, v0
	v_lshl_add_u32 v31, v29, 5, v3
	v_bfe_u32 v3, v0, 1, 4
	v_or_b32_e32 v4, 0x800, v4
	v_and_b32_e32 v37, 0x81c, v4
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v4, s9
	v_add_co_u32_e32 v7, vcc, s8, v3
	v_addc_co_u32_e32 v8, vcc, 0, v4, vcc
	v_add_co_u32_e32 v9, vcc, 16, v7
	v_addc_co_u32_e32 v10, vcc, 0, v8, vcc
	v_add_co_u32_e32 v11, vcc, 32, v7
	v_addc_co_u32_e32 v12, vcc, 0, v8, vcc
	v_add_co_u32_e32 v13, vcc, 48, v7
	v_addc_co_u32_e32 v14, vcc, 0, v8, vcc
	s_lshl_b64 s[28:29], s[0:1], 4
	v_mov_b32_e32 v2, 0
	v_add_co_u32_e32 v15, vcc, 64, v7
	v_mov_b32_e32 v1, v2
	s_add_u32 s34, s4, 0x58
	v_addc_co_u32_e32 v16, vcc, 0, v8, vcc
	s_movk_i32 s4, 0x50
	s_mov_b32 s30, 0
	s_movk_i32 s31, 0x800
	s_movk_i32 s2, 0x200
	v_add_co_u32_e32 v17, vcc, s4, v7
	v_lshlrev_b32_e32 v38, 2, v0
	v_lshrrev_b64 v[19:20], 5, v[0:1]
	s_mov_b32 s7, s30
	v_cmp_lt_u32_e64 s[0:1], 15, v0
	v_cmp_gt_u32_e64 s[2:3], s2, v0
	s_addc_u32 s35, s5, 0
	s_movk_i32 s33, 0xc0
	v_addc_co_u32_e32 v18, vcc, 0, v8, vcc
	s_movk_i32 s44, 0xe0
	s_mov_b32 s45, 0x4f800000
	s_movk_i32 s46, 0xe00
	s_movk_i32 s47, 0x60
	s_movk_i32 s48, 0x1bf
	v_or_b32_e32 v39, 0x800, v38
	v_lshl_or_b32 v40, v29, 2, s31
	v_and_b32_e32 v41, 31, v0
	v_add_u32_e32 v42, 4, v30
	v_add_u32_e32 v43, 8, v30
	v_add_u32_e32 v44, 12, v30
	v_subrev_u32_e32 v1, 64, v0
	s_branch .LBB3_3
.LBB3_1:                                ;   in Loop: Header=BB3_3 Depth=1
	s_or_b64 exec, exec, s[4:5]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	s_waitcnt lgkmcnt(0)
	s_load_dword s4, s[34:35], 0x0
	s_waitcnt lgkmcnt(0)
	s_add_u32 s6, s4, s6
	s_addc_u32 s7, 0, s7
	s_mov_b64 s[4:5], 0
.LBB3_2:                                ;   in Loop: Header=BB3_3 Depth=1
	s_and_b64 vcc, exec, s[4:5]
	s_cbranch_vccnz .LBB3_62
.LBB3_3:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB3_13 Depth 2
                                        ;     Child Loop BB3_20 Depth 2
                                        ;     Child Loop BB3_27 Depth 2
                                        ;     Child Loop BB3_34 Depth 2
                                        ;     Child Loop BB3_41 Depth 2
                                        ;     Child Loop BB3_48 Depth 2
	s_load_dword s4, s[26:27], 0x0
	s_waitcnt lgkmcnt(0)
	s_mul_hi_i32 s5, s4, 0xe0
	s_mulk_i32 s4, 0xe0
	v_mov_b32_e32 v3, s4
	v_mov_b32_e32 v4, s5
	v_cmp_lt_u64_e32 vcc, s[6:7], v[3:4]
	s_mov_b64 s[4:5], -1
	s_cbranch_vccz .LBB3_2
; %bb.4:                                ;   in Loop: Header=BB3_3 Depth=1
	v_mov_b32_e32 v3, 0x43600000
	v_mac_f32_e64 v3, 0, s45
	s_nop 0
	v_rcp_f32_e32 v3, v3
	v_mul_f32_e32 v3, 0x5f7ffffc, v3
	v_mul_f32_e32 v4, 0x2f800000, v3
	v_trunc_f32_e32 v4, v4
	v_mac_f32_e32 v3, 0xcf800000, v4
	v_cvt_u32_f32_e32 v3, v3
	v_cvt_u32_f32_e32 v4, v4
	v_readfirstlane_b32 s5, v3
	v_readfirstlane_b32 s4, v4
	s_mul_hi_u32 s9, s5, 0xffffff20
	s_mul_i32 s8, s4, 0xffffff20
	s_sub_i32 s9, s9, s5
	s_mul_i32 s31, s5, 0xffffff20
	s_add_i32 s9, s9, s8
	s_mul_hi_u32 s36, s5, s31
	s_mul_hi_u32 s8, s5, s9
	s_mul_i32 s5, s5, s9
	s_add_u32 s5, s36, s5
	s_addc_u32 s8, 0, s8
	s_mul_hi_u32 s37, s4, s31
	s_mul_i32 s31, s4, s31
	s_add_u32 s5, s5, s31
	s_mul_hi_u32 s36, s4, s9
	s_addc_u32 s5, s8, s37
	s_addc_u32 s8, s36, 0
	s_mul_i32 s9, s4, s9
	s_add_u32 s5, s5, s9
	s_addc_u32 s8, 0, s8
	v_add_co_u32_e32 v3, vcc, s5, v3
	s_cmp_lg_u64 vcc, 0
	s_addc_u32 s4, s4, s8
	v_readfirstlane_b32 s8, v3
	s_mul_hi_u32 s9, s8, 0xffffff20
	s_mul_i32 s5, s4, 0xffffff20
	s_sub_i32 s9, s9, s8
	s_add_i32 s9, s9, s5
	s_mul_i32 s36, s8, 0xffffff20
	s_mul_hi_u32 s5, s8, s9
	s_mul_i32 s31, s8, s9
	s_mul_hi_u32 s8, s8, s36
	s_add_u32 s8, s8, s31
	s_addc_u32 s5, 0, s5
	s_mul_hi_u32 s37, s4, s36
	s_mul_i32 s36, s4, s36
	s_add_u32 s8, s8, s36
	s_mul_hi_u32 s31, s4, s9
	s_addc_u32 s5, s5, s37
	s_addc_u32 s8, s31, 0
	s_mul_i32 s9, s4, s9
	s_add_u32 s5, s5, s9
	s_addc_u32 s8, 0, s8
	v_add_co_u32_e32 v3, vcc, s5, v3
	s_cmp_lg_u64 vcc, 0
	s_addc_u32 s4, s4, s8
	v_readfirstlane_b32 s9, v3
	s_mul_i32 s8, s6, s4
	s_mul_hi_u32 s31, s6, s9
	s_mul_hi_u32 s5, s6, s4
	s_add_u32 s8, s31, s8
	s_addc_u32 s5, 0, s5
	s_mul_hi_u32 s36, s7, s9
	s_mul_i32 s9, s7, s9
	s_add_u32 s8, s8, s9
	s_mul_hi_u32 s31, s7, s4
	s_addc_u32 s5, s5, s36
	s_addc_u32 s8, s31, 0
	s_mul_i32 s4, s7, s4
	s_add_u32 s4, s5, s4
	s_addc_u32 s5, 0, s8
	s_add_u32 s8, s4, 1
	s_addc_u32 s9, s5, 0
	s_add_u32 s31, s4, 2
	s_mul_i32 s37, s5, 0xe0
	s_mul_hi_u32 s38, s4, 0xe0
	s_addc_u32 s36, s5, 0
	s_add_i32 s38, s38, s37
	s_mul_i32 s37, s4, 0xe0
	v_mov_b32_e32 v3, s37
	v_sub_co_u32_e32 v3, vcc, s6, v3
	s_cmp_lg_u64 vcc, 0
	s_subb_u32 s37, s7, s38
	v_subrev_co_u32_e32 v4, vcc, s44, v3
	s_cmp_lg_u64 vcc, 0
	s_subb_u32 s38, s37, 0
	v_readfirstlane_b32 s39, v4
	s_cmpk_gt_u32 s39, 0xdf
	s_cselect_b32 s39, -1, 0
	s_cmp_eq_u32 s38, 0
	s_cselect_b32 s38, s39, -1
	s_cmp_lg_u32 s38, 0
	s_cselect_b32 s8, s31, s8
	v_readfirstlane_b32 s31, v3
	s_cselect_b32 s9, s36, s9
	s_cmpk_gt_u32 s31, 0xdf
	s_cselect_b32 s31, -1, 0
	s_cmp_eq_u32 s37, 0
	s_cselect_b32 s31, s31, -1
	s_cmp_lg_u32 s31, 0
	s_cselect_b32 s39, s9, s5
	s_cselect_b32 s38, s8, s4
	s_lshl_b64 s[36:37], s[38:39], 2
	s_add_u32 s4, s22, s36
	s_addc_u32 s5, s23, s37
	s_load_dword s4, s[4:5], 0x0
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_xor_b64 s[8:9], exec, s[8:9]
	s_cbranch_execz .LBB3_6
; %bb.5:                                ;   in Loop: Header=BB3_3 Depth=1
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s5, s4, 31
.LBB3_6:                                ;   in Loop: Header=BB3_3 Depth=1
	s_or_saveexec_b64 s[8:9], s[8:9]
	s_waitcnt lgkmcnt(0)
	v_mov_b64_e32 v[3:4], s[4:5]
	s_xor_b64 exec, exec, s[8:9]
	s_cbranch_execz .LBB3_10
; %bb.7:                                ;   in Loop: Header=BB3_3 Depth=1
	s_add_u32 s36, s24, s36
	s_addc_u32 s37, s25, s37
	s_ashr_i32 s5, s4, 31
	s_load_dword s31, s[36:37], 0x0
	s_lshl_b64 s[36:37], s[4:5], 2
	s_add_u32 s36, s18, s36
	s_addc_u32 s37, s19, s37
	s_load_dword s36, s[36:37], 0x0
	s_waitcnt lgkmcnt(0)
	v_add_u32_e32 v3, s31, v0
	v_cmp_gt_i32_e32 vcc, s36, v3
	v_mov_b32_e32 v3, -1
	s_and_saveexec_b64 s[36:37], vcc
	s_cbranch_execz .LBB3_9
; %bb.8:                                ;   in Loop: Header=BB3_3 Depth=1
	s_ashr_i32 s40, s31, 31
	v_mov_b32_e32 v4, s40
	v_add_co_u32_e32 v3, vcc, s31, v0
	v_addc_co_u32_e32 v4, vcc, 0, v4, vcc
	v_mov_b32_e32 v5, s28
	v_mad_u64_u32 v[3:4], s[40:41], s4, v5, v[3:4]
	s_mul_i32 s31, s5, s28
	v_mov_b32_e32 v5, s21
	v_add_u32_e32 v4, s31, v4
	s_mul_i32 s31, s4, s29
	v_add_u32_e32 v4, s31, v4
	v_lshlrev_b64 v[3:4], 2, v[3:4]
	v_add_co_u32_e32 v3, vcc, s20, v3
	v_addc_co_u32_e32 v4, vcc, v5, v4, vcc
	global_load_dword v3, v[3:4], off
.LBB3_9:                                ;   in Loop: Header=BB3_3 Depth=1
	s_or_b64 exec, exec, s[36:37]
	s_waitcnt vmcnt(0)
	ds_write_b32 v39, v3
	v_mov_b64_e32 v[3:4], s[4:5]
.LBB3_10:                               ;   in Loop: Header=BB3_3 Depth=1
	s_or_b64 exec, exec, s[8:9]
	v_mad_u64_u32 v[27:28], s[4:5], v3, s46, 0
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mov_b32_e32 v3, v28
	s_waitcnt lgkmcnt(0)
	ds_read_b32 v5, v40
	v_mad_u64_u32 v[3:4], s[4:5], v4, s46, v[3:4]
	v_mov_b32_e32 v46, v3
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB3_15
; %bb.11:                               ;   in Loop: Header=BB3_3 Depth=1
	v_add_co_u32_e32 v3, vcc, v27, v19
	v_addc_co_u32_e32 v4, vcc, v46, v20, vcc
	v_mad_u64_u32 v[3:4], s[4:5], s6, 16, v[3:4]
	v_mov_b32_e32 v6, 0xfffff200
	s_mov_b64 s[36:37], 0
	v_lshl_add_u32 v4, s7, 4, v4
	v_mad_u64_u32 v[21:22], s[4:5], s38, v6, v[3:4]
	v_mad_u64_u32 v[3:4], s[4:5], v21, s47, v[7:8]
	s_mul_i32 s4, s39, 0xfffff200
	v_add_u32_e32 v6, s4, v22
	v_subrev_u32_e32 v6, s38, v6
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[21:22], s[4:5], v6, s47, v[4:5]
	v_mov_b32_e32 v6, v38
	v_mov_b32_e32 v22, v1
	v_mov_b32_e32 v4, v21
	v_mov_b32_e32 v21, v37
	s_branch .LBB3_13
.LBB3_12:                               ;   in Loop: Header=BB3_13 Depth=2
	s_or_b64 exec, exec, s[4:5]
	global_load_ubyte v24, v[3:4], off
	v_and_b32_e32 v25, 4, v6
	s_getpc_b64 s[4:5]
	s_add_u32 s4, s4, __const._ZN6i8r1136decodeEh.q@rel32@lo+4
	s_addc_u32 s5, s5, __const._ZN6i8r1136decodeEh.q@rel32@hi+12
	s_waitcnt vmcnt(1)
	ds_write_b8 v22, v23 offset:1088
	v_add_co_u32_e32 v3, vcc, 0xc0, v3
	v_addc_co_u32_e32 v4, vcc, 0, v4, vcc
	v_add_u32_e32 v21, 8, v21
	v_add_u32_e32 v6, 0x100, v6
	s_waitcnt vmcnt(0)
	v_bfe_u32 v26, v24, v25, 3
	v_lshlrev_b32_e32 v26, 2, v26
	global_load_dword v26, v26, s[4:5]
	v_lshrrev_b32_e32 v23, v25, v24
	v_and_b32_e32 v23, 8, v23
	v_cmp_eq_u32_e64 s[4:5], 0, v23
	s_waitcnt vmcnt(0)
	v_sub_u32_e32 v24, 0, v26
	v_cndmask_b32_e64 v23, v24, v26, s[4:5]
	ds_write_b8 v22, v23 offset:1600
	v_add_u32_e32 v22, 64, v22
	v_cmp_lt_u32_e32 vcc, s48, v22
	s_or_b64 s[36:37], vcc, s[36:37]
	s_andn2_b64 exec, exec, s[36:37]
	s_cbranch_execz .LBB3_15
.LBB3_13:                               ;   Parent Loop BB3_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ds_read_b32 v24, v21
	v_mov_b32_e32 v23, 0
	s_waitcnt lgkmcnt(0)
	v_cmp_lt_i32_e32 vcc, -1, v24
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB3_12
; %bb.14:                               ;   in Loop: Header=BB3_13 Depth=2
	v_mad_u64_u32 v[23:24], s[40:41], v24, s33, 0
	v_mov_b32_e32 v25, s15
	v_or_b32_e32 v23, v23, v41
	v_or_b32_e32 v24, 0, v24
	v_add_co_u32_e32 v23, vcc, s14, v23
	v_addc_co_u32_e32 v24, vcc, v25, v24, vcc
	global_load_ubyte v23, v[23:24], off
	s_branch .LBB3_12
.LBB3_15:                               ;   in Loop: Header=BB3_3 Depth=1
	s_or_b64 exec, exec, s[8:9]
	v_add_u32_e32 v21, 0x400, v31
	s_mul_i32 s4, s39, 0xe0
	s_mul_hi_u32 s5, s38, 0xe0
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt lgkmcnt(0)
	ds_read2_b32 v[3:4], v21 offset1:1
	s_add_i32 s5, s5, s4
	s_mul_i32 s4, s38, 0xe0
	ds_read2_b32 v[21:22], v21 offset0:128 offset1:129
	s_sub_u32 s4, s6, s4
	s_subb_u32 s5, s7, s5
	s_lshl_b64 s[36:37], s[4:5], 4
	v_cmp_lt_i32_e64 s[4:5], -1, v5
	v_mov_b32_e32 v6, s37
	v_add_co_u32_e32 v45, vcc, s36, v27
	v_mad_u64_u32 v[23:24], s[8:9], v5, 6, 0
	s_waitcnt lgkmcnt(1)
	v_and_b32_e32 v5, 0xff00, v4
	v_addc_co_u32_e32 v28, vcc, v46, v6, vcc
	v_or_b32_sdwa v5, v4, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v6, 0xff00, v22
	v_and_b32_e32 v25, 0xff0000, v4
	v_and_b32_e32 v4, 0xff000000, v4
	v_or_b32_sdwa v6, v22, v6 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
	v_and_b32_e32 v47, 0xff0000, v22
	v_or3_b32 v26, v5, v25, v4
	v_or3_b32 v25, v3, 0, 0
	v_and_b32_e32 v3, 0xff000000, v22
	v_or3_b32 v22, v6, v47, v3
	v_mov_b32_e32 v3, v2
	v_mov_b32_e32 v4, v2
	v_mov_b32_e32 v5, v2
	v_mov_b32_e32 v6, v5
	v_or3_b32 v21, v21, 0, 0
	v_mov_b32_e32 v5, v4
	v_mov_b32_e32 v4, v3
	v_mov_b32_e32 v3, v2
	s_nop 1
	v_mmac_i32_16x16x32_i8 v[3:6], v[25:26], v[21:22], v[3:6]
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v26, 0
	s_and_saveexec_b64 s[8:9], s[4:5]
	s_cbranch_execz .LBB3_17
; %bb.16:                               ;   in Loop: Header=BB3_3 Depth=1
	v_add_co_u32_e32 v25, vcc, v45, v30
	v_addc_co_u32_e32 v49, vcc, 0, v28, vcc
	v_add_co_u32_e32 v47, vcc, v45, v42
	v_mad_u64_u32 v[25:26], s[40:41], v25, 6, s[10:11]
	v_mad_u64_u32 v[47:48], s[40:41], v47, 6, s[10:11]
	v_lshlrev_b64 v[21:22], 2, v[23:24]
	v_addc_co_u32_e32 v51, vcc, 0, v28, vcc
	v_mad_u64_u32 v[49:50], s[40:41], v49, 6, v[26:27]
	v_mov_b32_e32 v26, v48
	v_mad_u64_u32 v[50:51], s[40:41], v51, 6, v[26:27]
	v_mov_b32_e32 v52, s17
	v_add_co_u32_e32 v21, vcc, s16, v21
	v_addc_co_u32_e32 v22, vcc, v52, v22, vcc
	global_load_dword v21, v[21:22], off
	v_add_co_u32_e32 v22, vcc, v45, v43
	v_mov_b32_e32 v48, v50
	v_addc_co_u32_e32 v51, vcc, 0, v28, vcc
	global_load_ubyte v54, v[47:48], off
	v_mad_u64_u32 v[47:48], s[40:41], v22, 6, s[10:11]
	v_add_co_u32_e32 v22, vcc, v45, v44
	v_mov_b32_e32 v26, v49
	v_mad_u64_u32 v[49:50], s[40:41], v22, 6, s[10:11]
	v_mov_b32_e32 v22, v48
	v_addc_co_u32_e32 v53, vcc, 0, v28, vcc
	v_mad_u64_u32 v[51:52], s[40:41], v51, 6, v[22:23]
	v_mov_b32_e32 v22, v50
	v_mad_u64_u32 v[52:53], s[40:41], v53, 6, v[22:23]
	v_mov_b32_e32 v48, v51
	global_load_ubyte v51, v[25:26], off
	global_load_ubyte v53, v[47:48], off
	v_mov_b32_e32 v50, v52
	global_load_ubyte v47, v[49:50], off
	v_cvt_f32_i32_e32 v4, v4
	v_cvt_f32_i32_e32 v3, v3
	v_cvt_f32_i32_e32 v6, v6
	v_cvt_f32_i32_e32 v5, v5
	s_mov_b32 s31, s30
	v_mov_b64_e32 v[25:26], s[30:31]
	s_waitcnt vmcnt(4)
	v_mov_b32_e32 v22, v21
	s_waitcnt vmcnt(3)
	v_add_u32_e32 v49, 0xffffff81, v54
	s_waitcnt vmcnt(2)
	v_add_u32_e32 v48, 0xffffff81, v51
	s_waitcnt vmcnt(1)
	v_add_u32_e32 v50, 0xffffff81, v53
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v51, 0xffffff81, v47
	v_ldexp_f32 v47, 0.5, v48
	v_ldexp_f32 v48, 0.5, v49
	v_ldexp_f32 v49, 0.5, v50
	v_ldexp_f32 v50, 0.5, v51
	v_pk_mul_f32 v[47:48], v[21:22], v[47:48]
	v_pk_mul_f32 v[49:50], v[21:22], v[49:50]
	v_pk_fma_f32 v[21:22], v[3:4], v[47:48], v[25:26]
	v_pk_fma_f32 v[25:26], v[5:6], v[49:50], v[25:26]
.LBB3_17:                               ;   in Loop: Header=BB3_3 Depth=1
	s_or_b64 exec, exec, s[8:9]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt lgkmcnt(0)
	s_and_saveexec_b64 s[40:41], s[2:3]
	s_cbranch_execz .LBB3_22
; %bb.18:                               ;   in Loop: Header=BB3_3 Depth=1
	v_add_co_u32_e32 v3, vcc, v27, v19
	v_addc_co_u32_e32 v4, vcc, v46, v20, vcc
	v_mad_u64_u32 v[3:4], s[8:9], s6, 16, v[3:4]
	v_mov_b32_e32 v5, 0xfffff200
	s_mov_b64 s[42:43], 0
	v_lshl_add_u32 v4, s7, 4, v4
	v_mad_u64_u32 v[4:5], s[8:9], s38, v5, v[3:4]
	v_mov_b32_e32 v6, v38
	v_mov_b32_e32 v47, v1
	v_mad_u64_u32 v[3:4], s[8:9], v4, s47, v[9:10]
	s_mul_i32 s8, s39, 0xfffff200
	v_add_u32_e32 v5, s8, v5
	v_subrev_u32_e32 v5, s38, v5
	v_mad_u64_u32 v[4:5], s[8:9], v5, s47, v[4:5]
	v_mov_b32_e32 v5, v37
	s_branch .LBB3_20
.LBB3_19:                               ;   in Loop: Header=BB3_20 Depth=2
	s_or_b64 exec, exec, s[8:9]
	global_load_ubyte v49, v[3:4], off
	v_and_b32_e32 v50, 4, v6
	s_getpc_b64 s[8:9]
	s_add_u32 s8, s8, __const._ZN6i8r1136decodeEh.q@rel32@lo+4
	s_addc_u32 s9, s9, __const._ZN6i8r1136decodeEh.q@rel32@hi+12
	s_waitcnt vmcnt(1)
	ds_write_b8 v47, v48 offset:1088
	v_add_co_u32_e32 v3, vcc, 0xc0, v3
	v_add_u32_e32 v6, 0x100, v6
	v_add_u32_e32 v5, 8, v5
	v_addc_co_u32_e32 v4, vcc, 0, v4, vcc
	s_waitcnt vmcnt(0)
	v_bfe_u32 v51, v49, v50, 3
	v_lshlrev_b32_e32 v51, 2, v51
	global_load_dword v51, v51, s[8:9]
	v_lshrrev_b32_e32 v48, v50, v49
	v_and_b32_e32 v48, 8, v48
	v_cmp_eq_u32_e64 s[8:9], 0, v48
	s_waitcnt vmcnt(0)
	v_sub_u32_e32 v49, 0, v51
	v_cndmask_b32_e64 v48, v49, v51, s[8:9]
	ds_write_b8 v47, v48 offset:1600
	v_add_u32_e32 v47, 64, v47
	v_cmp_lt_u32_e64 s[8:9], s48, v47
	s_or_b64 s[42:43], s[8:9], s[42:43]
	s_andn2_b64 exec, exec, s[42:43]
	s_cbranch_execz .LBB3_22
.LBB3_20:                               ;   Parent Loop BB3_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ds_read_b32 v49, v5
	v_mov_b32_e32 v48, 0
	s_waitcnt lgkmcnt(0)
	v_cmp_lt_i32_e32 vcc, -1, v49
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB3_19
; %bb.21:                               ;   in Loop: Header=BB3_20 Depth=2
	v_mad_u64_u32 v[48:49], s[50:51], v49, s33, 0
	v_mov_b32_e32 v50, s15
	v_or_b32_e32 v48, v48, v41
	v_or_b32_e32 v49, 0, v49
	v_add_co_u32_e32 v48, vcc, s14, v48
	v_addc_co_u32_e32 v49, vcc, v50, v49, vcc
	global_load_ubyte v48, v[48:49], off offset:32
	s_branch .LBB3_19
.LBB3_22:                               ;   in Loop: Header=BB3_3 Depth=1
	s_or_b64 exec, exec, s[40:41]
	v_add_u32_e32 v5, 0x400, v31
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt lgkmcnt(0)
	ds_read2_b32 v[3:4], v5 offset1:1
	ds_read2_b32 v[5:6], v5 offset0:128 offset1:129
	s_waitcnt lgkmcnt(1)
	v_and_b32_e32 v47, 0xff00, v4
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v48, 0xff00, v6
	v_or_b32_sdwa v47, v4, v47 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
	v_or_b32_sdwa v49, v6, v48 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
	v_and_b32_e32 v48, 0xff0000, v4
	v_and_b32_e32 v4, 0xff000000, v4
	v_and_b32_e32 v50, 0xff0000, v6
	v_or3_b32 v48, v47, v48, v4
	v_or3_b32 v47, v3, 0, 0
	v_and_b32_e32 v3, 0xff000000, v6
	v_or3_b32 v50, v49, v50, v3
	v_or3_b32 v49, v5, 0, 0
	v_mov_b32_e32 v3, v2
	v_mov_b32_e32 v4, v2
	v_mov_b32_e32 v5, v2
	v_mov_b32_e32 v6, v5
	v_mov_b32_e32 v5, v4
	v_mov_b32_e32 v4, v3
	v_mov_b32_e32 v3, v2
	s_nop 1
	v_mmac_i32_16x16x32_i8 v[3:6], v[47:48], v[49:50], v[3:6]
	s_and_saveexec_b64 s[8:9], s[4:5]
	s_cbranch_execz .LBB3_24
; %bb.23:                               ;   in Loop: Header=BB3_3 Depth=1
	v_add_co_u32_e32 v47, vcc, v45, v30
	v_mad_u64_u32 v[47:48], s[40:41], v47, 6, s[10:11]
	v_lshlrev_b64 v[49:50], 2, v[23:24]
	v_addc_co_u32_e32 v51, vcc, 0, v28, vcc
	v_mad_u64_u32 v[51:52], s[40:41], v51, 6, v[48:49]
	v_add_co_u32_e32 v48, vcc, v45, v42
	v_mad_u64_u32 v[52:53], s[40:41], v48, 6, s[10:11]
	v_addc_co_u32_e32 v55, vcc, 0, v28, vcc
	v_mov_b32_e32 v54, s17
	v_add_co_u32_e32 v48, vcc, s16, v49
	v_addc_co_u32_e32 v49, vcc, v54, v50, vcc
	v_mov_b32_e32 v50, v53
	global_load_dword v49, v[48:49], off offset:4
	v_mov_b32_e32 v48, v51
	v_mad_u64_u32 v[50:51], s[40:41], v55, 6, v[50:51]
	v_add_co_u32_e32 v51, vcc, v45, v43
	v_addc_co_u32_e32 v53, vcc, 0, v28, vcc
	v_mad_u64_u32 v[54:55], s[40:41], v51, 6, s[10:11]
	v_add_co_u32_e32 v51, vcc, v45, v44
	v_mad_u64_u32 v[56:57], s[40:41], v51, 6, s[10:11]
	v_mov_b32_e32 v51, v55
	v_addc_co_u32_e32 v60, vcc, 0, v28, vcc
	v_mad_u64_u32 v[58:59], s[40:41], v53, 6, v[51:52]
	v_mov_b32_e32 v51, v57
	v_mad_u64_u32 v[59:60], s[40:41], v60, 6, v[51:52]
	v_mov_b32_e32 v53, v50
	global_load_ubyte v47, v[47:48], off offset:1
	v_mov_b32_e32 v55, v58
	v_mov_b32_e32 v57, v59
	global_load_ubyte v48, v[52:53], off offset:1
	global_load_ubyte v51, v[56:57], off offset:1
	global_load_ubyte v58, v[54:55], off offset:1
	v_cvt_f32_i32_e32 v4, v4
	v_cvt_f32_i32_e32 v3, v3
	v_cvt_f32_i32_e32 v6, v6
	v_cvt_f32_i32_e32 v5, v5
	s_waitcnt vmcnt(4)
	v_mov_b32_e32 v50, v49
	s_waitcnt vmcnt(3)
	v_add_u32_e32 v47, 0xffffff81, v47
	s_waitcnt vmcnt(2)
	v_add_u32_e32 v48, 0xffffff81, v48
	s_waitcnt vmcnt(1)
	v_add_u32_e32 v53, 0xffffff81, v51
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v52, 0xffffff81, v58
	v_ldexp_f32 v47, 0.5, v47
	v_ldexp_f32 v48, 0.5, v48
	v_ldexp_f32 v51, 0.5, v52
	v_ldexp_f32 v52, 0.5, v53
	v_pk_mul_f32 v[47:48], v[49:50], v[47:48]
	v_pk_mul_f32 v[49:50], v[49:50], v[51:52]
	v_pk_fma_f32 v[21:22], v[3:4], v[47:48], v[21:22]
	v_pk_fma_f32 v[25:26], v[5:6], v[49:50], v[25:26]
.LBB3_24:                               ;   in Loop: Header=BB3_3 Depth=1
	s_or_b64 exec, exec, s[8:9]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt lgkmcnt(0)
	s_and_saveexec_b64 s[40:41], s[2:3]
	s_cbranch_execz .LBB3_29
; %bb.25:                               ;   in Loop: Header=BB3_3 Depth=1
	v_add_co_u32_e32 v3, vcc, v27, v19
	v_addc_co_u32_e32 v4, vcc, v46, v20, vcc
	v_mad_u64_u32 v[3:4], s[8:9], s6, 16, v[3:4]
	v_mov_b32_e32 v5, 0xfffff200
	s_mov_b64 s[42:43], 0
	v_lshl_add_u32 v4, s7, 4, v4
	v_mad_u64_u32 v[4:5], s[8:9], s38, v5, v[3:4]
	v_mov_b32_e32 v6, v38
	v_mov_b32_e32 v47, v1
	v_mad_u64_u32 v[3:4], s[8:9], v4, s47, v[11:12]
	s_mul_i32 s8, s39, 0xfffff200
	v_add_u32_e32 v5, s8, v5
	v_subrev_u32_e32 v5, s38, v5
	v_mad_u64_u32 v[4:5], s[8:9], v5, s47, v[4:5]
	v_mov_b32_e32 v5, v37
	s_branch .LBB3_27
.LBB3_26:                               ;   in Loop: Header=BB3_27 Depth=2
	s_or_b64 exec, exec, s[8:9]
	global_load_ubyte v49, v[3:4], off
	v_and_b32_e32 v50, 4, v6
	s_getpc_b64 s[8:9]
	s_add_u32 s8, s8, __const._ZN6i8r1136decodeEh.q@rel32@lo+4
	s_addc_u32 s9, s9, __const._ZN6i8r1136decodeEh.q@rel32@hi+12
	s_waitcnt vmcnt(1)
	ds_write_b8 v47, v48 offset:1088
	v_add_co_u32_e32 v3, vcc, 0xc0, v3
	v_add_u32_e32 v6, 0x100, v6
	v_add_u32_e32 v5, 8, v5
	v_addc_co_u32_e32 v4, vcc, 0, v4, vcc
	s_waitcnt vmcnt(0)
	v_bfe_u32 v51, v49, v50, 3
	v_lshlrev_b32_e32 v51, 2, v51
	global_load_dword v51, v51, s[8:9]
	v_lshrrev_b32_e32 v48, v50, v49
	v_and_b32_e32 v48, 8, v48
	v_cmp_eq_u32_e64 s[8:9], 0, v48
	s_waitcnt vmcnt(0)
	v_sub_u32_e32 v49, 0, v51
	v_cndmask_b32_e64 v48, v49, v51, s[8:9]
	ds_write_b8 v47, v48 offset:1600
	v_add_u32_e32 v47, 64, v47
	v_cmp_lt_u32_e64 s[8:9], s48, v47
	s_or_b64 s[42:43], s[8:9], s[42:43]
	s_andn2_b64 exec, exec, s[42:43]
	s_cbranch_execz .LBB3_29
.LBB3_27:                               ;   Parent Loop BB3_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ds_read_b32 v49, v5
	v_mov_b32_e32 v48, 0
	s_waitcnt lgkmcnt(0)
	v_cmp_lt_i32_e32 vcc, -1, v49
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB3_26
; %bb.28:                               ;   in Loop: Header=BB3_27 Depth=2
	v_mad_u64_u32 v[48:49], s[50:51], v49, s33, 0
	v_mov_b32_e32 v50, s15
	v_or_b32_e32 v48, v48, v41
	v_or_b32_e32 v49, 0, v49
	v_add_co_u32_e32 v48, vcc, s14, v48
	v_addc_co_u32_e32 v49, vcc, v50, v49, vcc
	global_load_ubyte v48, v[48:49], off offset:64
	s_branch .LBB3_26
.LBB3_29:                               ;   in Loop: Header=BB3_3 Depth=1
	s_or_b64 exec, exec, s[40:41]
	v_add_u32_e32 v5, 0x400, v31
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt lgkmcnt(0)
	ds_read2_b32 v[3:4], v5 offset1:1
	ds_read2_b32 v[5:6], v5 offset0:128 offset1:129
	s_waitcnt lgkmcnt(1)
	v_and_b32_e32 v47, 0xff00, v4
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v48, 0xff00, v6
	v_or_b32_sdwa v47, v4, v47 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
	v_or_b32_sdwa v49, v6, v48 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
	v_and_b32_e32 v48, 0xff0000, v4
	v_and_b32_e32 v4, 0xff000000, v4
	v_and_b32_e32 v50, 0xff0000, v6
	v_or3_b32 v48, v47, v48, v4
	v_or3_b32 v47, v3, 0, 0
	v_and_b32_e32 v3, 0xff000000, v6
	v_or3_b32 v50, v49, v50, v3
	v_or3_b32 v49, v5, 0, 0
	v_mov_b32_e32 v3, v2
	v_mov_b32_e32 v4, v2
	v_mov_b32_e32 v5, v2
	v_mov_b32_e32 v6, v5
	v_mov_b32_e32 v5, v4
	v_mov_b32_e32 v4, v3
	v_mov_b32_e32 v3, v2
	s_nop 1
	v_mmac_i32_16x16x32_i8 v[3:6], v[47:48], v[49:50], v[3:6]
	s_and_saveexec_b64 s[8:9], s[4:5]
	s_cbranch_execz .LBB3_31
; %bb.30:                               ;   in Loop: Header=BB3_3 Depth=1
	v_add_co_u32_e32 v49, vcc, v45, v30
	v_lshlrev_b64 v[47:48], 2, v[23:24]
	v_mad_u64_u32 v[49:50], s[40:41], v49, 6, s[10:11]
	v_addc_co_u32_e32 v52, vcc, 0, v28, vcc
	v_mov_b32_e32 v51, s17
	v_add_co_u32_e32 v47, vcc, s16, v47
	v_addc_co_u32_e32 v48, vcc, v51, v48, vcc
	global_load_dword v47, v[47:48], off offset:8
	v_mov_b32_e32 v48, v50
	v_mad_u64_u32 v[50:51], s[40:41], v52, 6, v[48:49]
	v_add_co_u32_e32 v48, vcc, v45, v42
	v_addc_co_u32_e32 v55, vcc, 0, v28, vcc
	v_mad_u64_u32 v[51:52], s[40:41], v48, 6, s[10:11]
	v_add_co_u32_e32 v48, vcc, v45, v43
	v_mad_u64_u32 v[53:54], s[40:41], v48, 6, s[10:11]
	v_mov_b32_e32 v48, v52
	v_addc_co_u32_e32 v57, vcc, 0, v28, vcc
	v_mad_u64_u32 v[55:56], s[40:41], v55, 6, v[48:49]
	v_mov_b32_e32 v48, v54
	v_mad_u64_u32 v[56:57], s[40:41], v57, 6, v[48:49]
	v_add_co_u32_e32 v48, vcc, v45, v44
	v_mad_u64_u32 v[57:58], s[40:41], v48, 6, s[10:11]
	v_addc_co_u32_e32 v59, vcc, 0, v28, vcc
	v_mov_b32_e32 v48, v58
	v_mov_b32_e32 v52, v55
	v_mov_b32_e32 v54, v56
	v_mad_u64_u32 v[55:56], s[40:41], v59, 6, v[48:49]
	global_load_ubyte v56, v[49:50], off offset:2
	global_load_ubyte v59, v[53:54], off offset:2
	global_load_ubyte v60, v[51:52], off offset:2
	v_cvt_f32_i32_e32 v4, v4
	v_mov_b32_e32 v58, v55
	global_load_ubyte v49, v[57:58], off offset:2
	v_cvt_f32_i32_e32 v3, v3
	v_cvt_f32_i32_e32 v6, v6
	v_cvt_f32_i32_e32 v5, v5
	s_waitcnt vmcnt(4)
	v_mov_b32_e32 v48, v47
	s_waitcnt vmcnt(3)
	v_add_u32_e32 v50, 0xffffff81, v56
	s_waitcnt vmcnt(2)
	v_add_u32_e32 v52, 0xffffff81, v59
	s_waitcnt vmcnt(1)
	v_add_u32_e32 v51, 0xffffff81, v60
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v53, 0xffffff81, v49
	v_ldexp_f32 v49, 0.5, v50
	v_ldexp_f32 v50, 0.5, v51
	v_ldexp_f32 v51, 0.5, v52
	v_ldexp_f32 v52, 0.5, v53
	v_pk_mul_f32 v[49:50], v[47:48], v[49:50]
	v_pk_mul_f32 v[47:48], v[47:48], v[51:52]
	v_pk_fma_f32 v[21:22], v[3:4], v[49:50], v[21:22]
	v_pk_fma_f32 v[25:26], v[5:6], v[47:48], v[25:26]
.LBB3_31:                               ;   in Loop: Header=BB3_3 Depth=1
	s_or_b64 exec, exec, s[8:9]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt lgkmcnt(0)
	s_and_saveexec_b64 s[40:41], s[2:3]
	s_cbranch_execz .LBB3_36
; %bb.32:                               ;   in Loop: Header=BB3_3 Depth=1
	v_add_co_u32_e32 v3, vcc, v27, v19
	v_addc_co_u32_e32 v4, vcc, v46, v20, vcc
	v_mad_u64_u32 v[3:4], s[8:9], s6, 16, v[3:4]
	v_mov_b32_e32 v5, 0xfffff200
	s_mov_b64 s[42:43], 0
	v_lshl_add_u32 v4, s7, 4, v4
	v_mad_u64_u32 v[4:5], s[8:9], s38, v5, v[3:4]
	v_mov_b32_e32 v6, v38
	v_mov_b32_e32 v47, v1
	v_mad_u64_u32 v[3:4], s[8:9], v4, s47, v[13:14]
	s_mul_i32 s8, s39, 0xfffff200
	v_add_u32_e32 v5, s8, v5
	v_subrev_u32_e32 v5, s38, v5
	v_mad_u64_u32 v[4:5], s[8:9], v5, s47, v[4:5]
	v_mov_b32_e32 v5, v37
	s_branch .LBB3_34
.LBB3_33:                               ;   in Loop: Header=BB3_34 Depth=2
	s_or_b64 exec, exec, s[8:9]
	global_load_ubyte v49, v[3:4], off
	v_and_b32_e32 v50, 4, v6
	s_getpc_b64 s[8:9]
	s_add_u32 s8, s8, __const._ZN6i8r1136decodeEh.q@rel32@lo+4
	s_addc_u32 s9, s9, __const._ZN6i8r1136decodeEh.q@rel32@hi+12
	s_waitcnt vmcnt(1)
	ds_write_b8 v47, v48 offset:1088
	v_add_co_u32_e32 v3, vcc, 0xc0, v3
	v_add_u32_e32 v6, 0x100, v6
	v_add_u32_e32 v5, 8, v5
	v_addc_co_u32_e32 v4, vcc, 0, v4, vcc
	s_waitcnt vmcnt(0)
	v_bfe_u32 v51, v49, v50, 3
	v_lshlrev_b32_e32 v51, 2, v51
	global_load_dword v51, v51, s[8:9]
	v_lshrrev_b32_e32 v48, v50, v49
	v_and_b32_e32 v48, 8, v48
	v_cmp_eq_u32_e64 s[8:9], 0, v48
	s_waitcnt vmcnt(0)
	v_sub_u32_e32 v49, 0, v51
	v_cndmask_b32_e64 v48, v49, v51, s[8:9]
	ds_write_b8 v47, v48 offset:1600
	v_add_u32_e32 v47, 64, v47
	v_cmp_lt_u32_e64 s[8:9], s48, v47
	s_or_b64 s[42:43], s[8:9], s[42:43]
	s_andn2_b64 exec, exec, s[42:43]
	s_cbranch_execz .LBB3_36
.LBB3_34:                               ;   Parent Loop BB3_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ds_read_b32 v49, v5
	v_mov_b32_e32 v48, 0
	s_waitcnt lgkmcnt(0)
	v_cmp_lt_i32_e32 vcc, -1, v49
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB3_33
; %bb.35:                               ;   in Loop: Header=BB3_34 Depth=2
	v_mad_u64_u32 v[48:49], s[50:51], v49, s33, 0
	v_mov_b32_e32 v50, s15
	v_or_b32_e32 v48, v48, v41
	v_or_b32_e32 v49, 0, v49
	v_add_co_u32_e32 v48, vcc, s14, v48
	v_addc_co_u32_e32 v49, vcc, v50, v49, vcc
	global_load_ubyte v48, v[48:49], off offset:96
	s_branch .LBB3_33
.LBB3_36:                               ;   in Loop: Header=BB3_3 Depth=1
	s_or_b64 exec, exec, s[40:41]
	v_add_u32_e32 v5, 0x400, v31
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt lgkmcnt(0)
	ds_read2_b32 v[3:4], v5 offset1:1
	ds_read2_b32 v[5:6], v5 offset0:128 offset1:129
	s_waitcnt lgkmcnt(1)
	v_and_b32_e32 v47, 0xff00, v4
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v48, 0xff00, v6
	v_or_b32_sdwa v47, v4, v47 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
	v_or_b32_sdwa v49, v6, v48 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
	v_and_b32_e32 v48, 0xff0000, v4
	v_and_b32_e32 v4, 0xff000000, v4
	v_and_b32_e32 v50, 0xff0000, v6
	v_or3_b32 v48, v47, v48, v4
	v_or3_b32 v47, v3, 0, 0
	v_and_b32_e32 v3, 0xff000000, v6
	v_or3_b32 v50, v49, v50, v3
	v_or3_b32 v49, v5, 0, 0
	v_mov_b32_e32 v3, v2
	v_mov_b32_e32 v4, v2
	v_mov_b32_e32 v5, v2
	v_mov_b32_e32 v6, v5
	v_mov_b32_e32 v5, v4
	v_mov_b32_e32 v4, v3
	v_mov_b32_e32 v3, v2
	s_nop 1
	v_mmac_i32_16x16x32_i8 v[3:6], v[47:48], v[49:50], v[3:6]
	s_and_saveexec_b64 s[8:9], s[4:5]
	s_cbranch_execz .LBB3_38
; %bb.37:                               ;   in Loop: Header=BB3_3 Depth=1
	v_add_co_u32_e32 v49, vcc, v45, v30
	v_lshlrev_b64 v[47:48], 2, v[23:24]
	v_mad_u64_u32 v[49:50], s[40:41], v49, 6, s[10:11]
	v_addc_co_u32_e32 v52, vcc, 0, v28, vcc
	v_mov_b32_e32 v51, s17
	v_add_co_u32_e32 v47, vcc, s16, v47
	v_addc_co_u32_e32 v48, vcc, v51, v48, vcc
	global_load_dword v47, v[47:48], off offset:12
	v_mov_b32_e32 v48, v50
	v_mad_u64_u32 v[50:51], s[40:41], v52, 6, v[48:49]
	v_add_co_u32_e32 v48, vcc, v45, v42
	v_addc_co_u32_e32 v55, vcc, 0, v28, vcc
	v_mad_u64_u32 v[51:52], s[40:41], v48, 6, s[10:11]
	v_add_co_u32_e32 v48, vcc, v45, v43
	v_mad_u64_u32 v[53:54], s[40:41], v48, 6, s[10:11]
	v_mov_b32_e32 v48, v52
	v_addc_co_u32_e32 v57, vcc, 0, v28, vcc
	v_mad_u64_u32 v[55:56], s[40:41], v55, 6, v[48:49]
	v_mov_b32_e32 v48, v54
	v_mad_u64_u32 v[56:57], s[40:41], v57, 6, v[48:49]
	v_add_co_u32_e32 v48, vcc, v45, v44
	v_mad_u64_u32 v[57:58], s[40:41], v48, 6, s[10:11]
	v_addc_co_u32_e32 v59, vcc, 0, v28, vcc
	v_mov_b32_e32 v48, v58
	v_mov_b32_e32 v52, v55
	v_mov_b32_e32 v54, v56
	v_mad_u64_u32 v[55:56], s[40:41], v59, 6, v[48:49]
	global_load_ubyte v56, v[49:50], off offset:3
	global_load_ubyte v59, v[53:54], off offset:3
	global_load_ubyte v60, v[51:52], off offset:3
	v_cvt_f32_i32_e32 v4, v4
	v_mov_b32_e32 v58, v55
	global_load_ubyte v49, v[57:58], off offset:3
	v_cvt_f32_i32_e32 v3, v3
	v_cvt_f32_i32_e32 v6, v6
	v_cvt_f32_i32_e32 v5, v5
	s_waitcnt vmcnt(4)
	v_mov_b32_e32 v48, v47
	s_waitcnt vmcnt(3)
	v_add_u32_e32 v50, 0xffffff81, v56
	s_waitcnt vmcnt(2)
	v_add_u32_e32 v52, 0xffffff81, v59
	s_waitcnt vmcnt(1)
	v_add_u32_e32 v51, 0xffffff81, v60
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v53, 0xffffff81, v49
	v_ldexp_f32 v49, 0.5, v50
	v_ldexp_f32 v50, 0.5, v51
	v_ldexp_f32 v51, 0.5, v52
	v_ldexp_f32 v52, 0.5, v53
	v_pk_mul_f32 v[49:50], v[47:48], v[49:50]
	v_pk_mul_f32 v[47:48], v[47:48], v[51:52]
	v_pk_fma_f32 v[21:22], v[3:4], v[49:50], v[21:22]
	v_pk_fma_f32 v[25:26], v[5:6], v[47:48], v[25:26]
.LBB3_38:                               ;   in Loop: Header=BB3_3 Depth=1
	s_or_b64 exec, exec, s[8:9]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt lgkmcnt(0)
	s_and_saveexec_b64 s[40:41], s[2:3]
	s_cbranch_execz .LBB3_43
; %bb.39:                               ;   in Loop: Header=BB3_3 Depth=1
	v_add_co_u32_e32 v3, vcc, v27, v19
	v_addc_co_u32_e32 v4, vcc, v46, v20, vcc
	v_mad_u64_u32 v[3:4], s[8:9], s6, 16, v[3:4]
	v_mov_b32_e32 v5, 0xfffff200
	s_mov_b64 s[42:43], 0
	v_lshl_add_u32 v4, s7, 4, v4
	v_mad_u64_u32 v[4:5], s[8:9], s38, v5, v[3:4]
	v_mov_b32_e32 v6, v38
	v_mov_b32_e32 v47, v1
	v_mad_u64_u32 v[3:4], s[8:9], v4, s47, v[15:16]
	s_mul_i32 s8, s39, 0xfffff200
	v_add_u32_e32 v5, s8, v5
	v_subrev_u32_e32 v5, s38, v5
	v_mad_u64_u32 v[4:5], s[8:9], v5, s47, v[4:5]
	v_mov_b32_e32 v5, v37
	s_branch .LBB3_41
.LBB3_40:                               ;   in Loop: Header=BB3_41 Depth=2
	s_or_b64 exec, exec, s[8:9]
	global_load_ubyte v49, v[3:4], off
	v_and_b32_e32 v50, 4, v6
	s_getpc_b64 s[8:9]
	s_add_u32 s8, s8, __const._ZN6i8r1136decodeEh.q@rel32@lo+4
	s_addc_u32 s9, s9, __const._ZN6i8r1136decodeEh.q@rel32@hi+12
	s_waitcnt vmcnt(1)
	ds_write_b8 v47, v48 offset:1088
	v_add_co_u32_e32 v3, vcc, 0xc0, v3
	v_add_u32_e32 v6, 0x100, v6
	v_add_u32_e32 v5, 8, v5
	v_addc_co_u32_e32 v4, vcc, 0, v4, vcc
	s_waitcnt vmcnt(0)
	v_bfe_u32 v51, v49, v50, 3
	v_lshlrev_b32_e32 v51, 2, v51
	global_load_dword v51, v51, s[8:9]
	v_lshrrev_b32_e32 v48, v50, v49
	v_and_b32_e32 v48, 8, v48
	v_cmp_eq_u32_e64 s[8:9], 0, v48
	s_waitcnt vmcnt(0)
	v_sub_u32_e32 v49, 0, v51
	v_cndmask_b32_e64 v48, v49, v51, s[8:9]
	ds_write_b8 v47, v48 offset:1600
	v_add_u32_e32 v47, 64, v47
	v_cmp_lt_u32_e64 s[8:9], s48, v47
	s_or_b64 s[42:43], s[8:9], s[42:43]
	s_andn2_b64 exec, exec, s[42:43]
	s_cbranch_execz .LBB3_43
.LBB3_41:                               ;   Parent Loop BB3_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ds_read_b32 v49, v5
	v_mov_b32_e32 v48, 0
	s_waitcnt lgkmcnt(0)
	v_cmp_lt_i32_e32 vcc, -1, v49
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB3_40
; %bb.42:                               ;   in Loop: Header=BB3_41 Depth=2
	v_mad_u64_u32 v[48:49], s[50:51], v49, s33, 0
	v_mov_b32_e32 v50, s15
	v_or_b32_e32 v48, v48, v41
	v_or_b32_e32 v49, 0, v49
	v_add_co_u32_e32 v48, vcc, s14, v48
	v_addc_co_u32_e32 v49, vcc, v50, v49, vcc
	global_load_ubyte v48, v[48:49], off offset:128
	s_branch .LBB3_40
.LBB3_43:                               ;   in Loop: Header=BB3_3 Depth=1
	s_or_b64 exec, exec, s[40:41]
	v_add_u32_e32 v5, 0x400, v31
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt lgkmcnt(0)
	ds_read2_b32 v[3:4], v5 offset1:1
	ds_read2_b32 v[5:6], v5 offset0:128 offset1:129
	s_waitcnt lgkmcnt(1)
	v_and_b32_e32 v47, 0xff00, v4
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v48, 0xff00, v6
	v_or_b32_sdwa v47, v4, v47 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
	v_or_b32_sdwa v49, v6, v48 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
	v_and_b32_e32 v48, 0xff0000, v4
	v_and_b32_e32 v4, 0xff000000, v4
	v_and_b32_e32 v50, 0xff0000, v6
	v_or3_b32 v48, v47, v48, v4
	v_or3_b32 v47, v3, 0, 0
	v_and_b32_e32 v3, 0xff000000, v6
	v_or3_b32 v50, v49, v50, v3
	v_or3_b32 v49, v5, 0, 0
	v_mov_b32_e32 v3, v2
	v_mov_b32_e32 v4, v2
	v_mov_b32_e32 v5, v2
	v_mov_b32_e32 v6, v5
	v_mov_b32_e32 v5, v4
	v_mov_b32_e32 v4, v3
	v_mov_b32_e32 v3, v2
	s_nop 1
	v_mmac_i32_16x16x32_i8 v[3:6], v[47:48], v[49:50], v[3:6]
	s_and_saveexec_b64 s[8:9], s[4:5]
	s_cbranch_execz .LBB3_45
; %bb.44:                               ;   in Loop: Header=BB3_3 Depth=1
	v_lshlrev_b64 v[47:48], 2, v[23:24]
	v_mov_b32_e32 v49, s17
	v_add_co_u32_e32 v47, vcc, s16, v47
	v_addc_co_u32_e32 v48, vcc, v49, v48, vcc
	v_add_co_u32_e32 v49, vcc, v45, v30
	v_addc_co_u32_e32 v53, vcc, 0, v28, vcc
	v_add_co_u32_e32 v51, vcc, v45, v42
	v_mad_u64_u32 v[49:50], s[40:41], v49, 6, s[10:11]
	v_mad_u64_u32 v[51:52], s[40:41], v51, 6, s[10:11]
	v_addc_co_u32_e32 v55, vcc, 0, v28, vcc
	v_mad_u64_u32 v[53:54], s[40:41], v53, 6, v[50:51]
	v_mov_b32_e32 v50, v52
	v_mad_u64_u32 v[54:55], s[40:41], v55, 6, v[50:51]
	global_load_dword v47, v[47:48], off offset:16
	v_mov_b32_e32 v50, v53
	v_add_co_u32_e32 v48, vcc, v45, v43
	global_load_ubyte v56, v[49:50], off offset:4
	v_mad_u64_u32 v[48:49], s[40:41], v48, 6, s[10:11]
	v_addc_co_u32_e32 v50, vcc, 0, v28, vcc
	v_add_co_u32_e32 v53, vcc, v45, v44
	v_mov_b32_e32 v52, v54
	v_mad_u64_u32 v[53:54], s[40:41], v53, 6, s[10:11]
	v_mad_u64_u32 v[49:50], s[40:41], v50, 6, v[49:50]
	v_addc_co_u32_e32 v55, vcc, 0, v28, vcc
	v_mov_b32_e32 v50, v54
	v_mad_u64_u32 v[54:55], s[40:41], v55, 6, v[50:51]
	global_load_ubyte v50, v[48:49], off offset:4
	global_load_ubyte v55, v[51:52], off offset:4
                                        ; kill: killed $vgpr48 killed $vgpr49
                                        ; kill: killed $vgpr51 killed $vgpr52
	s_nop 0
	global_load_ubyte v49, v[53:54], off offset:4
	v_cvt_f32_i32_e32 v4, v4
	v_cvt_f32_i32_e32 v3, v3
	v_cvt_f32_i32_e32 v6, v6
	v_cvt_f32_i32_e32 v5, v5
	s_waitcnt vmcnt(4)
	v_mov_b32_e32 v48, v47
	s_waitcnt vmcnt(3)
	v_add_u32_e32 v51, 0xffffff81, v56
	s_waitcnt vmcnt(2)
	v_add_u32_e32 v53, 0xffffff81, v50
	s_waitcnt vmcnt(1)
	v_add_u32_e32 v52, 0xffffff81, v55
	v_ldexp_f32 v50, 0.5, v52
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v54, 0xffffff81, v49
	v_ldexp_f32 v49, 0.5, v51
	v_ldexp_f32 v51, 0.5, v53
	v_ldexp_f32 v52, 0.5, v54
	v_pk_mul_f32 v[49:50], v[47:48], v[49:50]
	v_pk_mul_f32 v[47:48], v[47:48], v[51:52]
	v_pk_fma_f32 v[21:22], v[3:4], v[49:50], v[21:22]
	v_pk_fma_f32 v[25:26], v[5:6], v[47:48], v[25:26]
.LBB3_45:                               ;   in Loop: Header=BB3_3 Depth=1
	s_or_b64 exec, exec, s[8:9]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt lgkmcnt(0)
	s_and_saveexec_b64 s[40:41], s[2:3]
	s_cbranch_execz .LBB3_50
; %bb.46:                               ;   in Loop: Header=BB3_3 Depth=1
	v_add_co_u32_e32 v3, vcc, v27, v19
	v_addc_co_u32_e32 v4, vcc, v46, v20, vcc
	v_mad_u64_u32 v[3:4], s[8:9], s6, 16, v[3:4]
	v_mov_b32_e32 v5, 0xfffff200
	v_mov_b32_e32 v6, v38
	v_lshl_add_u32 v4, s7, 4, v4
	v_mad_u64_u32 v[4:5], s[8:9], s38, v5, v[3:4]
	v_mov_b32_e32 v27, v1
	v_mad_u64_u32 v[3:4], s[8:9], v4, s47, v[17:18]
	s_mul_i32 s8, s39, 0xfffff200
	v_add_u32_e32 v5, s8, v5
	v_subrev_u32_e32 v5, s38, v5
	v_mad_u64_u32 v[4:5], s[8:9], v5, s47, v[4:5]
	s_mov_b64 s[38:39], 0
	v_mov_b32_e32 v5, v37
	s_branch .LBB3_48
.LBB3_47:                               ;   in Loop: Header=BB3_48 Depth=2
	s_or_b64 exec, exec, s[8:9]
	global_load_ubyte v47, v[3:4], off
	v_and_b32_e32 v48, 4, v6
	s_getpc_b64 s[8:9]
	s_add_u32 s8, s8, __const._ZN6i8r1136decodeEh.q@rel32@lo+4
	s_addc_u32 s9, s9, __const._ZN6i8r1136decodeEh.q@rel32@hi+12
	s_waitcnt vmcnt(1)
	ds_write_b8 v27, v46 offset:1088
	v_add_co_u32_e32 v3, vcc, 0xc0, v3
	v_add_u32_e32 v6, 0x100, v6
	v_add_u32_e32 v5, 8, v5
	v_addc_co_u32_e32 v4, vcc, 0, v4, vcc
	s_waitcnt vmcnt(0)
	v_bfe_u32 v49, v47, v48, 3
	v_lshlrev_b32_e32 v49, 2, v49
	global_load_dword v49, v49, s[8:9]
	v_lshrrev_b32_e32 v46, v48, v47
	v_and_b32_e32 v46, 8, v46
	v_cmp_eq_u32_e64 s[8:9], 0, v46
	s_waitcnt vmcnt(0)
	v_sub_u32_e32 v47, 0, v49
	v_cndmask_b32_e64 v46, v47, v49, s[8:9]
	ds_write_b8 v27, v46 offset:1600
	v_add_u32_e32 v27, 64, v27
	v_cmp_lt_u32_e64 s[8:9], s48, v27
	s_or_b64 s[38:39], s[8:9], s[38:39]
	s_andn2_b64 exec, exec, s[38:39]
	s_cbranch_execz .LBB3_50
.LBB3_48:                               ;   Parent Loop BB3_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ds_read_b32 v47, v5
	v_mov_b32_e32 v46, 0
	s_waitcnt lgkmcnt(0)
	v_cmp_lt_i32_e32 vcc, -1, v47
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB3_47
; %bb.49:                               ;   in Loop: Header=BB3_48 Depth=2
	v_mad_u64_u32 v[46:47], s[42:43], v47, s33, 0
	v_mov_b32_e32 v48, s15
	v_or_b32_e32 v46, v46, v41
	v_or_b32_e32 v47, 0, v47
	v_add_co_u32_e32 v46, vcc, s14, v46
	v_addc_co_u32_e32 v47, vcc, v48, v47, vcc
	global_load_ubyte v46, v[46:47], off offset:160
	s_branch .LBB3_47
.LBB3_50:                               ;   in Loop: Header=BB3_3 Depth=1
	s_or_b64 exec, exec, s[40:41]
	v_add_u32_e32 v5, 0x400, v31
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt lgkmcnt(0)
	ds_read2_b32 v[3:4], v5 offset1:1
	ds_read2_b32 v[5:6], v5 offset0:128 offset1:129
	s_waitcnt lgkmcnt(1)
	v_and_b32_e32 v27, 0xff00, v4
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v46, 0xff00, v6
	v_or_b32_sdwa v27, v4, v27 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
	v_or_b32_sdwa v48, v6, v46 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
	v_and_b32_e32 v46, 0xff0000, v4
	v_and_b32_e32 v4, 0xff000000, v4
	v_and_b32_e32 v49, 0xff0000, v6
	v_or3_b32 v47, v27, v46, v4
	v_or3_b32 v46, v3, 0, 0
	v_and_b32_e32 v3, 0xff000000, v6
	v_or3_b32 v49, v48, v49, v3
	v_or3_b32 v48, v5, 0, 0
	v_mov_b32_e32 v3, v2
	v_mov_b32_e32 v4, v2
	v_mov_b32_e32 v5, v2
	v_mov_b32_e32 v6, v5
	v_mov_b32_e32 v5, v4
	v_mov_b32_e32 v4, v3
	v_mov_b32_e32 v3, v2
	s_nop 1
	v_mmac_i32_16x16x32_i8 v[3:6], v[46:47], v[48:49], v[3:6]
	s_and_saveexec_b64 s[8:9], s[4:5]
	s_cbranch_execz .LBB3_52
; %bb.51:                               ;   in Loop: Header=BB3_3 Depth=1
	v_add_co_u32_e32 v46, vcc, v45, v30
	v_lshlrev_b64 v[23:24], 2, v[23:24]
	v_mad_u64_u32 v[46:47], s[38:39], v46, 6, s[10:11]
	v_addc_co_u32_e32 v48, vcc, 0, v28, vcc
	v_mov_b32_e32 v27, s17
	v_add_co_u32_e32 v23, vcc, s16, v23
	v_addc_co_u32_e32 v24, vcc, v27, v24, vcc
	global_load_dword v23, v[23:24], off offset:20
	v_mov_b32_e32 v24, v47
	v_mad_u64_u32 v[47:48], s[38:39], v48, 6, v[24:25]
	v_add_co_u32_e32 v24, vcc, v45, v42
	v_addc_co_u32_e32 v27, vcc, 0, v28, vcc
	v_mad_u64_u32 v[48:49], s[38:39], v24, 6, s[10:11]
	v_add_co_u32_e32 v24, vcc, v45, v43
	v_mad_u64_u32 v[50:51], s[38:39], v24, 6, s[10:11]
	v_mov_b32_e32 v24, v49
	v_addc_co_u32_e32 v54, vcc, 0, v28, vcc
	v_mad_u64_u32 v[52:53], s[38:39], v27, 6, v[24:25]
	v_mov_b32_e32 v24, v51
	v_mad_u64_u32 v[53:54], s[38:39], v54, 6, v[24:25]
	v_add_co_u32_e32 v24, vcc, v45, v44
	v_addc_co_u32_e32 v45, vcc, 0, v28, vcc
	v_mad_u64_u32 v[27:28], s[38:39], v24, 6, s[10:11]
	v_mov_b32_e32 v49, v52
	v_mov_b32_e32 v51, v53
	v_mov_b32_e32 v24, v28
	v_mad_u64_u32 v[52:53], s[38:39], v45, 6, v[24:25]
	global_load_ubyte v45, v[46:47], off offset:5
	global_load_ubyte v53, v[50:51], off offset:5
	global_load_ubyte v54, v[48:49], off offset:5
	v_cvt_f32_i32_e32 v4, v4
	v_mov_b32_e32 v28, v52
	global_load_ubyte v27, v[27:28], off offset:5
	v_cvt_f32_i32_e32 v3, v3
	v_cvt_f32_i32_e32 v6, v6
	v_cvt_f32_i32_e32 v5, v5
	s_waitcnt vmcnt(4)
	v_mov_b32_e32 v24, v23
	s_waitcnt vmcnt(3)
	v_add_u32_e32 v28, 0xffffff81, v45
	s_waitcnt vmcnt(2)
	v_add_u32_e32 v46, 0xffffff81, v53
	s_waitcnt vmcnt(1)
	v_add_u32_e32 v45, 0xffffff81, v54
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v47, 0xffffff81, v27
	v_ldexp_f32 v27, 0.5, v28
	v_ldexp_f32 v28, 0.5, v45
	v_ldexp_f32 v45, 0.5, v46
	v_ldexp_f32 v46, 0.5, v47
	v_pk_mul_f32 v[27:28], v[23:24], v[27:28]
	v_pk_mul_f32 v[23:24], v[23:24], v[45:46]
	v_pk_fma_f32 v[21:22], v[3:4], v[27:28], v[21:22]
	v_pk_fma_f32 v[25:26], v[5:6], v[23:24], v[25:26]
.LBB3_52:                               ;   in Loop: Header=BB3_3 Depth=1
	s_or_b64 exec, exec, s[8:9]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt lgkmcnt(0)
	s_and_saveexec_b64 s[8:9], s[4:5]
	s_cbranch_execz .LBB3_54
; %bb.53:                               ;   in Loop: Header=BB3_3 Depth=1
	ds_write2_b32 v32, v21, v22 offset1:4
	ds_write2_b32 v32, v25, v26 offset0:8 offset1:12
.LBB3_54:                               ;   in Loop: Header=BB3_3 Depth=1
	s_or_b64 exec, exec, s[8:9]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt lgkmcnt(0)
	ds_read_b32 v3, v33
	s_waitcnt lgkmcnt(0)
	v_cmp_lt_i32_e32 vcc, -1, v3
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB3_56
; %bb.55:                               ;   in Loop: Header=BB3_3 Depth=1
	v_mov_b64_e32 v[4:5], s[36:37]
	v_mad_u64_u32 v[3:4], s[8:9], v3, s46, v[4:5]
	ds_read_b32 v5, v38
	v_mov_b32_e32 v6, s13
	v_add_co_u32_e32 v3, vcc, v3, v29
	v_addc_co_u32_e32 v4, vcc, 0, v4, vcc
	v_lshlrev_b64 v[3:4], 2, v[3:4]
	v_add_co_u32_e32 v3, vcc, s12, v3
	v_addc_co_u32_e32 v4, vcc, v6, v4, vcc
	s_waitcnt lgkmcnt(0)
	global_store_dword v[3:4], v5, off
.LBB3_56:                               ;   in Loop: Header=BB3_3 Depth=1
	s_or_b64 exec, exec, s[4:5]
	ds_read_b32 v3, v34
	s_waitcnt lgkmcnt(0)
	v_cmp_lt_i32_e32 vcc, -1, v3
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB3_58
; %bb.57:                               ;   in Loop: Header=BB3_3 Depth=1
	v_mov_b64_e32 v[4:5], s[36:37]
	v_mad_u64_u32 v[3:4], s[8:9], v3, s46, v[4:5]
	ds_read_b32 v5, v38 offset:256
	v_mov_b32_e32 v6, s13
	v_add_co_u32_e32 v3, vcc, v3, v29
	v_addc_co_u32_e32 v4, vcc, 0, v4, vcc
	v_lshlrev_b64 v[3:4], 2, v[3:4]
	v_add_co_u32_e32 v3, vcc, s12, v3
	v_addc_co_u32_e32 v4, vcc, v6, v4, vcc
	s_waitcnt lgkmcnt(0)
	global_store_dword v[3:4], v5, off
.LBB3_58:                               ;   in Loop: Header=BB3_3 Depth=1
	s_or_b64 exec, exec, s[4:5]
	ds_read_b32 v3, v35
	s_waitcnt lgkmcnt(0)
	v_cmp_lt_i32_e32 vcc, -1, v3
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB3_60
; %bb.59:                               ;   in Loop: Header=BB3_3 Depth=1
	v_mov_b64_e32 v[4:5], s[36:37]
	v_mad_u64_u32 v[3:4], s[8:9], v3, s46, v[4:5]
	ds_read_b32 v5, v38 offset:512
	v_mov_b32_e32 v6, s13
	v_add_co_u32_e32 v3, vcc, v3, v29
	v_addc_co_u32_e32 v4, vcc, 0, v4, vcc
	v_lshlrev_b64 v[3:4], 2, v[3:4]
	v_add_co_u32_e32 v3, vcc, s12, v3
	v_addc_co_u32_e32 v4, vcc, v6, v4, vcc
	s_waitcnt lgkmcnt(0)
	global_store_dword v[3:4], v5, off
.LBB3_60:                               ;   in Loop: Header=BB3_3 Depth=1
	s_or_b64 exec, exec, s[4:5]
	ds_read_b32 v3, v36
	s_waitcnt lgkmcnt(0)
	v_cmp_lt_i32_e32 vcc, -1, v3
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB3_1
; %bb.61:                               ;   in Loop: Header=BB3_3 Depth=1
	v_mov_b64_e32 v[4:5], s[36:37]
	v_mad_u64_u32 v[3:4], s[8:9], v3, s46, v[4:5]
	ds_read_b32 v5, v38 offset:768
	v_mov_b32_e32 v6, s13
	v_add_co_u32_e32 v3, vcc, v3, v29
	v_addc_co_u32_e32 v4, vcc, 0, v4, vcc
	v_lshlrev_b64 v[3:4], 2, v[3:4]
	v_add_co_u32_e32 v3, vcc, s12, v3
	v_addc_co_u32_e32 v4, vcc, v6, v4, vcc
	s_waitcnt lgkmcnt(0)
	global_store_dword v[3:4], v5, off
	s_branch .LBB3_1
.LBB3_62:
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6, 0x0
	.amdhsa_kernel _ZN6i8r1136matrixEPfPKaPKfPKiS6_S6_S6_S6_PKhS8_m
		.amdhsa_group_segment_fixed_size 2112
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 344
		.amdhsa_user_sgpr_count 6
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 61
		.amdhsa_next_free_sgpr 52
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end3:
	.size	_ZN6i8r1136matrixEPfPKaPKfPKiS6_S6_S6_S6_PKhS8_m, .Lfunc_end3-_ZN6i8r1136matrixEPfPKaPKfPKiS6_S6_S6_S6_PKhS8_m
                                        ; -- End function
