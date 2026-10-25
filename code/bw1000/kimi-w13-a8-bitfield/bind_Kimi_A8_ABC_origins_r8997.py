import pathlib,ast
r=pathlib.Path.cwd();p=r/'Kimi-W13-A8-bitfield-private-r8996';f=p/'Kimi_A8_ABC_real_r8996.py';s=f.read_text();start=s.index("    tablelib=ctypes.CDLL");end=s.index("    x=torch.from_numpy(xbits)",start);block=s[start:end];s=s[:start]+s[end:]
block+='''    class DlInfo(ctypes.Structure):
        _fields_=[('dli_fname',ctypes.c_char_p),('dli_fbase',ctypes.c_void_p),('dli_sname',ctypes.c_char_p),('dli_saddr',ctypes.c_void_p)]
    dl=ctypes.CDLL('libdl.so.2');dl.dladdr.argtypes=[ctypes.c_void_p,ctypes.POINTER(DlInfo)];dl.dladdr.restype=ctypes.c_int
    binding={}
    for name,lib,expected in [('B_957_table',tablelib,m['table_native']),('C_bitfield',bitlib,m['native'])]:
        binding[name]={}
        for symbol in ['modern_kimi_A8_W13_r8993','kimi_A8_HIP_matrix_r8993']:
            info=DlInfo();address=ctypes.cast(getattr(lib,symbol),ctypes.c_void_p).value;assert dl.dladdr(address,ctypes.byref(info))!=0
            origin=str(pathlib.Path(info.dli_fname.decode()).resolve());assert origin==str(pathlib.Path(expected).resolve()),('DSO_origin_mismatch',name,symbol,origin,expected)
            binding[name][symbol]={'origin':origin,'base':int(info.dli_fbase),'address':address}
        assert len({v['base']for v in binding[name].values()})==1
    assert binding['B_957_table']['modern_kimi_A8_W13_r8993']['base']!=binding['C_bitfield']['modern_kimi_A8_W13_r8993']['base'],'DSO_base_collision'
'''
needle="    xbits,ids_host,rw_host=fixture";assert needle in s;s=s.replace(needle,block+needle,1)
s=s.replace("'qualification':{},'stream_ABI_scope'","'qualification':{},'actual_DSO_origins':binding,'stream_ABI_scope'")
s=s.replace(";result['qualification'][arms[arm]]=d\n        assert", ";result['qualification'][arms[arm]]=d;save()\n        assert")
s=s.replace("            d['device_error_flag']=int(extra[475392:475396].view(torch.int32).item());save()", "            d['device_error_flag']=int(extra[475392:475396].view(torch.int32).item());d['canary']=bool(torch.all(extra[475648:]==0x5a));save()\n            assert d['canary'],'first_canary_failure'")
assert s.index('class DlInfo')<s.index('xbits,ids_host,rw_host=fixture');ast.parse(s);f.write_bytes(s.encode());print('4dladdr origins + distinctbases; before weightsload')
