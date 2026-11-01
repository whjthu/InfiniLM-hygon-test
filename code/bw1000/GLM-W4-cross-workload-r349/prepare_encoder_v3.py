"""Keep671 failure; use distinct guards as the passed CPU272 did."""
import pathlib
D=pathlib.Path(__file__).parent
s=(D/'CPU_encode_fixtures_v2.py').read_text()
s=s.replace(" def forbidden(*a,**k):raise RuntimeError('GPU API forbidden in CPU encoding')\n for name in ['_lazy_init','init','set_device','current_device']:setattr(torch.cuda,name,forbidden)", " def guard(name):\n  def unique(*a,**k):raise RuntimeError('GPU API forbidden in CPU encoding: '+name)\n  return unique\n for name in ['_lazy_init','init','set_device','current_device']:setattr(torch.cuda,name,guard(name))")
s=s.replace('GLM349-official-encoding-v2-','GLM349-official-encoding-v3-')
(D/'CPU_encode_fixtures_v3.py').write_bytes(s.encode());compile(s,'CPU_encode_fixtures_v3.py','exec')
(D/'CPU_encode_inside_v3.sh').write_bytes((D/'CPU_encode_inside_v2.sh').read_bytes().replace(b'_v2',b'_v3'))
(D/'prepare_encoder_upload_v3.py').write_bytes((D/'prepare_encoder_upload_v2.py').read_bytes().replace(b'_v2',b'_v3').replace(b'-v2',b'-v3'))
