import pathlib
p=pathlib.Path('verify_Deep_C128_component_r8695.py');s=p.read_text();a="assert x['arm']==arms[x['trial']]and x['next_past']==x['past']+1 and x['warmup']==(x['trial']<2)";assert a in s;s=s.replace(a,a+"\n  assert 0<x['native_inner_wall_s']<=x['native_enclosing_wall_s'] and abs(x['outer_minus_inner_s']-(x['native_enclosing_wall_s']-x['native_inner_wall_s']))<1e-9")
p.write_text(s,newline='\n')
