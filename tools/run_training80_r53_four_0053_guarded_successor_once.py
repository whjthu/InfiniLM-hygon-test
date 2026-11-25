"""Renew four proven ordinary originals, preserving the separately closed R52 attempt."""

import ast
from hashlib import sha256
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
STEM = 'training80-r53-four-0053-guarded-bf16-v1'
WRAPPER = ROOT/'tools/run_training80_r53_four_0053_guarded_successor_once.py'
CLAIM = ROOT/'coordination/tasks'/f'{STEM}.json'
R49 = ('tools/run_training80_r49_four_2334_guarded_successor_once.py',
       'bd9585250012781d8aff74fd44a6054f674f51eeaa99ec9480f21f0d5ecef505')
R52 = ('tools/run_training80_r52_six_0036_guarded_successor_once.py',
       'dd5b2638110c4dba756ffdedde23ef8ae08b98be1b7391e7dd9589d779a5d37a')
READER = ('local-only/read_training80_r53_0053_four_ordinary_originals_once.py',
          'e4d40c7d47637d9e45c407f889caad18acc14f39c3a9929a0dca6d40eb5174db')
ORIGINALS = {
    '1103606': ('local-only/training80-r53-0053-four-ordinary-originals-1103606-5dfaca60b6294b5588524f6aa105e9ae.json',
                'cee9475512cb4656fdae5496f7c3afb4dea50ef26adbdb609e8d7c2710babede'),
    '1194900': ('local-only/training80-r53-0053-four-ordinary-originals-1194900-6e696809d05d4a63a8d0a206296598b4.json',
                '7813c893ad0720d34e14fa7731af41267977b33293a8c062562724126f765b17'),
}
ORDER = ('f13r2n05','f13r1n07','f13r2n11','f12r3n02')
UNSENT = {'f12r3n02'}
FENCES = {'f13r1n11','f13r2n10','f13r1n12','f13r3n13','f13r1n19','f12r3n15',
          'f13r1n02','f12r2n10','f12r2n11','f13r2n03','f13r2n07'}


def require(condition,message):
    if not condition:
        raise ValueError(message)


def pin(spec):
    path=ROOT/spec[0]
    require(path.is_relative_to(ROOT) and path.resolve(strict=True)==path,'frozen source path differs')
    raw=path.read_bytes()
    require(0<len(raw)<=262144 and sha256(raw).hexdigest()==spec[1],'frozen source bytes differ')
    return raw


def entry():
    disk=pin(R49).decode()
    r49=runpy.run_path(str(ROOT/R49[0]))
    pin(R52)
    r52=runpy.run_path(str(ROOT/R52[0]))
    require(set(ORDER)&set(r52['ORDER'])==UNSENT and set(ORDER).isdisjoint(FENCES),
            'only exact f12r3n02 earlier no-send may overlap R52')
    original=next(node for node in ast.parse(disk).body if isinstance(node,ast.FunctionDef) and node.name=='entry')
    source=ast.get_source_segment(disk,original)
    env=dict(r49,STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,READER=READER,ORIGINALS=ORIGINALS,
             ORDER=ORDER,UNSENT=UNSENT,FENCES=FENCES,R49=R49,R52=R52,__file__=__file__)
    edits=(
        ("    require(set(ORDER)&set(p47['ORDER'])=={'f12r3n04'} and set(ORDER)&set(p48['ORDER'])=={'f13r1n20'}\n"
         "            and set(ORDER).isdisjoint(FENCES), 'only the two positively closed earlier attempts may overlap')",
         "    require(not set(ORDER)&set(p47['ORDER']) and not set(ORDER)&set(p48['ORDER'])\n"
         "            and set(ORDER).isdisjoint(FENCES), 'R47/R48 or protected targets overlap')"),
        ('R35=R35,UNSENT=UNSENT,','R35=R35,R49=R49,R52=R52,UNSENT=UNSENT,'),
        ('''        "    prior_routes = module(R40)['load']()[6][5]\\n"
        "    _, load_r35, _ = module(R35)['exact_api']()\\n"
        "    require('f13r4n05' not in prior_routes, 'special source ordinary route unexpectedly collides')\\n"
        "    prior_routes = dict(prior_routes,f13r4n05=load_r35()[4][5]['f13r4n05'])\\n"''',
         '''        "    exact_46 = module(('tools/run_training80_r46_seven_2137_guarded_successor_once.py',\\n"
         "                       '437210faa80d64d792e6b6f4dabb33e54329631af128b342764b85c3fd48f1e4'))['load']()[2]\\n"
         "    exact_45 = previous['load']()\\n"
         "    prior_routes = dict(exact_46[5][5],f12r3n02=exact_45[5][5]['f12r3n02'])\\n"
         "    original_workers = dict(exact_46[-3],f12r3n02=exact_45[-3]['f12r3n02'])\\n"'''),
        ("require(prior_routes[node]['parent']==parent and old['known_launcher']['worker_seconds']==seconds,",
         "require(prior_routes[node]['parent']==parent and old['known_launcher']['worker_seconds']==seconds "
         "and original_workers[node]['new_request']==request and original_workers[node]['renewal_seconds']==seconds,"),
        ('605278d8-11d3-4902-a8b0-b04cd514c35a','abfcb4a4-851a-48f6-a26c-ba3ddd99e88d'),
        ('(PREVIOUS,R48,R47,R40,R35)','(PREVIOUS,R48,R49,R52)'),
        ('for spec in (R48,r47,r40,r46,R35):','for spec in (R48,R49,R52,r46):'),
        ('locked_no_send_positive=2,locked_no_send_vetoes=10,local_no_send_token_positive=2,local_no_send_token_vetoes=2',
         'locked_no_send_positive=1,locked_no_send_vetoes=5,local_no_send_token_positive=1,local_no_send_token_vetoes=1'),
    )
    for old,new in edits:
        require(source.count(old)==1,'verified no-send owner adapter boundary differs')
        source=source.replace(old,new,1)
    exec(compile(source,'<exact new four-node owner and no-send guard>','exec'),env)
    return env['entry']()


if __name__=='__main__':
    _,_,main=entry()
    main()
