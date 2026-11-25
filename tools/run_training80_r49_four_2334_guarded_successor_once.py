"""Renew the four passed ordinary workers with a new fixed request per node."""

import ast
from hashlib import sha256
import os
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / 'local-only'
STEM = 'training80-r49-four-2334-guarded-bf16-v1'
WRAPPER = ROOT / 'tools/run_training80_r49_four_2334_guarded_successor_once.py'
CLAIM = ROOT / 'coordination/tasks' / f'{STEM}.json'
R48 = ('tools/run_training80_r48_thirteen_2317_guarded_successor_once.py',
       '25fa7c92430b764838f23fedf94544ddcf4d1bfbe7c11c25f0917d085be702d2')
R35 = ('tools/run_training80_r35_following_four_guarded_successor_once.py',
       'a9e149ccc1e0aac5265fc8ee105a5703186c2c62a00ab7ee6454392008567b6c')
READER = ('local-only/read_training80_r49_2334_oldest_ordinary_originals_once.py',
          'f62560be183a421e6d5d1771420cdca9942565852d53f7ca4bde01e1f959a374')
ORIGINALS = {
    '1194900': ('local-only/training80-r49-2334-oldest-ordinary-originals-1194900-d34435aec8f74b298c247040ffbcc370.json',
                '5568c423be2a4aa289bdca3c7c0229c2b20ee49f629b1527741df1dd819f715c'),
    '1103606': ('local-only/training80-r49-2334-oldest-ordinary-originals-1103606-63bee3ae700b48718b3029b626d3e580.json',
                '2b8052a937c9f451e3e85aac6d16207fcbed582375307fab0dc1e0b3a06e83c1'),
}
ORDER = ('f14r3n05','f12r3n04','f13r1n20','f13r4n05')
UNSENT = {'f12r3n04','f13r1n20'}
FENCES = {'f13r1n12','f13r1n19','f13r2n03','f13r2n07','f13r3n09','f13r4n14','f12r2n10','f12r2n11','f13r3n13'}


def require(condition,message):
    if not condition:
        raise ValueError(message)


def pin(spec):
    path=ROOT/spec[0]
    require(path.is_relative_to(ROOT) and path.resolve(strict=True)==path,'frozen evidence path differs')
    raw=path.read_bytes()
    require(0<len(raw)<=262144 and sha256(raw).hexdigest()==spec[1],'frozen evidence bytes differ')
    return raw


def prior_token_absent(previous, exists=os.path.lexists):
    path=LOCAL/f"{previous['scope']}-send.json"
    require(path.parent==LOCAL and not exists(path),'sealed old sender token became visible')


def entry():
    pin(R48)
    p48=runpy.run_path(str(ROOT/R48[0]))
    r47=p48['R47']
    pin(r47)
    p47=runpy.run_path(str(ROOT/r47[0]))
    r46=p47['R46']
    frozen_bytes=pin(r46)
    frozen=runpy.run_path(str(ROOT/r46[0]))
    r40=p48['R40']
    require(set(ORDER)&set(p47['ORDER'])=={'f12r3n04'} and set(ORDER)&set(p48['ORDER'])=={'f13r1n20'}
            and set(ORDER).isdisjoint(FENCES), 'only the two positively closed earlier attempts may overlap')
    scope=dict(frozen,STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,READER=READER,ORIGINALS=ORIGINALS,
               ORDER=ORDER,FENCES=FENCES,R48=R48,R40=r40,R47=r47,R35=R35,UNSENT=UNSENT,
               prior_token_absent=prior_token_absent,__file__=__file__)
    raw=frozen_bytes.decode()
    tree=ast.parse(raw)

    def extract(name):
        f=next(item for item in tree.body if isinstance(item,ast.FunctionDef) and item.name==name)
        return ast.get_source_segment(raw,f)

    def adapt(fn,changes):
        for old,new in changes:
            require(fn.count(old)==1,'verified guard adapter boundary differs')
            fn=fn.replace(old,new,1)
        return fn

    preserved=extract('load')
    start=preserved.index('    _, load_following, _')
    end=preserved.index('    controller = budget',start)
    preserved=preserved[:start]+(
        "    prior_routes = module(R40)['load']()[6][5]\n"
        "    _, load_r35, _ = module(R35)['exact_api']()\n"
        "    require('f13r4n05' not in prior_routes, 'special source ordinary route unexpectedly collides')\n"
        "    prior_routes = dict(prior_routes,f13r4n05=load_r35()[4][5]['f13r4n05'])\n"
        "    require(set(ORDER)<=set(prior_routes) and all(\n"
        "        {'ledger_old','ledger_request','published_group','parent'}<=set(prior_routes[n]) for n in ORDER),\n"
        "        'existing exact ordinary58 owner lineage does not cover all four')\n"
        "    context = (*context[:5],{n:prior_routes[n] for n in ORDER},context[6])\n"
    )+preserved[end:]
    shadow_fields=(
        "            shadow_proof=old['prior_transport_not_sent']\n"
        "            if node in UNSENT:\n"
        "                require(shadow_proof and files['cgroup']==old['cgroup_SHA256'] and proof.get('previous_unsent_attempt')==dict(\n"
        "                    request=shadow_proof['request'],prior_reply_SHA256=shadow_proof['reply_SHA256'],\n"
        "                    prior_intent_SHA256=shadow_proof['intent_SHA256'],gpu_label_ids=[],cpu_label_ids=[],\n"
        "                    gpu_namespace_exists=False,control_namespace_exists=False,history_and_namespaces_empty=True),\n"
        "                    'original shadow labels, namespace, cgroup or positive no-send is no longer closed')\n"
        "                prior_token_absent(shadow_proof)\n"
        "                shadows[node]=shadow_proof\n"
        "            else:\n"
        "                require(shadow_proof is None and 'previous_unsent_attempt' not in proof,'unexpected earlier send')\n"
        "            require(prior_routes[node]['parent']==parent and old['known_launcher']['worker_seconds']==seconds,")
    preserved=adapt(preserved,(
        ("    read_core,read_base,inputs = reader['load']()",
         "    _,read_core,read_base,by_node = reader['load']()\n    inputs = [by_node[n] for n in ORDER]"),
        ("    rows, targets = {}, {}","    rows, targets, shadows = {}, {}, {}"),
        ("            old_route = warning[-2] if node in SPECIAL else last[-3] if node=='f13r2n11' else following[-2]\n"
         "            require(old_route[node]['new_request']==request and old_route[node]['renewal_seconds']==seconds,",shadow_fields),
        ("c90b7f77-0127-48da-9f70-395fdb980bd3","605278d8-11d3-4902-a8b0-b04cd514c35a"),
        ("require(new!=request and 10500<=renewal<=10800", "require(new not in {request,(shadow_proof or {}).get('request')} and 10500<=renewal<=10800"),
        ("queue_ownership='R34_SPECIALIST' if node in SPECIAL else 'ADOPTED_ORDINARY58_EXACT_ANCESTOR')",
         "queue_ownership='ADOPTED_ORDINARY58_EXACT_ANCESTOR',closed_prior_transport=shadow_proof)"),
        ("set(rows)==set(ORDER) and len({v['new_request'] for v in rows.values()})==7,'fixed seven exclusive new requests differ'",
         "set(rows)==set(ORDER) and len({v['new_request'] for v in rows.values()})==4 and set(shadows)==UNSENT,'four fixed distinct successor requests differ'"),
        ("(PREVIOUS,FOLLOWING,WARNING,LAST40)","(PREVIOUS,R48,R47,R40,R35)"),
        ("return previous, warning, (budget,first,r34,early,n02,context,rows,{},claim)",
         "return previous, None, (budget,first,r34,early,n02,context,rows,shadows,claim)"),
    ))
    exec(compile(preserved,'<four exact original terminal and older no-send proofs>','exec'),scope)

    def runtime(previous,_unused,material,claim_sha):
        clone=material[1]['_fn']
        common=clone(previous['runtime'],STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,__file__=__file__)
        context,source,read,render,dispatch,observe=common(material,claim_sha)
        original_pub=dispatch.__globals__['publication']
        original_fence=dispatch.__globals__['exact_no_other_claims']
        def publication(actual,revision):
            original_pub(actual,revision)
            _,publisher=actual[3][-2]['modules']()
            for spec in (R48,r47,r40,r46,R35):
                require(publisher['git']('cat-file','blob',revision+':'+spec[0])==pin(spec),
                        'published exact accepted route or guard differs')
        def fence(ordinary,authority,row):
            original_fence(ordinary,authority,row)
            if row['node'] in material[7]:
                prior_token_absent(material[7][row['node']])
        return context,source,read,render,clone(dispatch,publication=publication,exact_no_other_claims=fence),observe

    scope['runtime']=runtime
    main=adapt(extract('main'),(
        ('seven authorized originals','four authorized originals'),
        ('                cases[node] = dict(request=', '''                if node in material[7]:
                    shadow=material[7][node]
                    previous['shadow_fixture'](rendered['launch'],row,shadow)
                    prior_token_absent(shadow,exists=lambda path:False)
                    try:
                        prior_token_absent(shadow,exists=lambda path:True)
                    except ValueError:
                        pass
                    else:
                        raise ValueError('visible exact old local send token accepted')
                cases[node] = dict(request='''),
        ('locked_cgroup_CPU_positive=7,locked_cgroup_CPU_vetoes=14',
         'locked_cgroup_CPU_positive=len(rows),locked_cgroup_CPU_vetoes=2*len(rows)'),
        ('HOST_spawn_positive=7,HOST_spawn_vetoes=14,SSH=0',
         'HOST_spawn_positive=len(rows),HOST_spawn_vetoes=2*len(rows),locked_no_send_positive=2,locked_no_send_vetoes=10,local_no_send_token_positive=2,local_no_send_token_vetoes=2,SSH=0'),
    ))
    exec(compile(main,'<four one-shot guards and actual local fixtures>','exec'),scope)
    return scope['load'],runtime,scope['main']


if __name__=='__main__':
    _,_,main=entry()
    main()
