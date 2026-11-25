"""Prepare the exact R67 and R68 replacements on f13r3n09 and f12r3n15."""

import ast
from hashlib import sha256
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = Path(__file__).resolve()
STEM = 'training80-r81-r67-r68-two-guarded-bf16-v1'
CLAIM = ROOT / 'coordination/tasks' / f'{STEM}.json'
R80 = ('tools/run_training80_r80_r64_r66_five_guarded_successor_once.py',
       '3dbcc9574ddc81eb6a64d9a8056eab2c6e7374999b1dda13aba489a37f18b4ac')
R67 = ('tools/run_training80_r67_n09_0419_guarded_successor_once.py',
       'c3383ac53803d66a3b66edc04f49816a87a6f1117a77e7c58aafeec3ffb4bd83')
R68 = ('tools/run_training80_r68_r54_pair_0427_guarded_successor_once.py',
       '4de30666274739909b78070cc2e07480878f06bf5024977a86d60cea86848329')
FLEET = ('local-only/training80-after-t07-v2-original-fresh-numbered-scheduler-88638a41df0e421a81779f527b25ef05/receipt.json',
         'ddc826fac885508691445c55c476cc7da15938b4acc6fa130ddc728ca1bf4599')
R67_FAMILY = 'training80-r67-n09-0419-guarded-bf16-v1'
R68_FAMILY = 'training80-r68-r54-pair-0427-guarded-bf16-v1'
FRONTIERS = {
    'f13r3n09': ('1194900.1572','1194900.1571',R67_FAMILY,'efe6be4fa21748998a6fde1a1e817eee',
                 '0b87f11ba2f94bdacfee1cdbd6dfa286a6c732c972d9adbcf8a48631fb31664e',
                 '986c095cc3e0a5d235d3f72a15a9354a1d928f64ea4f0a1d341de9c184854e71'),
    'f12r3n15': ('1194900.1574','1194900.1573',R68_FAMILY,'2629e1e8da0a49c08338580f7c01a1b0',
                 '0e9ee0ea143cb949068af6bd09a7894d6797357431faf6c5245f1288bac69b65',
                 '61ca12af420905dba3f6bf12ad98a0ef3b24e1dc88721659d34e6c4f6193adab'),
}


def entry():
    path=ROOT/R80[0]
    raw=path.read_bytes()
    if path.resolve(strict=True)!=path or sha256(raw).hexdigest()!=R80[1]:
        raise ValueError('exact R80 ordinary runtime differs')
    base=runpy.run_path(str(path))
    require=base['require']
    protected=base['PROTECTED']|set(base['FRONTIERS'])|{'f13r2n07','f13r2n08','f13r3n13','f13r2n19'}
    require(set(FRONTIERS).isdisjoint(protected),'R80, R79 or current protected nodes overlap')
    namespace=dict(base,STEM=STEM,WRAPPER=WRAPPER,CLAIM=CLAIM,R80=R80,R64=R67,R66=R68,
                   R64_FAMILY=R67_FAMILY,R66_FAMILY=R68_FAMILY,FLEET=FLEET,FRONTIERS=FRONTIERS,
                   PROTECTED=protected,__file__=__file__,__doc__=__doc__)
    text=raw.decode()
    tree=ast.parse(text)

    def adapt(name, replacements):
        node=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name==name)
        source=ast.get_source_segment(text,node)
        for before,after in replacements:
            require(source.count(before)==1,'frozen R80 source boundary differs')
            source=source.replace(before,after,1)
        exec(compile(source,f'<R81 inherited {name}>','exec'),namespace)

    adapt('load',(
        ("    snap=fleet['response']['snapshot']",
         "    snap=fleet['response']\n"
         "    parent_rows=snap['parents']\n"
         "    require(snap['allocated_DCU']==640 and snap['distinct_nodes']==80\n"
         "            and all(v['state']=='RUNNING' and v['allocated_DCU']==320 for v in parent_rows.values()),\n"
         "            'the two exact parent allocations differ')"),
        ("snap['numbered_cards']==320","snap['running_numbered_card_bounds']==[304,304]"),
        ("node in snap['no_numbered'][parent]",
         "node in parent_rows[parent]['nodes_without_running_DCU_step']\n"
         "                and node not in parent_rows[parent]['unknown_DCU_allocation_nodes']"),
        ('57b51ef5-4c3a-4f47-94bd-6c6443160b6a','a9173967-635c-4e70-b44f-f9c8f1c815d0'),
        ("})==5, 'new request identity collision')","})==2, 'new request identity collision')"),
        ('source_pins=dict((R44,R64,R66))','source_pins=dict((R44,R64,R66,R80))'),
        ('current formal work owns an R80 target','current formal work owns an R81 target'),
    ))
    adapt('runtime',(
        ('exact-published-R80-revision','exact-published-R81-revision'),
        ('for spec in (R44,R64,R66):','for spec in (R44,R64,R66,R80):'),
    ))
    adapt('main',(
        ('frozen five-node scope differs','frozen two-node scope differs'),
        ('prior_cgroup_CPU_vetoes=10,','prior_cgroup_CPU_vetoes=2*len(rows),'),
        ('runtime_budget_vetoes=10,distinct_publication_reader_fixtures=5,',
         'runtime_budget_vetoes=2*len(rows),distinct_publication_reader_fixtures=len(rows),'),
    ))
    return namespace['load'],namespace['runtime'],namespace['main']


if __name__=='__main__':
    _,_,main=entry()
    main()
